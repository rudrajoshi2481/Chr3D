# Copyright (c) 2026 Rudhra Joshi and Yong Chen
# Licensed under the MIT License. See LICENSE in the project root for details.
# This software was developed with support from the National Science Foundation under CAREER Award DBI-2239350.
"""
GraphSAGE-based unsupervised clustering for scHi-C data.

Reads a cell × bin feature matrix (output of preprocessing.py), applies TF-IDF
+ StandardScaler, reduces with PCA / VAE / VQ-VAE, optionally fuses bin-index
positional embeddings, builds a kNN graph, trains a GraphSAGE encoder with
NT-Xent contrastive loss, then clusters embeddings with Leiden.

No cell-type labels are required (default).  If labels are provided via
--labels (CSV with [cell_id, label]) or --label_pkl (Higashi-style pickle
with one of: 'major', 'cell type', 'cell_type', 'cluster label', 'label',
'labels'), supervised metrics (NMI, ARI) and a confusion matrix are reported.

Encoders:
  --encoder pca     Linear PCA (default)
  --encoder vae     Variational autoencoder (nonlinear, KL-regularised)
  --encoder vqvae   Vector-quantised VAE (discrete codebook latent)

Fusion:
  --fuse-bin-indices --bin-indices-file hvb_selected_bins_100kb.csv
  Concatenates positional embeddings of selected bin indices with the
  encoder latent before kNN graph construction.

Embed-only mode:
  --embed-only   Stop after embedding (no GraphSAGE / Leiden).
  Useful for comparing PCA vs VAE vs VQ-VAE latent spaces.

Usage:
    python gnn_clustering.py --data matrix.csv --output-dir results/
    python gnn_clustering.py --data matrix.csv --output-dir results/ --labels labels.csv
    python gnn_clustering.py --data matrix.csv --output-dir results/ --label_pkl labels.pkl
    python gnn_clustering.py --data matrix.csv --output-dir results/ --encoder vae --embed-only
    python gnn_clustering.py --data matrix.csv --output-dir results/ --encoder vqvae \
        --fuse-bin-indices --bin-indices-file hvb_selected_bins_100kb.csv
"""

import argparse
import copy
import math
import os
import pickle
import time
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('POLARS_MAX_THREADS', '24')
os.environ.setdefault('OMP_NUM_THREADS',    '12')

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             normalized_mutual_info_score as NMI,
                             adjusted_rand_score as ARI,
                             confusion_matrix)
import scanpy as sc
import anndata
import umap as umap_lib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# Default colours used when plotting true labels
COLORS = ['#5B8DB8', '#F4A35A', '#6DBF8A', '#D96B6B', '#A48CC4',
          '#E8A0BF', '#7DB8A2', '#B8A25A', '#9C8BC4', '#8DA88B']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',       required=True,  metavar='CSV',
                   help='Cell × bin feature matrix CSV (output of preprocessing.py)')
    p.add_argument('--output-dir', required=True,  metavar='DIR',
                   help='Output directory for plots / embeddings / results')
    p.add_argument('--labels',     default=None,   metavar='CSV',
                   help='Optional CSV with columns [cell_id, label]')
    p.add_argument('--label_pkl',  default=None,   metavar='PKL',
                   help='Optional Higashi-style pickle with one of '
                        "'major'/'cell type'/'cell_type'/'cluster label'/'label'/'labels'")
    p.add_argument('--seed',       type=int,   default=0,    metavar='INT')
    p.add_argument('--pca-dim',    type=int,   default=50,   metavar='INT')
    p.add_argument('--k-graph',    type=int,   default=15,   metavar='INT')
    p.add_argument('--hidden-dim', type=int,   default=128,  metavar='INT')
    p.add_argument('--out-dim',    type=int,   default=32,   metavar='INT')
    p.add_argument('--n-layers',   type=int,   default=2,    metavar='INT')
    p.add_argument('--dropout',    type=float, default=0.3,  metavar='FLOAT')
    p.add_argument('--lr',         type=float, default=1e-3, metavar='FLOAT')
    p.add_argument('--epochs',     type=int,   default=200,  metavar='INT')
    p.add_argument('--tau',        type=float, default=0.2,  metavar='FLOAT')
    p.add_argument('--neg-ratio',  type=int,   default=5,    metavar='INT')
    p.add_argument('--n-clusters',  type=int,   default=None, metavar='INT',
                   help='Target number of clusters. If set, Leiden grid search only '
                        'considers clusterings within ±1 of this value. '
                        'If unset, all cluster counts are considered (default).')
    # ─── Encoder / fusion options ───────────────────────────────────────────
    p.add_argument('--encoder',  default='pca', choices=['pca', 'vae', 'vqvae'],
                   help='Dimensionality reduction method (default: pca).')
    p.add_argument('--vae-hidden-dim', type=int, default=256, metavar='INT',
                   help='Hidden dim for VAE / VQ-VAE encoder (default: 256).')
    p.add_argument('--vae-latent-dim', type=int, default=50, metavar='INT',
                   help='Latent dim for VAE / VQ-VAE (replaces --pca-dim, default: 50).')
    p.add_argument('--vae-epochs',     type=int, default=300, metavar='INT',
                   help='Training epochs for VAE / VQ-VAE (default: 300).')
    p.add_argument('--vae-lr',         type=float, default=1e-3, metavar='FLOAT',
                   help='Learning rate for VAE / VQ-VAE (default: 1e-3).')
    p.add_argument('--vae-beta',       type=float, default=1.0, metavar='FLOAT',
                   help='KL divergence weight for VAE (default: 1.0).')
    p.add_argument('--vqvae-n-embed',  type=int, default=512, metavar='INT',
                   help='Codebook size for VQ-VAE (default: 512).')
    p.add_argument('--vqvae-commitment', type=float, default=0.25, metavar='FLOAT',
                   help='Commitment loss weight for VQ-VAE (default: 0.25).')
    p.add_argument('--vqvae-ema',  action='store_true',
                   help='Use EMA codebook update for VQ-VAE instead of gradient.')
    p.add_argument('--vqvae-ema-decay', type=float, default=0.99, metavar='FLOAT',
                   help='EMA decay for VQ-VAE codebook (default: 0.99).')
    # ─── Fusion options ─────────────────────────────────────────────────────
    p.add_argument('--fuse-bin-indices', action='store_true',
                   help='Fuse bin-index positional embeddings with the encoder latent.')
    p.add_argument('--bin-indices-file', default=None, metavar='CSV',
                   help='CSV from preprocessing with [rank, original_index, bin_name, dispersion]. '
                        'Required when --fuse-bin-indices is set.')
    p.add_argument('--pos-embed-dim', type=int, default=16, metavar='INT',
                   help='Dimension of the bin-index positional embedding (default: 16).')
    p.add_argument('--pos-embed-mode', default='learned', choices=['learned', 'sinusoidal'],
                   help='Positional embedding type for bin indices (default: learned).')
    # ─── Embed-only mode ────────────────────────────────────────────────────
    p.add_argument('--embed-only', action='store_true',
                   help='Stop after embedding (no kNN graph, GraphSAGE, or Leiden). '
                        'Useful for comparing PCA / VAE / VQ-VAE embeddings.')
    return p.parse_args()


# ─── MODEL ────────────────────────────────────────────────────────────────────

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.convs    = nn.ModuleList()
        self.bns      = nn.ModuleList()
        self.n_layers = n_layers
        self.dropout  = dropout
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for i in range(self.n_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.normalize(x, dim=1)


def nt_xent_graph_loss(z, edge_index, tau=0.2, neg_sample_ratio=5):
    src, dst  = edge_index
    n_edges   = src.size(0)
    n_nodes   = z.size(0)
    max_edges = min(n_edges, 50_000)
    if n_edges > max_edges:
        idx      = torch.randperm(n_edges, device=z.device)[:max_edges]
        src, dst = src[idx], dst[idx]
    pos_sim = F.cosine_similarity(z[src], z[dst]) / tau
    neg_src = src.repeat_interleave(neg_sample_ratio)
    neg_dst = torch.randint(0, n_nodes, (neg_src.size(0),), device=z.device)
    neg_sim = F.cosine_similarity(z[neg_src], z[neg_dst]) / tau
    neg_sim = neg_sim.view(-1, neg_sample_ratio)
    logits  = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, targets)


# ─── VAE ENCODER ─────────────────────────────────────────────────────────────

class VariationalAutoEncoder(nn.Module):
    """Standard VAE with Gaussian latent.  Replaces PCA for nonlinear reduction."""

    def __init__(self, in_dim, hidden_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """Reconstruction (MSE) + KL divergence."""
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss.item(), kld.item()


def train_vae(data, in_dim, hidden_dim, latent_dim, epochs, lr, beta,
              device, seed=0, batch_size=256):
    """Train a VAE and return (latent_embeddings, model, loss_history)."""
    torch.manual_seed(seed)
    model = VariationalAutoEncoder(in_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    x = torch.tensor(data, dtype=torch.float32).to(device)
    n = x.size(0)
    losses = []

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = x[idx]
            optimizer.zero_grad()
            recon, mu, logvar, z = model(xb)
            loss, rl, kl = vae_loss(recon, xb, mu, logvar, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / n
        losses.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            print(f"    VAE Epoch {epoch+1:4d}/{epochs}  loss={avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x)
        embeddings = mu.cpu().numpy().astype(np.float32)
    return embeddings, model, losses


# ─── VQ-VAE ENCODER ───────────────────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    """Vector quantization layer with codebook, commitment loss, and optional EMA."""

    def __init__(self, n_embed, embed_dim, commitment=0.25, ema=False, ema_decay=0.99):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.commitment = commitment
        self.ema = ema
        self.ema_decay = ema_decay
        embed = torch.randn(n_embed, embed_dim)
        self.register_buffer('embed', embed)
        self.register_buffer('embed_avg', embed.clone())
        self.register_buffer('cluster_size', torch.zeros(n_embed))

    def forward(self, z):
        # z: (B, D) -> find nearest codebook entry
        dist = (z.unsqueeze(1) - self.embed.unsqueeze(0)).pow(2).sum(dim=2)
        indices = dist.argmin(dim=1)
        z_q = self.embed[indices]
        # commitment loss
        commit_loss = F.mse_loss(z_q.detach(), z) * self.commitment
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = commit_loss + codebook_loss
        # straight-through estimator
        z_q_st = z + (z_q - z).detach()
        if self.ema and self.training:
            one_hot = F.one_hot(indices, self.n_embed).type(z.dtype)
            self.cluster_size.data.mul_(self.ema_decay).add_(
                one_hot.sum(0), alpha=1 - self.ema_decay)
            embed_sum = one_hot.t() @ z
            self.embed_avg.data.mul_(self.ema_decay).add_(
                embed_sum, alpha=1 - self.ema_decay)
            n = self.cluster_size.unsqueeze(1)
            self.embed.data.copy_(self.embed_avg / (n + 1e-6))
        return z_q_st, loss, indices


class VQVAEEncoder(nn.Module):
    """VQ-VAE with discrete codebook latent.  Replaces PCA for nonlinear reduction."""

    def __init__(self, in_dim, hidden_dim, latent_dim, n_embed=512,
                 commitment=0.25, dropout=0.1, ema=False, ema_decay=0.99):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.vq = VectorQuantizer(n_embed, latent_dim, commitment, ema, ema_decay)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        recon = self.decoder(z_q)
        return recon, z_e, z_q, vq_loss, indices


def train_vqvae(data, in_dim, hidden_dim, latent_dim, n_embed, commitment,
                ema, ema_decay, epochs, lr, device, seed=0, batch_size=256):
    """Train a VQ-VAE and return (latent_embeddings, model, loss_history)."""
    torch.manual_seed(seed)
    model = VQVAEEncoder(in_dim, hidden_dim, latent_dim, n_embed,
                         commitment, ema=ema, ema_decay=ema_decay).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    x = torch.tensor(data, dtype=torch.float32).to(device)
    n = x.size(0)
    losses = []

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = x[idx]
            optimizer.zero_grad()
            recon, z_e, z_q, vq_loss, _ = model(xb)
            recon_loss = F.mse_loss(recon, xb, reduction='sum')
            loss = recon_loss + vq_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / n
        losses.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            print(f"    VQ-VAE Epoch {epoch+1:4d}/{epochs}  loss={avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        z_e = model.encoder(x)
        embeddings = z_e.cpu().numpy().astype(np.float32)
    return embeddings, model, losses


# ─── BIN-INDEX FUSION ─────────────────────────────────────────────────────────

class BinIndexEmbedding(nn.Module):
    """Positional embedding for bin indices.

    Two modes:
      - 'learned': nn.Embedding lookup (max_index + 1 entries)
      - 'sinusoidal': fixed sinusoidal positional encoding (no learnable params)
    """

    def __init__(self, max_index, embed_dim, mode='learned'):
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim
        if mode == 'learned':
            self.embed = nn.Embedding(max_index + 1, embed_dim)
        else:
            self.register_buffer('pos_table',
                                 _sinusoidal_table(max_index + 1, embed_dim))

    def forward(self, indices):
        if self.mode == 'learned':
            return self.embed(indices)
        else:
            return self.pos_table[indices]


def _sinusoidal_table(n, dim):
    """Generate sinusoidal positional encoding table (n x dim)."""
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    table = torch.zeros(n, dim)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)
    return table


def load_bin_indices(csv_path):
    """Load bin-index CSV from preprocessing. Returns array of original_index."""
    df = pd.read_csv(csv_path)
    return df['original_index'].to_numpy().astype(np.int64)


def fuse_embeddings(latent, bin_indices, pos_embed_dim, mode, device, max_index=None):
    """Fuse VAE/VQ-VAE latent with bin-index positional embeddings.

    latent:      (n_cells, latent_dim) numpy
    bin_indices: (n_bins,) numpy — original column indices of selected bins
    Returns:     (n_cells, latent_dim + pos_embed_dim) numpy

    The positional embedding is per-bin.  For each cell, we average the
    positional embeddings across all selected bins (since the cell x bin
    matrix has the same column ordering for every cell) and concatenate
    with the cell's latent vector.
    """
    if max_index is None:
        max_index = int(bin_indices.max())
    indices_t = torch.tensor(bin_indices, dtype=torch.long).to(device)
    embedder = BinIndexEmbedding(max_index, pos_embed_dim, mode).to(device)
    with torch.no_grad():
        pos_emb = embedder(indices_t)
        pos_emb_avg = pos_emb.mean(dim=0, keepdim=True)
        pos_expanded = pos_emb_avg.expand(latent.shape[0], -1)
    fused = np.concatenate([latent, pos_expanded.cpu().numpy()], axis=1)
    return fused.astype(np.float32)


# ─── DATA ─────────────────────────────────────────────────────────────────────

def load_features(data_csv):
    """Load cell × bin matrix via Polars (fast multi-threaded read)."""
    df = pl.read_csv(data_csv)
    # First column is the cell identifier (cell_id / sample_id / index column)
    id_col = df.columns[0]
    cell_ids = df[id_col].to_numpy()
    data     = df.select(df.columns[1:]).to_numpy().astype(np.float32)

    tf       = data / (data.sum(axis=1, keepdims=True) + 1e-8)
    idf      = np.log1p(data.shape[0] / (1 + (data > 0).sum(axis=0)))
    tfidf    = tf * idf
    tfidf_ss = StandardScaler().fit_transform(tfidf).astype(np.float32)
    return tfidf_ss, [str(c) for c in cell_ids]


def load_labels(label_csv, label_pkl, cell_ids):
    """Load optional labels.  Returns (labels_int, label_names) or (None, None).

    Priority: label_pkl -> label_csv -> none.
    """
    # 1) Higashi-style pickle
    if label_pkl and os.path.exists(label_pkl):
        with open(label_pkl, 'rb') as fh:
            info = pickle.load(fh)
        key = next((k for k in ('major', 'cell type', 'cell_type', 'celltype',
                                'cluster label', 'cluster_label', 'label', 'labels')
                    if k in info), None)
        if key is None:
            raise KeyError(f"No label field in {label_pkl}. Keys: {list(info.keys())}")
        raw = list(info[key])
        if len(raw) > len(cell_ids):
            raw = raw[:len(cell_ids)]
        elif len(raw) < len(cell_ids):
            raise ValueError(f"label_pkl has {len(raw)} labels but data has {len(cell_ids)} cells")
        names = sorted(set(raw))
        lmap  = {l: i for i, l in enumerate(names)}
        return np.array([lmap[c] for c in raw]), names

    # 2) CSV with [cell_id, label]
    if label_csv and os.path.exists(label_csv):
        ldf = pd.read_csv(label_csv).set_index(pd.read_csv(label_csv).columns[0])
        col = ldf.columns[0]
        raw = [str(ldf.loc[c, col]) if c in ldf.index else 'unknown' for c in cell_ids]
        names = sorted(set(raw))
        lmap  = {l: i for i, l in enumerate(names)}
        return np.array([lmap[l] for l in raw]), names

    return None, None


# ─── CLUSTERING ───────────────────────────────────────────────────────────────

def cluster_leiden(embeddings, true_labels=None, n_clusters=None):
    """Leiden grid search.

    With labels: optimise ARI.  Without labels: optimise silhouette.
    If n_clusters is set, only consider clusterings within ±1 of target.
    Returns (best_pred, best_tag, best_nmi, best_ari, best_sil).
    """
    use_ari = true_labels is not None
    best_score = -1.0
    best_pred, best_tag = None, ""
    best_nmi, best_ari, best_sil = float('nan'), float('nan'), float('nan')

    for k in [15, 20, 25, 30]:
        for res in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            try:
                adata = anndata.AnnData(X=embeddings.astype(np.float32))
                sc.pp.neighbors(adata, n_neighbors=k, use_rep='X')
                sc.tl.leiden(adata, resolution=res, random_state=42,
                             flavor='igraph', n_iterations=2, key_added='leiden')
                pred = adata.obs['leiden'].astype(int).values
                nc   = len(np.unique(pred))
                if nc < 2:
                    continue
                if n_clusters is not None and nc != n_clusters:
                    continue
                sil = silhouette_score(embeddings, pred)
                if use_ari:
                    score = ARI(true_labels, pred)
                else:
                    score = sil
                if score > best_score:
                    best_score = score
                    best_pred  = pred.copy()
                    best_tag   = f'k={k},r={res:.2f},nc={nc}'
                    best_sil   = sil
                    if use_ari:
                        best_ari = ARI(true_labels, pred)
                        best_nmi = NMI(true_labels, pred)
            except Exception:
                pass

    return best_pred, best_tag, best_nmi, best_ari, best_sil


# ─── PLOTS ────────────────────────────────────────────────────────────────────

def _cmap_for(n):
    return plt.cm.get_cmap('tab20' if n > 10 else 'tab10', max(n, 2))


def plot_umap(embeddings, pred_labels, out_prefix, true_labels=None, label_names=None):
    print("  UMAP...")
    reducer = umap_lib.UMAP(n_neighbors=20, min_dist=0.1, random_state=42, verbose=False)
    umap_xy = reducer.fit_transform(embeddings)
    nc      = len(np.unique(pred_labels))
    cmap    = _cmap_for(nc)

    n_panels = 2 if true_labels is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    for c in range(nc):
        m = pred_labels == c
        if not m.any():
            continue
        axes[0].scatter(umap_xy[m, 0], umap_xy[m, 1], c=[cmap(c)], s=10, alpha=0.7,
                        label=f'C{c} (n={m.sum()})', edgecolors='none')
    sil_str = f'sil={silhouette_score(embeddings, pred_labels):.3f}' if nc >= 2 else 'sil=n/a'
    axes[0].set_title(f'Predicted clusters  ({sil_str})', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('UMAP-1');  axes[0].set_ylabel('UMAP-2')
    axes[0].legend(fontsize=8, frameon=True, markerscale=2)

    if true_labels is not None:
        ntl = len(np.unique(true_labels))
        for i in range(ntl):
            m = true_labels == i
            lbl = label_names[i] if label_names else str(i)
            axes[1].scatter(umap_xy[m, 0], umap_xy[m, 1],
                            c=COLORS[i % len(COLORS)], s=10, alpha=0.7,
                            label=f'{lbl} ({m.sum()})', edgecolors='none')
        axes[1].set_title('True labels', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('UMAP-1');  axes[1].set_ylabel('UMAP-2')
        axes[1].legend(fontsize=8, frameon=True, markerscale=2)

    fig.tight_layout()
    fig.savefig(f'{out_prefix}_umap.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_pca(embeddings, pred_labels, out_prefix, true_labels=None, label_names=None):
    print("  PCA...")
    pca2   = PCA(n_components=2)
    pca_xy = pca2.fit_transform(embeddings)
    nc     = len(np.unique(pred_labels))
    cmap   = _cmap_for(nc)

    n_panels = 2 if true_labels is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    for c in range(nc):
        m = pred_labels == c
        if not m.any():
            continue
        axes[0].scatter(pca_xy[m, 0], pca_xy[m, 1], c=[cmap(c)], s=10, alpha=0.7,
                        label=f'C{c} (n={m.sum()})', edgecolors='none')
    axes[0].set_title('Predicted clusters', fontsize=13, fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]:.1%})')
    axes[0].legend(fontsize=8, frameon=True, markerscale=2)

    if true_labels is not None:
        ntl = len(np.unique(true_labels))
        for i in range(ntl):
            m = true_labels == i
            lbl = label_names[i] if label_names else str(i)
            axes[1].scatter(pca_xy[m, 0], pca_xy[m, 1],
                            c=COLORS[i % len(COLORS)], s=10, alpha=0.7,
                            label=f'{lbl} ({m.sum()})', edgecolors='none')
        axes[1].set_title('True labels', fontsize=13, fontweight='bold')
        axes[1].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]:.1%})')
        axes[1].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]:.1%})')
        axes[1].legend(fontsize=8, frameon=True, markerscale=2)

    fig.tight_layout()
    fig.savefig(f'{out_prefix}_pca.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_silhouette(embeddings, pred_labels, out_prefix):
    print("  Silhouette...")
    sil_vals = silhouette_samples(embeddings, pred_labels)
    nc       = len(np.unique(pred_labels))
    cmap     = _cmap_for(nc)
    fig, ax  = plt.subplots(figsize=(10, max(4, nc)))
    y_lower  = 10
    for c in sorted(np.unique(pred_labels)):
        vals = np.sort(sil_vals[pred_labels == c])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         facecolor=cmap(c), edgecolor=cmap(c), alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(vals), f'C{c}', fontsize=9)
        y_lower = y_upper + 5
    ax.axvline(sil_vals.mean(), color='red', linestyle='--', alpha=0.7,
               label=f'avg={sil_vals.mean():.3f}')
    ax.set_xlabel('Silhouette coefficient', fontsize=11)
    ax.set_title('Silhouette by predicted cluster', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_silhouette.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_confusion(true_labels, pred_labels, label_names, out_prefix):
    print("  Confusion matrix...")
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(max(6, cm.shape[1]), max(5, cm.shape[0])))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[f'C{i}' for i in range(cm.shape[1])],
                yticklabels=label_names if label_names else range(cm.shape[0]))
    ax.set_xlabel('Predicted cluster', fontsize=11)
    ax.set_ylabel('True label', fontsize=11)
    nmi_v = NMI(true_labels, pred_labels)
    ari_v = ARI(true_labels, pred_labels)
    ax.set_title(f'Confusion matrix  NMI={nmi_v:.3f}  ARI={ari_v:.3f}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_confusion.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_loss_curve(losses, out_prefix):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, lw=1.5, color='steelblue')
    ax.set_xlabel('Epoch');  ax.set_ylabel('NT-Xent loss')
    ax.set_title('GNN training loss', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{out_prefix}_loss.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    out_prefix = os.path.join(args.output_dir, 'gnn_clustering')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("CHR3D  —  GNN CLUSTERING  (GraphSAGE + Leiden)")
    print("=" * 70)
    print(f"  Data       : {args.data}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Encoder    : {args.encoder}")
    print(f"  Fuse bins  : {args.fuse_bin_indices}")
    print(f"  Embed only : {args.embed_only}")
    print(f"  Labels     : {args.labels or args.label_pkl or 'none (unsupervised mode)'}")
    print(f"  Device     : {DEVICE}")
    print(f"  Seed       : {args.seed}")
    print("=" * 70)

    t0 = time.time()

    print("\n[DATA] Loading features...")
    tfidf_ss, cell_ids = load_features(args.data)
    print(f"  {len(cell_ids)} cells, {tfidf_ss.shape[1]} features")

    true_labels, label_names = load_labels(args.labels, args.label_pkl, cell_ids)
    if true_labels is not None:
        print(f"  Labels loaded: {label_names}")
    else:
        print("  No labels — running fully unsupervised (silhouette objective).")

    # ─── ENCODER ──────────────────────────────────────────────────────────────
    encoder_losses = []
    pca_model = None
    vae_model = None
    vqvae_model = None

    if args.encoder == 'pca':
        latent_dim = min(args.pca_dim, tfidf_ss.shape[0] - 1, tfidf_ss.shape[1])
        print(f"\n[PCA] {tfidf_ss.shape[1]}D → {latent_dim}D...")
        pca_model = PCA(n_components=latent_dim, random_state=args.seed)
        X_latent = pca_model.fit_transform(tfidf_ss).astype(np.float32)
        print(f"  Cumulative var (top-{min(5, latent_dim)}): "
              f"{pca_model.explained_variance_ratio_[:5].cumsum()[-1]:.1%}")

    elif args.encoder == 'vae':
        latent_dim = args.vae_latent_dim
        print(f"\n[VAE] {tfidf_ss.shape[1]}D → {latent_dim}D  "
              f"hidden={args.vae_hidden_dim}  epochs={args.vae_epochs}  "
              f"β={args.vae_beta}  lr={args.vae_lr}")
        X_latent, vae_model, encoder_losses = train_vae(
            tfidf_ss, tfidf_ss.shape[1], args.vae_hidden_dim, latent_dim,
            args.vae_epochs, args.vae_lr, args.vae_beta,
            DEVICE, seed=args.seed)
        print(f"  Embeddings: {X_latent.shape}")

    elif args.encoder == 'vqvae':
        latent_dim = args.vae_latent_dim
        print(f"\n[VQ-VAE] {tfidf_ss.shape[1]}D → {latent_dim}D  "
              f"hidden={args.vae_hidden_dim}  codebook={args.vqvae_n_embed}  "
              f"commit={args.vqvae_commitment}  ema={args.vqvae_ema}  "
              f"epochs={args.vae_epochs}")
        X_latent, vqvae_model, encoder_losses = train_vqvae(
            tfidf_ss, tfidf_ss.shape[1], args.vae_hidden_dim, latent_dim,
            args.vqvae_n_embed, args.vqvae_commitment,
            args.vqvae_ema, args.vqvae_ema_decay,
            args.vae_epochs, args.vae_lr,
            DEVICE, seed=args.seed)
        print(f"  Embeddings: {X_latent.shape}")

    # ─── FUSION ───────────────────────────────────────────────────────────────
    if args.fuse_bin_indices:
        if not args.bin_indices_file:
            raise ValueError("--bin-indices-file is required when --fuse-bin-indices is set")
        print(f"\n[FUSION] Loading bin indices from {args.bin_indices_file}...")
        bin_indices = load_bin_indices(args.bin_indices_file)
        print(f"  {len(bin_indices)} bin indices, max={int(bin_indices.max())}")
        print(f"  Pos embed: dim={args.pos_embed_dim}  mode={args.pos_embed_mode}")
        X_latent = fuse_embeddings(
            X_latent, bin_indices, args.pos_embed_dim,
            args.pos_embed_mode, DEVICE)
        print(f"  Fused shape: {X_latent.shape}")

    # ─── EMBED-ONLY: save and exit ────────────────────────────────────────────
    if args.embed_only:
        elapsed = time.time() - t0
        np.save(f'{out_prefix}_embeddings.npy', X_latent)
        if encoder_losses:
            plot_loss_curve(encoder_losses, out_prefix)
        if pca_model is not None:
            torch.save({'pca_model': pca_model,
                        'config': {'encoder': 'pca', 'latent_dim': latent_dim}},
                       f'{out_prefix}_model.pth')
        elif vae_model is not None:
            torch.save({'model_state_dict': vae_model.state_dict(),
                        'config': {'encoder': 'vae', 'latent_dim': latent_dim,
                                   'hidden_dim': args.vae_hidden_dim,
                                   'beta': args.vae_beta}},
                       f'{out_prefix}_model.pth')
        elif vqvae_model is not None:
            torch.save({'model_state_dict': vqvae_model.state_dict(),
                        'config': {'encoder': 'vqvae', 'latent_dim': latent_dim,
                                   'hidden_dim': args.vae_hidden_dim,
                                   'n_embed': args.vqvae_n_embed}},
                       f'{out_prefix}_model.pth')
        print(f"\n[EMBED-ONLY] Saved embeddings {X_latent.shape} in {elapsed:.0f}s")
        print(f"  {out_prefix}_embeddings.npy")
        if encoder_losses:
            print(f"  {out_prefix}_loss.png")
        print(f"  {out_prefix}_model.pth")
        print("=" * 70)
        return

    # ─── GRAPH + GraphSAGE + Leiden (full pipeline) ───────────────────────────
    k_graph = min(args.k_graph, len(cell_ids) - 1)
    print(f"\n[GRAPH] Building kNN graph (k={k_graph})...")
    A = kneighbors_graph(X_latent, n_neighbors=k_graph, mode='connectivity',
                         include_self=False, n_jobs=-1)
    A = A + A.T
    A[A > 1] = 1
    rows, cols = A.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long).to(DEVICE)
    print(f"  {edge_index.size(1):,} edges")

    gnn_in_dim = X_latent.shape[1]
    print(f"\n[TRAIN] GraphSAGE  {gnn_in_dim}→{args.hidden_dim}→{args.out_dim}  "
          f"layers={args.n_layers}  epochs={args.epochs}  τ={args.tau}")
    x_tensor  = torch.tensor(X_latent, dtype=torch.float32).to(DEVICE)
    model     = GraphSAGEEncoder(gnn_in_dim, args.hidden_dim, args.out_dim,
                                 args.n_layers, args.dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=1e-6)

    best_loss, best_state = float('inf'), None
    all_losses = []
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        z    = model(x_tensor, edge_index)
        loss = nt_xent_graph_loss(z, edge_index, tau=args.tau,
                                  neg_sample_ratio=args.neg_ratio)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        all_losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:4d}/{args.epochs}  loss={loss.item():.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embeddings = model(x_tensor, edge_index).cpu().numpy()
    print(f"  Best loss : {best_loss:.4f}")
    print(f"  Embeddings: {embeddings.shape}")

    ncl_str = f", target≈{args.n_clusters}" if args.n_clusters else ""
    print("\n[CLUSTER] Leiden grid search "
          f"(optimising {'ARI' if true_labels is not None else 'silhouette'}{ncl_str})...")
    best_pred, best_tag, best_nmi, best_ari, best_sil = cluster_leiden(
        embeddings, true_labels, n_clusters=args.n_clusters)

    if best_pred is None:
        print("  WARNING: clustering failed — assigning all to cluster 0")
        best_pred = np.zeros(len(cell_ids), dtype=int)

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Clusters    : {len(np.unique(best_pred))}")
    print(f"  Silhouette  : {best_sil:.4f}")
    if true_labels is not None:
        print(f"  NMI         : {best_nmi:.4f}")
        print(f"  ARI         : {best_ari:.4f}")
    print(f"  Best config : {best_tag}")
    print(f"  Time        : {elapsed:.0f}s")

    print("\n[CLUSTER ANALYSIS]")
    for c in sorted(np.unique(best_pred)):
        m = best_pred == c
        print(f"  Cluster {c:2d}: n={m.sum()}", end="")
        if true_labels is not None:
            cnts = np.bincount(true_labels[m], minlength=len(label_names))
            dom  = label_names[int(np.argmax(cnts))]
            pur  = cnts.max() / cnts.sum()
            print(f"  dominant={dom} ({pur:.0%})", end="")
        print()

    # Save results
    res_row = {'seed': args.seed, 'encoder': args.encoder, 'silhouette': best_sil,
               'n_clusters': int(len(np.unique(best_pred))), 'config': best_tag}
    if args.fuse_bin_indices:
        res_row['fuse_bin_indices'] = True
        res_row['pos_embed_dim'] = args.pos_embed_dim
        res_row['pos_embed_mode'] = args.pos_embed_mode
    if true_labels is not None:
        res_row.update({'nmi': best_nmi, 'ari': best_ari})
    pd.DataFrame([res_row]).to_csv(f'{out_prefix}_results.csv', index=False)

    pred_df = pd.DataFrame({'cell_id': cell_ids, 'cluster': best_pred})
    if true_labels is not None:
        pred_df['true_label'] = [label_names[l] for l in true_labels]
    pred_df.to_csv(f'{out_prefix}_predictions.csv', index=False)

    np.save(f'{out_prefix}_embeddings.npy', embeddings)

    model_save = {
        'model_state_dict': model.state_dict(),
        'config': {'seed': args.seed, 'encoder': args.encoder,
                   'latent_dim': latent_dim, 'gnn_in_dim': gnn_in_dim,
                   'k_graph': k_graph,
                   'hidden_dim': args.hidden_dim, 'out_dim': args.out_dim}
    }
    if pca_model is not None:
        model_save['pca_model'] = pca_model
    if vae_model is not None:
        model_save['vae_state_dict'] = vae_model.state_dict()
    if vqvae_model is not None:
        model_save['vqvae_state_dict'] = vqvae_model.state_dict()
    torch.save(model_save, f'{out_prefix}_model.pth')

    print("\n[PLOTS] Generating visualizations...")
    if encoder_losses:
        plot_loss_curve(encoder_losses, f'{out_prefix}_encoder')
    plot_loss_curve(all_losses, out_prefix)
    nc_final = len(np.unique(best_pred))
    if nc_final >= 2:
        plot_umap(embeddings, best_pred, out_prefix, true_labels, label_names)
        plot_pca (embeddings, best_pred, out_prefix, true_labels, label_names)
        plot_silhouette(embeddings, best_pred, out_prefix)
        if true_labels is not None:
            plot_confusion(true_labels, best_pred, label_names, out_prefix)
    else:
        print(f"  Skipping UMAP/PCA/silhouette: only {nc_final} cluster (need ≥2)")

    saved = [f'{out_prefix}_loss.png', f'{out_prefix}_results.csv',
             f'{out_prefix}_predictions.csv', f'{out_prefix}_embeddings.npy',
             f'{out_prefix}_model.pth']
    if encoder_losses:
        saved.append(f'{out_prefix}_encoder_loss.png')
    if nc_final >= 2:
        saved += [f'{out_prefix}_umap.png', f'{out_prefix}_pca.png',
                  f'{out_prefix}_silhouette.png']
        if true_labels is not None:
            saved.append(f'{out_prefix}_confusion.png')

    print("\n" + "=" * 70)
    print("Files saved:")
    for f in saved:
        print(f"  {f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
