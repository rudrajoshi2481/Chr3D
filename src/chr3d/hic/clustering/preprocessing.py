# Copyright (c) 2026 Rudhra Joshi and Yong Chen
# Licensed under the MIT License. See LICENSE in the project root for details.
# This software was developed with support from the National Science Foundation under CAREER Award DBI-2239350.
"""
Unified preprocessing for scHi-C data. Three modes:

  - text  : tab-delimited Higashi-style data.txt
            (columns: cell_id, chrom1, chrom2, pos1, pos2, count, [normalized_count])
  - mcool : one .mcool file per cell in a directory (recursive glob)
  - cool  : one .cool  file per cell in a directory (recursive glob)

Pipeline:
  1. Build raw cell × bin contact matrix using a chrom.sizes file
     (no hardcoded bin counts; bins per chrom = ceil(size / resolution))
  2. Coverage normalisation: divide each cell's row by its non-zero fraction,
     rescaled to the mean coverage across cells
  3. Log10 transform: log10(1 + x)
  4. Quantise globally to 0..255 int

Output (consumed by gnn_clustering.py):
  <output_dir>/cell_bin_matrix_{resolution}_coverage_log10int.csv
  <output_dir>/cell_bin_matrix_{resolution}_coverage.csv          (optional)
  <output_dir>/summary_{resolution}.json

Optional HVB (Highly Variable Bins) filtering (off by default):
  --hvb_configs   topN | pctX | none   (one or more; each writes a hvb_<tag>/ subdir)
  --hvb_coverage_filter   minimum per-bin cell coverage before HVB ranking

Examples:
  python preprocessing.py --mode text  --resolution 100kb \
      --data_file data.txt --chrom_sizes hg38.chrom.sizes --output_dir ./out
  python preprocessing.py --mode mcool --resolution 100kb \
      --mcool_dir cells/ --chrom_sizes hg38.chrom.sizes --output_dir ./out
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import polars as pl


def parse_resolution(resolution_str):
    """Parse '100kb', '500kb', '1Mb' -> base pairs."""
    rl = resolution_str.lower().strip()
    if rl.endswith('kb'):
        return int(float(rl[:-2]) * 1_000)
    if rl.endswith('mb'):
        return int(float(rl[:-2]) * 1_000_000)
    if rl.endswith('bp'):
        return int(rl[:-2])
    return int(rl)


def load_chrom_sizes(chrom_sizes_file, allow_chroms=None):
    """Load a chrom.sizes file (chrom\tsize). Returns OrderedDict {chrom: size}.

    If `allow_chroms` is given (list of names), restrict to those (in file order).
    """
    if not os.path.exists(chrom_sizes_file):
        raise FileNotFoundError(f"chrom.sizes not found: {chrom_sizes_file}")

    chrom_sizes = {}
    with open(chrom_sizes_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            chrom, size = parts[0], int(parts[1])
            if allow_chroms is None or chrom in allow_chroms:
                chrom_sizes[chrom] = size
    if not chrom_sizes:
        raise ValueError(f"No chromosomes parsed from {chrom_sizes_file}")
    return chrom_sizes


def build_bin_layout(chrom_sizes, resolution_bp):
    """From a chrom_sizes dict + resolution, build a column layout.

    Returns (col_names, chrom_offsets, chrom_n_bins, total_bins).
    """
    col_names    = []
    chrom_offsets = {}
    chrom_n_bins  = {}
    offset = 0
    width = max(4, int(math.log10(max(1, max(
        math.ceil(s / resolution_bp) for s in chrom_sizes.values())))) + 1)
    for chrom, size in chrom_sizes.items():
        n_bins = int(math.ceil(size / resolution_bp))
        chrom_offsets[chrom] = offset
        chrom_n_bins[chrom]  = n_bins
        for b in range(n_bins):
            col_names.append(f"{chrom}_bin{b:0{width}d}")
        offset += n_bins
    return col_names, chrom_offsets, chrom_n_bins, offset


def coverage_normalize_and_log10(raw_matrix, return_float=True):
    """Coverage-normalise + log10 quantise to 0..255 int.

    Pipeline:
      coverage_i = nnz(row_i) / total_bins
      mean_cov   = mean(coverage)
      norm_i     = row_i / coverage_i * mean_cov
      log_data   = log10(1 + norm)
      int_data   = round(log_data / log_max * 255)  in [0, 255]

    If `return_float` is False, the float `norm` matrix is freed after
    quantisation (returns None for it) — saves ~one full-matrix allocation
    when only the int CSV is needed.
    """
    total_bins = raw_matrix.shape[1]
    coverage   = np.count_nonzero(raw_matrix, axis=1) / max(total_bins, 1)
    mean_cov   = float(coverage.mean()) if coverage.size else 0.0
    scale      = mean_cov if mean_cov > 0 else 1.0
    cov_safe   = np.where(coverage > 0, coverage, scale).astype(np.float32)

    # In-place row scaling: norm = raw_matrix * (scale / cov_safe)
    factors = (scale / cov_safe).astype(np.float32)
    raw_matrix *= factors[:, None]   # raw_matrix is now `norm`
    norm = raw_matrix

    gmax    = float(norm.max()) if norm.size else 0.0
    log_max = np.log10(1.0 + gmax) if gmax > 0 else 1.0

    # Build int_data in chunks to cap peak memory
    int_data  = np.empty(norm.shape, dtype=np.int32)
    chunk     = max(1, 4_000_000 // max(norm.shape[1], 1))   # ~16 MB float32 per chunk
    for s in range(0, norm.shape[0], chunk):
        e = min(s + chunk, norm.shape[0])
        log_chunk = np.log10(1.0 + norm[s:e])
        np.clip(np.round(log_chunk / log_max * 255.0), 0, 255,
                out=log_chunk)
        int_data[s:e] = log_chunk.astype(np.int32)

    if not return_float:
        norm = None
    return norm, int_data, coverage, mean_cov


# ---------------------------------------------------------------------------
# HVB (Highly Variable Bins) filtering — optional
# ---------------------------------------------------------------------------

def _parse_hvb_spec(spec):
    """Parse 'none' | 'topN' | 'pctX' -> (kind, value)."""
    s = str(spec).strip().lower()
    if s in ('none', 'full', 'all'):
        return ('none', 0)
    if s.startswith('top'):
        return ('topn', int(s[3:]))
    if s.startswith('pct'):
        return ('pct', float(s[3:]) / 100.0)
    raise ValueError(f"Unknown HVB spec: {spec} (expected 'none', 'topN', or 'pctX')")


def _select_hvb_bins(norm_matrix, kind, value, coverage_filter=0.05):
    """Return (mask, n_selected, stats, selected_indices) over the original bin axis.

    selected_indices is a 1-D array of original column positions, ordered by
    descending dispersion (most variable first).
    """
    n_bins  = norm_matrix.shape[1]
    bin_nz  = (norm_matrix > 0).mean(axis=0)
    cov_msk = bin_nz > coverage_filter
    n_after = int(cov_msk.sum())
    if n_after == 0:
        return np.zeros(n_bins, dtype=bool), 0, {
            'n_original_bins': n_bins, 'coverage_filter': coverage_filter,
            'n_after_coverage': 0, 'n_selected': 0}, np.array([], dtype=np.int64)

    sub  = norm_matrix[:, cov_msk]
    eps  = 1e-8
    disp = sub.var(axis=0) / (sub.mean(axis=0) + eps)

    if kind == 'topn':
        n_sel = int(value)
    elif kind == 'pct':
        n_sel = int(round(value * n_bins))
    else:
        n_sel = n_after
    n_sel = max(1, min(n_sel, n_after))

    top_sub = np.argpartition(disp, -n_sel)[-n_sel:]
    top_sub = top_sub[np.argsort(disp[top_sub])[::-1]]
    pos     = np.where(cov_msk)[0][top_sub]
    mask    = np.zeros(n_bins, dtype=bool)
    mask[pos] = True

    return mask, int(mask.sum()), {
        'n_original_bins':  n_bins,
        'coverage_filter':  coverage_filter,
        'n_after_coverage': n_after,
        'n_selected':       int(mask.sum()),
        'dispersion_min':   float(disp[top_sub].min()),
        'dispersion_max':   float(disp[top_sub].max()),
    }, pos


# ---------------------------------------------------------------------------
# Bin-index saving (for downstream fusion in gnn_clustering.py)
# ---------------------------------------------------------------------------

def _save_bin_indices(output_dir, resolution_str, selected_indices, col_names,
                      disp_values=None):
    """Save selected bin indices to a CSV for downstream positional embedding.

    Columns: rank, original_index, bin_name, dispersion
    """
    os.makedirs(output_dir, exist_ok=True)
    idx_file = os.path.join(output_dir, f'hvb_selected_bins_{resolution_str}.csv')
    n = len(selected_indices)
    rows = {
        'rank':            np.arange(1, n + 1, dtype=np.int32),
        'original_index':  selected_indices.astype(np.int64),
        'bin_name':        [col_names[i] for i in selected_indices],
    }
    if disp_values is not None:
        rows['dispersion'] = disp_values.astype(np.float32)
    df = pl.DataFrame(rows)
    df.write_csv(idx_file)
    print(f"  [write] bin indices -> {idx_file} ({n} bins)")
    return idx_file


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _polars_write_matrix(path, matrix, index, col_names, is_int):
    arr = np.ascontiguousarray(matrix, dtype=np.int32 if is_int else np.float32)
    df  = pl.from_numpy(arr, schema=list(col_names))
    df  = df.with_columns(pl.Series('cell_id', list(index))).select(['cell_id', *col_names])
    df.write_csv(path)


def _save_outputs(output_dir, resolution_str, norm_matrix, int_data, index, col_names,
                  summary_extra, save_float_csv=False):
    os.makedirs(output_dir, exist_ok=True)
    cov_file = os.path.join(output_dir, f'cell_bin_matrix_{resolution_str}_coverage.csv')
    log_file = os.path.join(output_dir, f'cell_bin_matrix_{resolution_str}_coverage_log10int.csv')

    t = time.time()
    _polars_write_matrix(log_file, int_data, index, col_names, is_int=True)
    print(f"  [write] log10int csv ({os.path.getsize(log_file)/1e6:.1f} MB) in {time.time()-t:.1f}s")

    if save_float_csv:
        t = time.time()
        _polars_write_matrix(cov_file, norm_matrix, index, col_names, is_int=False)
        print(f"  [write] float coverage csv ({os.path.getsize(cov_file)/1e6:.1f} MB) in {time.time()-t:.1f}s")
    else:
        cov_file = None

    summary = {'resolution': resolution_str,
               'resolution_bp': parse_resolution(resolution_str),
               'coverage_file': cov_file,
               'log10_file':    log_file}
    summary.update(summary_extra)
    summary_file = os.path.join(output_dir, f'summary_{resolution_str}.json')
    with open(summary_file, 'w') as fp:
        json.dump(summary, fp, indent=2, default=float)
    return cov_file, log_file, summary_file, summary


def _write_all_hvb_variants(base_output_dir, resolution_str, norm_matrix, int_data,
                            index, col_names, hvb_configs, summary_extra,
                            save_float_csv=False, coverage_filter=0.05):
    outputs = {}
    if not hvb_configs:
        outputs['full'] = _save_outputs(
            base_output_dir, resolution_str, norm_matrix, int_data,
            index=index, col_names=col_names,
            summary_extra={**summary_extra, 'hvb': None},
            save_float_csv=save_float_csv)
        return outputs

    print(f"\n[HVB] Writing {len(hvb_configs)} variant(s): {hvb_configs}")
    for spec in hvb_configs:
        kind, value = _parse_hvb_spec(spec)
        tag = str(spec).strip().lower()
        sub_dir = os.path.join(base_output_dir, f'hvb_{tag}')

        if kind == 'none':
            print(f"  [HVB:{tag}] keeping all {norm_matrix.shape[1]:,} bins")
            sel_norm, sel_int, sel_cols = norm_matrix, int_data, col_names
            sel_indices = np.arange(norm_matrix.shape[1], dtype=np.int64)
            hvb_stats = {'hvb': tag,
                         'n_original_bins': int(norm_matrix.shape[1]),
                         'n_selected_bins': int(norm_matrix.shape[1])}
            _save_bin_indices(sub_dir, resolution_str, sel_indices, col_names)
        else:
            mask, n_sel, stats, sel_indices = _select_hvb_bins(
                norm_matrix, kind, value, coverage_filter=coverage_filter)
            print(f"  [HVB:{tag}] coverage>{coverage_filter:.0%} -> "
                  f"{stats['n_after_coverage']:,} bins, selected top {n_sel:,}")
            sel_norm = norm_matrix[:, mask]
            sel_int  = int_data[:, mask]
            sel_cols = [col_names[i] for i in np.where(mask)[0]]
            hvb_stats = {'hvb': tag, 'hvb_kind': kind, 'hvb_value': value, **stats}
            sub_for_disp = norm_matrix[:, mask]
            eps = 1e-8
            disp_vals = (sub_for_disp.var(axis=0) /
                         (sub_for_disp.mean(axis=0) + eps)).astype(np.float32)
            disp_ordered = disp_vals[np.argsort(disp_vals)[::-1]]
            _save_bin_indices(sub_dir, resolution_str, sel_indices,
                              col_names, disp_values=disp_ordered)

        outputs[tag] = _save_outputs(
            sub_dir, resolution_str, sel_norm, sel_int,
            index=index, col_names=sel_cols,
            summary_extra={**summary_extra, **hvb_stats},
            save_float_csv=save_float_csv)
    return outputs


# ---------------------------------------------------------------------------
# TEXT MODE
# ---------------------------------------------------------------------------

def preprocess_text(data_file, resolution_str, output_dir, chrom_sizes_file,
                    n_threads=24, save_float_csv=False,
                    hvb_configs=None, hvb_coverage_filter=0.05,
                    chroms_keep=None):
    """Higashi-style data.txt -> cell × bin matrix at `resolution_str`."""
    resolution_bp = parse_resolution(resolution_str)
    os.environ['POLARS_MAX_THREADS'] = str(n_threads)

    chrom_sizes = load_chrom_sizes(chrom_sizes_file, allow_chroms=chroms_keep)
    col_names, chrom_offsets, chrom_n_bins, total_bins = build_bin_layout(
        chrom_sizes, resolution_bp)

    print("=" * 80)
    print("scHi-C Preprocessing (TEXT mode)")
    print("=" * 80)
    print(f"  Resolution  : {resolution_str} ({resolution_bp:,} bp)")
    print(f"  Data file   : {data_file}")
    print(f"  Chrom sizes : {chrom_sizes_file}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Threads     : {n_threads}")
    print(f"  Genome bins : {total_bins:,} across {len(chrom_sizes)} chromosomes")

    print("\n[1/4] Loading data...")
    t0 = time.time()
    df = pl.read_csv(data_file, separator='\t', schema_overrides={
        'cell_id':          pl.Int64,
        'chrom1':           pl.String,
        'chrom2':           pl.String,
        'pos1':             pl.Float64,
        'pos2':             pl.Float64,
        'count':            pl.Float32,
        'normalized_count': pl.Float32,
    }).with_columns([
        pl.col('pos1').cast(pl.Int64),
        pl.col('pos2').cast(pl.Int64),
    ])
    print(f"  Loaded {len(df):,} contacts in {time.time()-t0:.1f}s")

    df = df.filter(pl.col('chrom1') == pl.col('chrom2'))
    df = df.filter(pl.col('chrom1').is_in(list(chrom_sizes.keys())))
    print(f"  Intra-chromosomal: {len(df):,}")

    df = df.with_columns([
        (pl.col('pos1') // resolution_bp).cast(pl.Int64).alias('bin'),
        pl.col('chrom1').replace_strict(chrom_offsets, default=-1).cast(pl.Int64)
          .alias('chrom_offset'),
        pl.col('chrom1').replace_strict(chrom_n_bins, default=0).cast(pl.Int64)
          .alias('chrom_n_bins'),
    ])
    df = df.filter(pl.col('chrom_offset') >= 0)
    df = df.filter(pl.col('bin') < pl.col('chrom_n_bins'))
    df = df.with_columns((pl.col('chrom_offset') + pl.col('bin')).alias('global_bin'))

    unique_cells = df.select('cell_id').unique().sort('cell_id')['cell_id'].to_numpy()
    print(f"  Unique cells: {len(unique_cells):,}")

    print("\n[2/4] Building contact matrix...")
    t1 = time.time()
    cell_to_row = {int(cid): i for i, cid in enumerate(unique_cells)}
    row_idx = df['cell_id'].replace_strict(cell_to_row, default=-1).to_numpy().astype(np.int32)
    col_idx = df['global_bin'].to_numpy().astype(np.int32)
    vals    = df['count'].to_numpy().astype(np.float32)
    raw_matrix = np.zeros((len(unique_cells), total_bins), dtype=np.float32)
    np.add.at(raw_matrix, (row_idx, col_idx), vals)
    print(f"  Matrix: {raw_matrix.shape[0]:,} × {raw_matrix.shape[1]:,}  ({time.time()-t1:.1f}s)")

    print("\n[3/4] Coverage normalization + log10 quantisation...")
    t2 = time.time()
    keep_float = save_float_csv or bool(hvb_configs)
    norm, int_data, coverage, mean_cov = coverage_normalize_and_log10(
        raw_matrix, return_float=keep_float)
    print(f"  coverage min={coverage.min():.4f} max={coverage.max():.4f} mean={mean_cov:.4f}")
    print(f"  done in {time.time()-t2:.1f}s")

    print("\n[4/4] Saving...")
    summary_extra = {
        'mode': 'text', 'n_cells': int(len(unique_cells)), 'n_bins': int(total_bins),
        'coverage_range': [float(coverage.min()), float(coverage.max())],
        'mean_coverage':  float(mean_cov),
        'sparsity':       float(1.0 - np.count_nonzero(int_data) / max(int_data.size, 1)),
        'processing_time': float(time.time() - t0),
        'chromosomes':    list(chrom_sizes.keys()),
    }
    outputs = _write_all_hvb_variants(
        output_dir, resolution_str, norm, int_data,
        index=unique_cells, col_names=col_names,
        hvb_configs=hvb_configs, summary_extra=summary_extra,
        save_float_csv=save_float_csv,
        coverage_filter=hvb_coverage_filter)

    print("\n" + "=" * 80)
    print("TEXT PREPROCESSING COMPLETE")
    print(f"Base matrix: {len(unique_cells):,} × {total_bins:,}")
    for tag, (_cov, logf, _sumf, _summ) in outputs.items():
        print(f"  [{tag}] -> {logf}")
    print("=" * 80)
    return outputs


# ---------------------------------------------------------------------------
# MCOOL / COOL MODE
# ---------------------------------------------------------------------------

def _get_reference_bins_mcool(mcool_files, resolution_bp, allow_chroms):
    import cooler
    c    = cooler.Cooler(f"{mcool_files[0]}::/resolutions/{resolution_bp}")
    bins = c.bins()[:]
    chroms_avail = [ch for ch in allow_chroms if ch in c.chromnames]
    keep = bins['chrom'].isin(chroms_avail).values
    bins_f = bins[keep].reset_index(drop=True)
    orig_to_new = np.full(len(bins), -1, dtype=np.int64)
    orig_to_new[np.where(keep)[0]] = np.arange(int(keep.sum()))
    return bins_f, orig_to_new, chroms_avail


def _get_reference_bins_cool(cool_files, target_res, allow_chroms):
    import cooler
    c0 = cooler.Cooler(cool_files[0])
    if c0.binsize > target_res:
        raise ValueError(f"cool binsize {c0.binsize} > target {target_res}")
    factor = target_res // c0.binsize
    bins   = c0.bins()[:]
    chroms_avail = [ch for ch in allow_chroms if ch in bins['chrom'].unique().tolist()]
    bins   = bins[bins['chrom'].isin(chroms_avail)].reset_index(drop=True)
    return bins, factor, chroms_avail


def _extract_contacts_mcool(mcool_file, resolution_bp, orig_to_new):
    import cooler
    try:
        c       = cooler.Cooler(f"{mcool_file}::/resolutions/{resolution_bp}")
        pixels  = c.pixels()[:]
        if len(pixels) == 0:
            return None
        bins    = c.bins()[:]
        ch_arr  = bins['chrom'].values
        c1      = ch_arr[pixels['bin1_id'].values]
        c2      = ch_arr[pixels['bin2_id'].values]
        intra   = (c1 == c2)
        if not intra.any():
            return None
        ipx = pixels[intra].reset_index(drop=True)
        b1  = orig_to_new[ipx['bin1_id'].values]
        b2  = orig_to_new[ipx['bin2_id'].values]
        cs  = ipx['count'].values
        ok  = (b1 >= 0) & (b2 >= 0)
        return {'bin1': b1[ok].astype(np.int32),
                'bin2': b2[ok].astype(np.int32),
                'count': cs[ok].astype(np.float32),
                'n_total_pixels': len(pixels),
                'n_intra': int(intra.sum()),
                'total_count': float(cs[ok].sum())}
    except Exception as e:
        print(f"    ERROR {mcool_file}: {e}")
        return None


def _build_matrix_per_cell(all_contacts, total_bins, variant='bin1'):
    n_cells = len(all_contacts)
    matrix  = np.zeros((n_cells, total_bins), dtype=np.float32)
    for i, c in enumerate(all_contacts):
        if c is None:
            continue
        b1, b2, cs = c['bin1'], c['bin2'], c['count']
        if variant == 'bin1':
            np.add.at(matrix[i], b1, cs)
        elif variant in ('both', 'marginal'):
            np.add.at(matrix[i], b1, cs)
            off = b1 != b2
            np.add.at(matrix[i], b2[off], cs[off])
        else:
            raise ValueError(f"Unknown variant: {variant}")
    return matrix


def preprocess_mcool(mcool_dir, resolution_str, output_dir, chrom_sizes_file,
                     variant='bin1', n_cells_limit=None, save_float_csv=False,
                     hvb_configs=None, hvb_coverage_filter=0.05,
                     chroms_keep=None, mode='mcool'):
    """One .mcool (or .cool) file per cell -> cell × bin matrix.

    `mode='mcool'` uses `<file>::/resolutions/<bp>`; `mode='cool'` reads files
    directly and coarsens if needed.
    """
    resolution_bp = parse_resolution(resolution_str)
    chrom_sizes   = load_chrom_sizes(chrom_sizes_file, allow_chroms=chroms_keep)
    allow_chroms  = list(chrom_sizes.keys())

    print("=" * 80)
    print(f"scHi-C Preprocessing ({mode.upper()} mode)  variant={variant}")
    print("=" * 80)
    print(f"  Resolution  : {resolution_str} ({resolution_bp:,} bp)")
    print(f"  Directory   : {mcool_dir}")
    print(f"  Chrom sizes : {chrom_sizes_file}")
    print(f"  Output dir  : {output_dir}")

    t0 = time.time()
    pat   = '*.mcool' if mode == 'mcool' else '*.cool'
    files = sorted(glob.glob(os.path.join(mcool_dir, '**', pat), recursive=True))
    if n_cells_limit:
        files = files[:n_cells_limit]
    print(f"\n[1/4] Found {len(files)} {pat} files")
    if not files:
        raise FileNotFoundError(f"No {pat} files in {mcool_dir}")

    if mode == 'mcool':
        ref_bins, orig_to_new, chroms_avail = _get_reference_bins_mcool(
            files, resolution_bp, allow_chroms)
        total_bins = len(ref_bins)
        print(f"  Reference bins: {total_bins:,} from {len(chroms_avail)} chroms")

        print("\n[2/4] Extracting contacts...")
        t1 = time.time()
        all_contacts, stats = [], {'pixels': 0, 'intra': 0, 'counts': 0.0}
        for i, f in enumerate(files):
            r = _extract_contacts_mcool(f, resolution_bp, orig_to_new)
            all_contacts.append(r)
            if r is not None:
                stats['pixels'] += r['n_total_pixels']
                stats['intra']  += r['n_intra']
                stats['counts'] += r['total_count']
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(files)} ({time.time()-t1:.1f}s)")
        n_valid = sum(1 for c in all_contacts if c is not None)
        print(f"  Valid cells: {n_valid}/{len(all_contacts)}  ({time.time()-t1:.1f}s)")

        print(f"\n[3/4] Building matrix (variant={variant})...")
        matrix = _build_matrix_per_cell(all_contacts, total_bins, variant=variant)

        col_names = []
        counter   = {}
        for ch in ref_bins['chrom'].values:
            k = counter.get(ch, 0)
            col_names.append(f"{ch}_bin{k:05d}")
            counter[ch] = k + 1
    else:
        # cool mode (per-cell .cool): use chrom.sizes-driven layout to
        # tolerate per-cell binsize differences via simple integer coarsening.
        col_names, chrom_offsets, chrom_n_bins, total_bins = build_bin_layout(
            chrom_sizes, resolution_bp)
        print(f"  Reference bins: {total_bins:,} from {len(chrom_sizes)} chroms")
        import cooler
        print("\n[2/4] Extracting contacts...")
        t1 = time.time()
        all_contacts, stats = [], {'pixels': 0, 'intra': 0, 'counts': 0.0}
        for i, f in enumerate(files):
            try:
                c = cooler.Cooler(f)
                if c.binsize > resolution_bp:
                    raise ValueError(f"binsize {c.binsize} > target {resolution_bp}")
                factor = resolution_bp // c.binsize
                px     = c.pixels()[:]
                bins   = c.bins()[:]
                ch_arr = bins['chrom'].values
                b1, b2 = px['bin1_id'].values, px['bin2_id'].values
                ch1, ch2 = ch_arr[b1], ch_arr[b2]
                ok = (ch1 == ch2) & np.isin(ch1, allow_chroms)
                if not ok.any():
                    all_contacts.append(None); continue
                starts = bins['start'].values
                b1, b2, ch1 = b1[ok], b2[ok], ch1[ok]
                cs  = px['count'].values[ok].astype(np.float32)
                offsets = np.array([chrom_offsets[c] for c in ch1], dtype=np.int64)
                gb1 = offsets + (starts[b1] // resolution_bp)
                gb2 = offsets + (starts[b2] // resolution_bp)
                all_contacts.append({'bin1': gb1.astype(np.int32),
                                     'bin2': gb2.astype(np.int32),
                                     'count': cs,
                                     'n_total_pixels': len(px),
                                     'n_intra': int(ok.sum()),
                                     'total_count': float(cs.sum())})
                stats['pixels'] += len(px); stats['intra'] += int(ok.sum())
                stats['counts'] += float(cs.sum())
            except Exception as e:
                print(f"    ERROR {f}: {e}")
                all_contacts.append(None)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(files)} ({time.time()-t1:.1f}s)")
        n_valid = sum(1 for c in all_contacts if c is not None)
        print(f"  Valid cells: {n_valid}/{len(all_contacts)}  ({time.time()-t1:.1f}s)")

        print(f"\n[3/4] Building matrix (variant={variant})...")
        matrix = _build_matrix_per_cell(all_contacts, total_bins, variant=variant)
        chroms_avail = list(chrom_sizes.keys())

    cell_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]

    keep_float = save_float_csv or bool(hvb_configs)
    norm, int_data, coverage, mean_cov = coverage_normalize_and_log10(
        matrix, return_float=keep_float)

    print("\n[4/4] Saving...")
    summary_extra = {
        'mode': mode, 'variant': variant,
        'n_cells': int(len(files)), 'n_valid_cells': int(n_valid),
        'n_bins': int(total_bins), 'chromosomes': chroms_avail,
        'total_pixels':       int(stats['pixels']),
        'total_intra_pixels': int(stats['intra']),
        'total_contact_count': float(stats['counts']),
        'coverage_range':     [float(coverage.min()), float(coverage.max())],
        'mean_coverage':      float(mean_cov),
        'sparsity':           float(1.0 - np.count_nonzero(int_data) / max(int_data.size, 1)),
        'total_time':         float(time.time() - t0),
    }
    outputs = _write_all_hvb_variants(
        output_dir, resolution_str, norm, int_data,
        index=cell_ids, col_names=col_names,
        hvb_configs=hvb_configs, summary_extra=summary_extra,
        save_float_csv=save_float_csv,
        coverage_filter=hvb_coverage_filter)

    print("\n" + "=" * 80)
    print(f"{mode.upper()} PREPROCESSING COMPLETE in {time.time()-t0:.1f}s")
    print(f"Base matrix: {matrix.shape[0]} × {matrix.shape[1]}")
    for tag, (_cov, logf, _sumf, _summ) in outputs.items():
        print(f"  [{tag}] -> {logf}")
    print("=" * 80)
    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified scHi-C preprocessing (text / mcool / cool modes).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode',         required=True, choices=['text', 'mcool', 'cool'])
    parser.add_argument('--resolution',   default='100kb',
                        help='Bin resolution, e.g. 100kb, 500kb, 1Mb (default: 100kb)')
    parser.add_argument('--output_dir',   required=True)
    parser.add_argument('--chrom_sizes',  required=True, metavar='FILE',
                        help='UCSC-style chrom.sizes file (chrom\\tsize per line). '
                             'No bin counts are hardcoded.')
    parser.add_argument('--chroms',       default=None, metavar='CSV',
                        help='Comma-separated subset of chromosomes to keep '
                             '(default: all chroms in the chrom.sizes file).')

    parser.add_argument('--data_file',    type=str, help='[text mode] data.txt')
    parser.add_argument('--n_threads',    type=int, default=24,
                        help='[text mode] Polars thread count (default: 24)')

    parser.add_argument('--mcool_dir',    type=str,
                        help='[mcool/cool mode] Directory of .mcool / .cool files')
    parser.add_argument('--variant',      default='bin1',
                        choices=['bin1', 'both', 'marginal'],
                        help='[mcool/cool mode] Feature construction variant (default: bin1)')
    parser.add_argument('--limit',        type=int, default=None,
                        help='[mcool/cool mode] Limit number of files (testing)')

    parser.add_argument('--save_float_csv', action='store_true',
                        help='Also write the float-valued coverage CSV (off by default).')

    parser.add_argument('--hvb_configs',  nargs='+', default=None, metavar='SPEC',
                        help='Optional HVB specs (each writes hvb_<spec>/ subdir). '
                             'Examples: top1000 pct5 pct10 none. '
                             'If omitted, all bins are kept (default).')
    parser.add_argument('--hvb_coverage_filter', type=float, default=0.05,
                        help='Min per-bin coverage before HVB ranking (default 0.05).')

    args = parser.parse_args()
    chroms_keep = args.chroms.split(',') if args.chroms else None

    if args.mode == 'text':
        if not args.data_file:
            parser.error('--data_file is required when --mode text')
        if not os.path.exists(args.data_file):
            raise FileNotFoundError(f"Data file not found: {args.data_file}")
        preprocess_text(args.data_file, args.resolution, args.output_dir,
                        chrom_sizes_file=args.chrom_sizes,
                        n_threads=args.n_threads,
                        save_float_csv=args.save_float_csv,
                        hvb_configs=args.hvb_configs,
                        hvb_coverage_filter=args.hvb_coverage_filter,
                        chroms_keep=chroms_keep)
    else:
        if not args.mcool_dir:
            parser.error('--mcool_dir is required when --mode mcool/cool')
        if not os.path.isdir(args.mcool_dir):
            raise FileNotFoundError(f"directory not found: {args.mcool_dir}")
        preprocess_mcool(args.mcool_dir, args.resolution, args.output_dir,
                         chrom_sizes_file=args.chrom_sizes,
                         variant=args.variant, n_cells_limit=args.limit,
                         save_float_csv=args.save_float_csv,
                         hvb_configs=args.hvb_configs,
                         hvb_coverage_filter=args.hvb_coverage_filter,
                         chroms_keep=chroms_keep,
                         mode=args.mode)


if __name__ == '__main__':
    main()
