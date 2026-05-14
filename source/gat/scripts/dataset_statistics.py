#!/usr/bin/env python3
# ============================================================================
#  GraphPerf-RT — Per-platform dataset statistics
#  IJCAI-ECAI 2026 supplementary §"Per-Platform Dataset Statistics"
# ============================================================================
#  Reproduces:
#      Table tab:per-platform-stats  (samples / benchmarks / freq levels /
#                                     train-val-test split sizes per platform)
#
#  Inputs
#  ------
#  Reads the normalized per-platform CSVs that the preprocessing pipeline
#  emits in `gat/data/processed/<platform>.csv`.  Each row carries the columns
#  benchmark, platform, frequency_level, and the makespan target.
#
#  Outputs
#  -------
#      results/dataset_statistics.csv   tidy per-platform statistics
#      results/dataset_statistics.tex   drop-in LaTeX table for the appendix
#
#  Usage
#  -----
#      export GRAPHPERF_RT_ROOT="$(pwd)"
#      python gat/scripts/dataset_statistics.py \\
#          --processed-dir gat/data/processed \\
#          --splits-dir    gat/data/splits \\
#          --output        gat/results
# ============================================================================

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('dataset_statistics')

REPO_ROOT = Path(os.environ.get(
    'GRAPHPERF_RT_ROOT',
    Path(__file__).resolve().parents[2])).resolve()


def _load_platform_csvs(processed_dir: Path) -> pd.DataFrame:
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"missing {processed_dir}")
    frames = []
    for csv in sorted(processed_dir.glob('*.csv')):
        df = pd.read_csv(csv)
        df['platform'] = df.get('platform', csv.stem)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no CSVs found in {processed_dir}")
    return pd.concat(frames, ignore_index=True)


def _split_sizes(splits_dir: Path) -> pd.DataFrame:
    """Return a (platform × split) → sample count table."""
    if not splits_dir.is_dir():
        log.warning("splits dir %s not found — split sizes will be empty",
                    splits_dir)
        return pd.DataFrame()
    rows = []
    for split in ('train', 'val', 'test'):
        f = splits_dir / f'{split}.csv'
        if not f.is_file():
            log.warning("missing %s", f)
            continue
        df = pd.read_csv(f)
        for plat, g in df.groupby('platform'):
            rows.append({'platform': plat, 'split': split, 'count': len(g)})
    return pd.DataFrame(rows)


def aggregate(processed_dir: Path, splits_dir: Path) -> pd.DataFrame:
    """Per-platform counts: total samples, distinct benchmarks, freq levels,
    and (where available) train/val/test split sizes."""
    df = _load_platform_csvs(processed_dir)
    grouped = df.groupby('platform').agg(
        samples       =('benchmark',       'size'),
        benchmarks    =('benchmark',       'nunique'),
        freq_levels   =('frequency_level', 'nunique'),
    ).reset_index()

    splits = _split_sizes(splits_dir)
    if not splits.empty:
        wide = splits.pivot(index='platform', columns='split', values='count').fillna(0)
        wide = wide.rename(columns={'train': 'n_train', 'val': 'n_val', 'test': 'n_test'})
        grouped = grouped.merge(wide.reset_index(), on='platform', how='left')

    return grouped


def emit_latex(df: pd.DataFrame, out_tex: Path) -> None:
    """LaTeX table matching the supplementary appendix style."""
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    cols = ['platform', 'samples', 'benchmarks', 'freq_levels',
            'n_train', 'n_val', 'n_test']
    cols = [c for c in cols if c in df.columns]
    lines = [
        r'\begin{tabular}{@{}l' + 'r' * (len(cols) - 1) + r'@{}}',
        r'\toprule',
        ' & '.join(rf'\textbf{{{c.replace("_", " ")}}}' for c in cols) + r' \\',
        r'\midrule',
    ]
    for _, r in df.iterrows():
        cells = [str(r[c]) if isinstance(r[c], str) else f"{int(r[c]):,d}"
                 for c in cols]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    out_tex.write_text('\n'.join(lines))
    log.info("Wrote LaTeX table → %s", out_tex)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed-dir', type=Path,
                    default=REPO_ROOT / 'gat' / 'data' / 'processed')
    ap.add_argument('--splits-dir',    type=Path,
                    default=REPO_ROOT / 'gat' / 'data' / 'splits')
    ap.add_argument('--output',        type=Path,
                    default=REPO_ROOT / 'gat' / 'results')
    args = ap.parse_args(argv)

    df = aggregate(args.processed_dir, args.splits_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / 'dataset_statistics.csv', index=False)
    emit_latex(df, args.output / 'dataset_statistics.tex')
    log.info("Per-platform statistics:\n%s", df.to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
