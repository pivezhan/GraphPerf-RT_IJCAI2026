#!/usr/bin/env python3
# ============================================================================
#  GraphPerf-RT — Per-benchmark scalability analysis
#  IJCAI-ECAI 2026 supplementary §"Scalability Analysis"
# ============================================================================
#  Reproduces:
#      Table tab:per-benchmark-scalability  (28 benchmarks × 7 metrics)
#      Figure fig8_scalability_scatter       (MAPE vs cyclomatic complexity)
#
#  Pipeline
#  --------
#  1. Loads test predictions from `gat/experiments/<exp>/test_predictions.npz`.
#  2. Joins each prediction with its benchmark metadata (cyclomatic complexity,
#     loop count, branch count) read from the bundled `data/cfg_metrics.csv`
#     (extracted verbatim from supplementary tab:per-benchmark-scalability).
#  3. Aggregates per-benchmark R², MAPE, and mean epistemic σ.
#  4. Tags each benchmark with a complexity tier:
#         Low    : cyclomatic <  15
#         Medium : 15 ≤ cyclomatic ≤ 30
#         High   : cyclomatic >  30
#  5. Writes:
#         results/per_benchmark.csv   ← consumed by figures/plot_figures.py
#         results/per_benchmark.tex   ← drop-in LaTeX table for the appendix
#
#  Usage
#  -----
#      export GRAPHPERF_RT_ROOT="$(pwd)"
#      python gat/scripts/scalability_analysis.py \\
#          --predictions gat/experiments/exp_v2_gat_log/test_predictions.npz \\
#          --cfg-metrics data/cfg_metrics.csv \\
#          --output gat/results
#
#  For a smoke-test without trained checkpoints, point ``--predictions`` at
#  the synthetic fixture:
#      python data/generate_sample_predictions.py
#      python gat/scripts/scalability_analysis.py \\
#          --predictions data/sample_test_predictions.npz
# ============================================================================

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('scalability_analysis')

REPO_ROOT = Path(os.environ.get(
    'GRAPHPERF_RT_ROOT',
    Path(__file__).resolve().parents[2])).resolve()


def tier(cyclomatic: int) -> str:
    """Map cyclomatic complexity to the tier labels used in the supplementary."""
    if cyclomatic < 15:
        return 'Low'
    if cyclomatic <= 30:
        return 'Medium'
    return 'High'


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def _mape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean absolute percentage error. Inputs are log-makespan; we exponentiate
    before computing the percent error so the supplementary numbers match."""
    yt, ypt = np.exp(y), np.exp(y_hat)
    nz = yt > 0
    return float(np.mean(np.abs(ypt[nz] - yt[nz]) / yt[nz]))


def _epistemic_sigma(p: dict) -> np.ndarray:
    """If NIG parameters are present, return the epistemic σ; else fall back to
    the predictive standard deviation column ('stds')."""
    nu, alpha, beta = p.get('nu'), p.get('alpha'), p.get('beta')
    if nu is not None and alpha is not None and beta is not None:
        return np.sqrt(beta / (nu * (alpha - 1)))
    if 'stds' in p:
        return p['stds']
    return np.full(len(p['predictions']), np.nan)


def aggregate_per_benchmark(predictions_npz: Path,
                            cfg_metrics_csv: Path) -> pd.DataFrame:
    """Compute the per-benchmark accuracy / complexity table."""
    if not predictions_npz.is_file():
        raise FileNotFoundError(f"missing {predictions_npz}")
    if not cfg_metrics_csv.is_file():
        raise FileNotFoundError(f"missing {cfg_metrics_csv} — see "
                                f"supplementary §'Graph Extraction Pipeline' for the "
                                f"expected schema (benchmark,cyclomatic,loops,branches)")

    p = dict(np.load(predictions_npz, allow_pickle=True))
    if 'benchmarks' not in p:
        raise KeyError("test_predictions.npz must include a 'benchmarks' field "
                       "(per-sample benchmark name) — re-run gat/scripts/evaluate.py "
                       "with --emit-benchmarks")

    y      = p['targets']
    y_hat  = p['predictions']
    bench  = np.asarray(p['benchmarks']).astype(str)
    sigma  = _epistemic_sigma(p)
    cfg    = pd.read_csv(cfg_metrics_csv).set_index('benchmark')

    rows = []
    for b in sorted(set(bench)):
        m = bench == b
        if m.sum() < 2:
            continue
        if b not in cfg.index:
            log.warning("CFG metrics missing for benchmark '%s' — skipping", b)
            continue
        row = {
            'benchmark':  b,
            'cyclomatic': int(cfg.loc[b, 'cyclomatic']),
            'loops':      int(cfg.loc[b, 'loops']),
            'branches':   int(cfg.loc[b, 'branches']),
            'r2':         _r2(y[m], y_hat[m]),
            'mape':       _mape(y[m], y_hat[m]),
            'sigma_epi':  float(np.nanmean(sigma[m])),
        }
        row['tier'] = tier(row['cyclomatic'])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(['tier', 'cyclomatic'], inplace=True)
    return df


def emit_latex(df: pd.DataFrame, out_tex: Path) -> None:
    """Write the LaTeX table referenced as tab:per-benchmark-scalability."""
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r'\begin{tabular}{@{}llrrrrrr@{}}',
        r'\toprule',
        r'\textbf{Tier} & \textbf{Bench.} & \textbf{Cycl.} & \textbf{Loops} & '
        r'\textbf{Br.} & \textbf{$R^2$} & \textbf{MAPE} & '
        r'\textbf{$\bar{\sigma}_{\text{epi}}$} \\',
        r'\midrule',
    ]
    last_tier = None
    for _, r in df.iterrows():
        sep = r'\midrule' if (last_tier is not None and r['tier'] != last_tier) else ''
        if sep:
            lines.append(sep)
        lines.append(
            f"{r['tier'] if r['tier'] != last_tier else ''} & "
            f"{r['benchmark']} & {r['cyclomatic']} & {r['loops']} & "
            f"{r['branches']} & {r['r2']:.2f} & {r['mape']*100:.1f}\\% & "
            f"{r['sigma_epi']:.2f} \\\\"
        )
        last_tier = r['tier']
    lines += [r'\bottomrule', r'\end{tabular}']
    out_tex.write_text('\n'.join(lines))
    log.info("Wrote LaTeX table → %s", out_tex)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', type=Path,
                    default=REPO_ROOT / 'gat' / 'experiments' / 'exp_v2_gat_log'
                            / 'test_predictions.npz')
    ap.add_argument('--cfg-metrics', type=Path,
                    default=REPO_ROOT / 'data' / 'cfg_metrics.csv',
                    help='per-benchmark complexity metrics (default: bundled '
                         'data/cfg_metrics.csv extracted from supplementary '
                         'tab:per-benchmark-scalability)')
    ap.add_argument('--output',      type=Path,
                    default=REPO_ROOT / 'gat' / 'results')
    args = ap.parse_args(argv)

    df = aggregate_per_benchmark(args.predictions, args.cfg_metrics)
    args.output.mkdir(parents=True, exist_ok=True)

    csv_path = args.output / 'per_benchmark.csv'
    df.to_csv(csv_path, index=False)
    log.info("Wrote per-benchmark CSV → %s", csv_path)

    emit_latex(df, args.output / 'per_benchmark.tex')

    rho = df['cyclomatic'].corr(df['mape'], method='spearman')
    log.info("Spearman ρ(cyclomatic, MAPE) = %.3f", rho)
    return 0


if __name__ == '__main__':
    sys.exit(main())
