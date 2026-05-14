#!/usr/bin/env python3
# ============================================================================
#  GraphPerf-RT — Synthetic-prediction smoke-test fixture
# ============================================================================
#  Generates a minimal NPZ that mimics the schema of `test_predictions.npz`
#  written by `nig/train.py` and `gat/scripts/train.py`.  Used to **smoke-test**
#  the plotting pipeline (`figures/plot_figures.py`) without rerunning the
#  full GAT/NIG training.
#
#  Schema (matches what plot_figures.py expects):
#      predictions : (N,)
#      targets     : (N,)
#      stds        : (N,)        predictive std-dev = sqrt(beta / (alpha-1))
#      nu          : (N,)        NIG  ν parameter   (epistemic evidence)
#      alpha       : (N,)        NIG  α parameter
#      beta        : (N,)        NIG  β parameter
#      benchmarks  : (N,) <U…>   per-sample benchmark name (used by
#                                gat/scripts/scalability_analysis.py)
#
#  The values are statistically reasonable but not fitted — DO NOT publish
#  numbers from this fixture.
#
#  Usage
#  -----
#      python data/generate_sample_predictions.py             # default 1 000 samples
#      python data/generate_sample_predictions.py --n 5000    # custom size
# ============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',          type=int, default=1_000,
                    help='number of synthetic samples (default 1 000)')
    ap.add_argument('--seed',       type=int, default=42)
    ap.add_argument('--inventory',  type=Path,
                    default=Path(__file__).parent / 'benchmark_inventory.csv',
                    help='benchmark inventory CSV')
    ap.add_argument('--output',     type=Path,
                    default=Path(__file__).parent / 'sample_test_predictions.npz')
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    targets = rng.uniform(low=-2.0, high=4.0, size=args.n)            # log-makespan
    noise   = rng.normal(scale=0.30, size=args.n)
    preds   = targets + noise

    # NIG parameters reasonable for hp_036 (the paper's best config)
    alpha   = rng.uniform(low=2.5,  high=4.0,  size=args.n)
    beta    = rng.uniform(low=0.05, high=0.20, size=args.n)
    nu      = rng.uniform(low=4.0,  high=8.0,  size=args.n)
    stds    = np.sqrt(beta / (alpha - 1))

    # Per-sample benchmark name drawn from the inventory
    inv = pd.read_csv(args.inventory)
    benches = rng.choice(inv['benchmark'].to_numpy(), size=args.n, replace=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output,
             predictions=preds,
             targets=targets,
             stds=stds,
             nu=nu,
             alpha=alpha,
             beta=beta,
             benchmarks=benches.astype('<U24'))
    print(f'Wrote {args.output}  ({args.n} samples, '
          f'{args.output.stat().st_size / 1024:.1f} KB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
