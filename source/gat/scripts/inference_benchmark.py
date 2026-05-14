#!/usr/bin/env python3
# ============================================================================
#  GraphPerf-RT — Inference latency / memory / method benchmark
#  IJCAI-ECAI 2026 supplementary §"Inference Latency and Memory Footprint"
# ============================================================================
#  Reproduces:
#      Table tab:inference-latency      (§A.4) — latency / RAM / model size
#                                                across TX2 / Orin NX / RUBIK Pi
#      Table tab:latency-comparison     (§A.4) — evidential vs ensemble vs
#                                                MC-Dropout cost comparison
#
#  Two modes
#  ---------
#      --mode platform     run a microbenchmark on the *current* host and
#                          append a row to the platform-comparison CSV.
#                          Used to populate Table 14 of the supplementary
#                          (run once on each Jetson / RUBIK Pi).
#
#      --mode methods      compare evidential vs ensemble vs MC-Dropout on
#                          the *same* host (Table 15).  All methods load the
#                          same GAT backbone; ensemble adds a 5×-replica
#                          forward-pass loop, MC-Dropout enables eval-time
#                          dropout and runs 32 stochastic forwards.
#
#  Outputs
#  -------
#      results/inference_latency.csv   accumulated across hosts
#      results/inference_methods.csv   one row per method on the current host
#      results/*.tex                   drop-in LaTeX tables for the appendix
#
#  Usage
#  -----
#      export GRAPHPERF_RT_ROOT="$(pwd)"
#      python gat/scripts/inference_benchmark.py --mode platform \\
#          --checkpoint gat/experiments/exp_v2_gat_log/best.pth \\
#          --platform-name 'TX2 (GPU)' --device cuda
#
#      python gat/scripts/inference_benchmark.py --mode methods \\
#          --checkpoint gat/experiments/exp_v2_gat_log/best.pth \\
#          --device cuda
# ============================================================================

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

try:
    import psutil
except ImportError:                 # noqa: BLE001
    psutil = None                   # peak-RAM measurement becomes optional

import torch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('inference_benchmark')

REPO_ROOT = Path(os.environ.get(
    'GRAPHPERF_RT_ROOT',
    Path(__file__).resolve().parents[2])).resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_ram_mb() -> float:
    """Best-effort peak resident-set size (MB)."""
    if psutil is None:
        return float('nan')
    return psutil.Process().memory_info().rss / (1024 ** 2)


def _model_size_mb(checkpoint: Path) -> float:
    return checkpoint.stat().st_size / (1024 ** 2)


def _load_model(checkpoint: Path, device: str) -> torch.nn.Module:
    """Load the GAT backbone (TorchScript first, plain state_dict fallback)."""
    sys.path.insert(0, str(REPO_ROOT / 'gat'))
    sys.path.insert(0, str(REPO_ROOT / 'gat' / 'src'))
    try:
        return torch.jit.load(str(checkpoint), map_location=device).eval()
    except Exception:                # noqa: BLE001
        from src.models.gat import create_model       # type: ignore
        ckpt = torch.load(checkpoint, map_location=device)
        model = create_model(ckpt.get('config', {}))
        model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
        return model.to(device).eval()


def _make_dummy_batch(batch_size: int, device: str):
    """Match the production input signature: 8 nodes × 3 features + global."""
    nodes  = torch.randn(batch_size * 8, 3, device=device)
    batch  = torch.arange(batch_size, device=device).repeat_interleave(8)
    edges  = torch.tensor(
        [[i, j] for i in range(8) for j in range(8) if i != j],
        device=device).t().contiguous()
    edges  = torch.cat([edges + 8 * b for b in range(batch_size)], dim=1)
    glob   = torch.randn(batch_size, 69, device=device)
    return nodes, edges, batch, glob


def _timed(fn: Callable[[], torch.Tensor], n_warmup: int = 20,
           n_iter: int = 100) -> float:
    """Median wall-clock latency in milliseconds."""
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Mode A — platform comparison (Table tab:inference-latency)
# ---------------------------------------------------------------------------

def run_platform(args) -> dict:
    """Benchmark batch-1 and batch-16 on the current host."""
    device = args.device
    model  = _load_model(args.checkpoint, device)

    b1 = _make_dummy_batch(1,  device)
    b16 = _make_dummy_batch(16, device)

    with torch.no_grad():
        lat_b1  = _timed(lambda: model(*b1))
        lat_b16 = _timed(lambda: model(*b16))

    return {
        'platform':   args.platform_name,
        'batch_1_ms': round(lat_b1, 2),
        'batch_16_ms': round(lat_b16, 2),
        'peak_ram_mb': round(_peak_ram_mb(), 1),
        'model_size_mb': round(_model_size_mb(args.checkpoint), 2),
    }


# ---------------------------------------------------------------------------
# Mode B — method comparison (Table tab:latency-comparison)
# ---------------------------------------------------------------------------

def run_methods(args) -> List[dict]:
    """Compare evidential vs deep-ensemble vs MC-Dropout on the same host."""
    device = args.device
    model  = _load_model(args.checkpoint, device)
    inputs = _make_dummy_batch(1, device)

    rows: List[dict] = []

    # ---- Evidential (single forward pass) -----------------------------------
    with torch.no_grad():
        rows.append({
            'method':       'GraphPerf-RT (evidential)',
            'latency_ms':    round(_timed(lambda: model(*inputs)), 2),
            'peak_ram_mb':   round(_peak_ram_mb(), 1),
            'uncertainty':   'Yes',
        })

    # ---- Deep ensemble (5× replicas) ----------------------------------------
    ensemble = [_load_model(args.checkpoint, device) for _ in range(5)]

    def _ensemble_forward():
        with torch.no_grad():
            return torch.stack([m(*inputs) for m in ensemble]).mean(0)

    rows.append({
        'method':       'Deep Ensemble (5×)',
        'latency_ms':   round(_timed(_ensemble_forward), 2),
        'peak_ram_mb':  round(_peak_ram_mb(), 1),
        'uncertainty':  'Yes',
    })

    # ---- MC-Dropout (32 stochastic forwards) --------------------------------
    model.train()                                     # keep dropout active

    def _mc_forward():
        with torch.no_grad():
            return torch.stack([model(*inputs) for _ in range(32)]).mean(0)

    rows.append({
        'method':       'MC-Dropout (32×)',
        'latency_ms':   round(_timed(_mc_forward), 2),
        'peak_ram_mb':  round(_peak_ram_mb(), 1),
        'uncertainty':  'Yes',
    })

    model.eval()
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _append_csv(path: Path, row, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    rows = row if isinstance(row, list) else [row]
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if fresh:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info("Wrote %d row(s) → %s", len(rows), path)


def _emit_tex(rows: List[dict], path: Path, headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r'\begin{tabular}{@{}l' + 'c' * (len(headers) - 1) + r'@{}}',
        r'\toprule',
        ' & '.join(rf'\textbf{{{h}}}' for h in headers) + r' \\',
        r'\midrule',
    ]
    for r in rows:
        lines.append(' & '.join(str(r[h.lower().replace(' ', '_')])
                                for h in headers) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    path.write_text('\n'.join(lines))
    log.info("Wrote LaTeX table → %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['platform', 'methods'], required=True)
    ap.add_argument('--checkpoint',     type=Path, required=True)
    ap.add_argument('--platform-name',  type=str, default=os.uname().nodename,
                    help='label written into the CSV (e.g. "TX2 (GPU)")')
    ap.add_argument('--device',         type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output',         type=Path,
                    default=REPO_ROOT / 'gat' / 'results')
    args = ap.parse_args(argv)

    if args.mode == 'platform':
        row = run_platform(args)
        log.info("%s", row)
        _append_csv(args.output / 'inference_latency.csv', row,
                    fieldnames=list(row.keys()))
    else:
        rows = run_methods(args)
        for r in rows:
            log.info("%s", r)
        _append_csv(args.output / 'inference_methods.csv', rows,
                    fieldnames=list(rows[0].keys()))
        _emit_tex(rows,
                  args.output / 'inference_methods.tex',
                  headers=['Method', 'Latency_ms', 'Peak_RAM_mb', 'Uncertainty'])

    return 0


if __name__ == '__main__':
    sys.exit(main())
