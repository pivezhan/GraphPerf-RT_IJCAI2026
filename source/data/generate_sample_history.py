#!/usr/bin/env python3
"""Generate a minimal training_history.json for plot smoke-tests."""
import argparse, json, math, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=150)
ap.add_argument('--seed',   type=int, default=42)
ap.add_argument('--output', type=Path,
                default=Path(__file__).parent / 'sample_training_history.json')
args = ap.parse_args()

random.seed(args.seed)
def decay(start, end, n, jitter=0.02):
    return [end + (start - end) * math.exp(-3 * i / n) + random.uniform(-jitter, jitter)
            for i in range(n)]
def grow(start, end, n, jitter=0.01):
    return [start + (end - start) * (1 - math.exp(-3 * i / n)) + random.uniform(-jitter, jitter)
            for i in range(n)]

history = {
    'train_loss': decay(2.5, 0.55,  args.epochs, 0.04),
    'val_loss':   decay(2.4, 0.62,  args.epochs, 0.05),
    'train_r2':   grow(-0.2, 0.84,  args.epochs, 0.01),
    'val_r2':     grow(-0.3, 0.81,  args.epochs, 0.015),
    'val_ece':    decay(0.55, 0.40, args.epochs, 0.01),
}

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(history, indent=2))
print(f"Wrote {args.output}")
