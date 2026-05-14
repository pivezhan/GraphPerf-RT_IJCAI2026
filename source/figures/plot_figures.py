#!/usr/bin/env python3
# ============================================================================
#  GraphPerf-RT — Unified Figure Regenerator
#  IJCAI-ECAI 2026 (AI4Tech) — main paper + supplementary
# ============================================================================
#  Single CLI entry point that re-emits every Python-generated figure in the
#  camera-ready paper and supplementary appendix.  It replaces five legacy
#  scripts:
#      figures/generate_nig_figures.py
#      nig/regenerate_charts.py
#      nig/visualize_attention.py
#      rl/jetson_tx2/generate_paper_figures.py
#      rl/jetson_tx2/fixed_plot.py
#
#  Subcommands map 1:1 to figure filenames.  The supplementary appendix
#  (`paper/camera_ready/ijcai_appendix_red.tex`) references twelve PNGs by
#  exact name; this script emits each of them with the matching filename so
#  the LaTeX `\includegraphics{}` calls resolve unchanged.
#
#  Section 4 — GAT surrogate
#      gat-training            Fig. 4: loss / R² / scatter
#
#  Section 5 — NIG evidential head (supplementary §"NIG Evidential Results")
#      fig5a-training-loss     fig5a_training_loss.png        (training loss)
#      fig5b-r2-score          fig5b_r2_score.png             (R² progression)
#      fig5c-predictions       fig5c_predictions_scatter.png  (preds vs true)
#      fig5d-uncertainty-error fig5d_uncertainty_vs_error.png
#      fig5e-calibration       fig5e_calibration_curve.png    (ECE)
#      fig5f-ece-training      fig5f_ece_training.png         (ECE over epochs)
#      fig6a-picp              fig6a_picp_coverage.png        (PICP vs coverage)
#      fig6b-mpiw              fig6b_mpiw_confidence.png      (MPIW vs confidence)
#      fig7a-uncertainty-trend fig7a_uncertainty_error_trend.png
#      fig7b-uncertainty-pie   fig7b_uncertainty_pie.png      (94/6 split)
#      fig7c-binned-calibration fig7c_binned_calibration.png
#      attention                Per-sample attention heatmaps
#      hyperparam-sweep         48-config sweep summary
#
#  Section 6 — RL integration / scalability
#      rl-curves               Fig. 5a: episode-return learning curves
#      rl-thermal              Fig. 5d / Fig. 8: thermal-violation tracking
#      rl-energy               Fig. 5b: energy-vs-makespan trade-off
#      fig8-scalability        fig8_scalability_scatter.png   (MAPE vs cyclomatic)
#
#  Group commands
#      nig-all                  All eleven NIG figures (fig5*, fig6*, fig7*)
#      supplementary            All twelve supplementary figures
#      all                      Every figure above
#
#  Usage
#  -----
#      export GRAPHPERF_RT_ROOT="$(pwd)"
#      python figures/plot_figures.py supplementary --output figures/output
#      python figures/plot_figures.py fig7b-uncertainty-pie
#      python figures/plot_figures.py all
#
#  Inputs are JSON / NPZ artefacts produced by the training scripts; this
#  module never re-trains models.
# ============================================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------

REPO_ROOT = Path(os.environ.get(
    'GRAPHPERF_RT_ROOT',
    Path(__file__).resolve().parents[1])).resolve()

DEFAULT_NIG_EXPERIMENT = REPO_ROOT / 'nig' / 'experiments' / \
    'hp_036_lmse20.0_lns0.001_lr0.0005_hd128'
DEFAULT_GAT_EXPERIMENT = REPO_ROOT / 'gat' / 'experiments' / 'exp_v2_gat_log'
DEFAULT_RL_RESULTS = REPO_ROOT / 'rl' / 'on_device' / 'save_model'
DEFAULT_SCALABILITY = REPO_ROOT / 'gat' / 'results' / 'per_benchmark.csv'
DEFAULT_OUTPUT = REPO_ROOT / 'figures' / 'output'

# Bundled fixtures (small, version-controlled).  Used as a fallback so the
# plot pipeline can be smoke-tested before the GAT/NIG models have been trained.
BUNDLED_DATA = REPO_ROOT / 'data'
BUNDLED_SWEEP_CSV  = BUNDLED_DATA / 'nig_hyperparam_sweep.csv'
BUNDLED_NIG_NPZ    = BUNDLED_DATA / 'sample_test_predictions.npz'
BUNDLED_NIG_HISTORY = BUNDLED_DATA / 'sample_training_history.json'
BUNDLED_NIG_RESULTS = BUNDLED_DATA / 'sample_results.json'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('plot_figures')


# ---------------------------------------------------------------------------
# Style configuration — matches the IJCAI camera-ready typography
# ---------------------------------------------------------------------------

STYLE = {
    'font_family':  'serif',
    'font_serif':   ['Libertinus Serif', 'Nimbus Roman', 'Times New Roman', 'serif'],
    'font_sans':    ['Libertinus Sans',   'DejaVu Sans', 'sans-serif'],
    'title_size':   16,
    'label_size':   14,
    'tick_size':    12,
    'legend_size':  12,
    'linewidth':    2.0,
    'markersize':   6,
    'grid_alpha':   0.3,
    'grid_style':   '--',
    'dpi':          300,
    'colors': {
        'primary':   '#2c3e50',
        'secondary': '#3498db',
        'accent':    '#e74c3c',
        'positive':  '#27ae60',
        'warning':   '#f39c12',
        'low':       '#3498db',
        'med':       '#e67e22',
        'high':      '#e74c3c',
    },
}

#: Method colours used across all RL plots.
METHOD_COLOURS = {
    'SAMFRL':       '#7f8c8d',
    'SAMBRL':       '#3498db',
    'MAMFRL_D3QN':  '#e67e22',
    'MAMBRL_D3QN':  '#27ae60',
}


def configure_style() -> None:
    """Apply the project-wide matplotlib style (Libertinus when present)."""
    plt.rcdefaults()

    libertinus_regular = os.environ.get('LIBERTINUS_REGULAR')
    libertinus_bold    = os.environ.get('LIBERTINUS_BOLD')
    if libertinus_regular and Path(libertinus_regular).exists():
        font_manager.fontManager.addfont(libertinus_regular)
        if libertinus_bold and Path(libertinus_bold).exists():
            font_manager.fontManager.addfont(libertinus_bold)
        STYLE['font_serif'].insert(0, 'Libertinus Serif')

    plt.rcParams['font.family']     = STYLE['font_family']
    plt.rcParams['font.serif']      = STYLE['font_serif']
    plt.rcParams['font.sans-serif'] = STYLE['font_sans']
    plt.rcParams['axes.titlesize']  = STYLE['title_size']
    plt.rcParams['axes.labelsize']  = STYLE['label_size']
    plt.rcParams['xtick.labelsize'] = STYLE['tick_size']
    plt.rcParams['ytick.labelsize'] = STYLE['tick_size']
    plt.rcParams['legend.fontsize'] = STYLE['legend_size']
    plt.rcParams['lines.linewidth'] = STYLE['linewidth']
    plt.rcParams['lines.markersize'] = STYLE['markersize']
    plt.rcParams['grid.alpha']      = STYLE['grid_alpha']
    plt.rcParams['grid.linestyle']  = STYLE['grid_style']
    plt.rcParams['figure.dpi']      = STYLE['dpi']
    plt.rcParams['savefig.dpi']     = STYLE['dpi']
    plt.rcParams['savefig.bbox']    = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        log.warning("Missing JSON: %s", path)
        return None
    with path.open() as f:
        return json.load(f)


def _load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.is_file():
        log.warning("Missing NPZ: %s", path)
        return None
    return dict(np.load(path))


def load_nig_artefacts(exp_dir: Path) -> Dict:
    """Load JSON history / NPZ predictions written by ``nig/train.py``.

    Falls back to the bundled smoke-test fixtures under ``data/`` if the
    experiment directory is empty — useful for verifying the plot pipeline
    before the full NIG training has finished.
    """
    history = _load_json(exp_dir / 'training_history.json') \
              or _load_json(BUNDLED_NIG_HISTORY)
    results = _load_json(exp_dir / 'results.json') \
              or _load_json(BUNDLED_NIG_RESULTS)
    pred    = _load_npz(exp_dir / 'test_predictions.npz') \
              or _load_npz(BUNDLED_NIG_NPZ)
    if pred is BUNDLED_NIG_NPZ or any(
            v is None for v in (history, results, pred)):
        log.info("Using bundled NIG fixtures from %s", BUNDLED_DATA)
    return {'history': history, 'results': results, 'predictions': pred}


def load_gat_artefacts(exp_dir: Path) -> Dict:
    """Load JSON metrics / NPZ predictions written by ``gat/scripts/train.py``."""
    return {
        'history':     _load_json(exp_dir / 'history.json'),
        'metrics':     _load_json(exp_dir / 'metrics.json'),
        'predictions': _load_npz(exp_dir / 'test_predictions.npz'),
    }


def load_rl_results(results_dir: Path,
                    methods: Sequence[str] = ('SAMFRL', 'SAMBRL',
                                              'MAMFRL_D3QN', 'MAMBRL_D3QN'),
                    seeds: Sequence[int] = (42, 123, 456, 789, 1024)) -> Dict:
    """Load per-method, per-seed RL training logs."""
    out = {}
    for m in methods:
        seed_runs = []
        for s in seeds:
            ep = _load_json(results_dir / m / f'seed{s}' / 'episodes.json')
            if ep is not None:
                seed_runs.append(ep)
        if seed_runs:
            out[m] = seed_runs
    if not out:
        log.warning("No RL results found under %s", results_dir)
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, output: Path, name: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f'{name}.pdf'
    png = output / f'{name}.png'
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    log.info("Saved %s and %s", pdf.name, png.name)
    return pdf


def _shade_ci(ax, x, mean, std, colour, label=None):
    ax.plot(x, mean, color=colour, label=label)
    ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.15)


def _gauss_z(coverage: float) -> float:
    """Z-score for a symmetric Gaussian interval at ``coverage`` probability."""
    from scipy.stats import norm
    return float(norm.ppf(0.5 + coverage / 2))


# ---------------------------------------------------------------------------
# Section 4 — GAT surrogate (Fig. 4)
# ---------------------------------------------------------------------------

def plot_gat_training(exp_dir: Path, output: Path) -> Path:
    """Three-panel GAT training summary (Fig. 4 in the main paper)."""
    art = load_gat_artefacts(exp_dir)
    if art['history'] is None or art['predictions'] is None:
        raise FileNotFoundError(f"GAT artefacts missing in {exp_dir}")

    history = art['history']; pred = art['predictions']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    epochs = np.arange(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='train', color=STYLE['colors']['primary'])
    axes[0].plot(epochs, history['val_loss'],   label='val',   color=STYLE['colors']['accent'])
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('MSE loss'); axes[0].set_title('(a) Loss')
    axes[0].grid(True); axes[0].legend()

    if 'val_r2' in history:
        axes[1].plot(epochs, history['val_r2'], color=STYLE['colors']['secondary'])
        axes[1].set_xlabel('epoch'); axes[1].set_ylabel('R²')
        axes[1].set_title('(b) Validation $R^2$'); axes[1].grid(True)

    y = pred['targets']; y_hat = pred['predictions']
    axes[2].scatter(y, y_hat, s=8, alpha=0.4, color=STYLE['colors']['secondary'])
    lim = [min(y.min(), y_hat.min()), max(y.max(), y_hat.max())]
    axes[2].plot(lim, lim, 'k--', linewidth=1)
    axes[2].set_xlabel('true log-makespan'); axes[2].set_ylabel('predicted')
    axes[2].set_title('(c) Test fit'); axes[2].grid(True)

    fig.tight_layout()
    return _save(fig, output, 'fig4_gat_training')


# ---------------------------------------------------------------------------
# Supplementary §"NIG Evidential Regression Results" — Fig. 5a–f
# ---------------------------------------------------------------------------

def fig5a_training_loss(exp_dir: Path, output: Path) -> Path:
    """fig5a — train/val NIG loss curves over epochs."""
    art = load_nig_artefacts(exp_dir)
    if art['history'] is None:
        raise FileNotFoundError(f"NIG history missing in {exp_dir}")
    h = art['history']; epochs = np.arange(1, len(h['train_loss']) + 1)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(epochs, h['train_loss'], label='train', color=STYLE['colors']['primary'])
    ax.plot(epochs, h['val_loss'],   label='val',   color=STYLE['colors']['accent'])
    ax.set_yscale('log'); ax.set_xlabel('epoch'); ax.set_ylabel('NIG loss')
    ax.set_title('Training and validation loss'); ax.legend(); ax.grid(True)
    fig.tight_layout()
    return _save(fig, output, 'fig5a_training_loss')


def fig5b_r2_score(exp_dir: Path, output: Path) -> Path:
    """fig5b — R² progression during training."""
    art = load_nig_artefacts(exp_dir)
    if art['history'] is None or 'val_r2' not in art['history']:
        raise FileNotFoundError(f"val_r2 missing in NIG history at {exp_dir}")
    h = art['history']; epochs = np.arange(1, len(h['val_r2']) + 1)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(epochs, h['val_r2'], color=STYLE['colors']['secondary'])
    if 'train_r2' in h:
        ax.plot(epochs, h['train_r2'], color=STYLE['colors']['primary'],
                linestyle='--', alpha=0.6, label='train')
        ax.legend(['val', 'train'])
    ax.set_xlabel('epoch'); ax.set_ylabel(r'$R^2$')
    ax.set_title('$R^2$ progression'); ax.grid(True); ax.set_ylim(-0.1, 1.0)
    fig.tight_layout()
    return _save(fig, output, 'fig5b_r2_score')


def fig5c_predictions_scatter(exp_dir: Path, output: Path) -> Path:
    """fig5c — predicted vs true scatter (target $R^2 = 0.81$)."""
    art = load_nig_artefacts(exp_dir)
    if art['predictions'] is None:
        raise FileNotFoundError(f"NIG predictions missing in {exp_dir}")
    p = art['predictions']
    y, y_hat = p['targets'], p['predictions']
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y, y_hat, s=6, alpha=0.35, color=STYLE['colors']['secondary'])
    lim = [min(y.min(), y_hat.min()), max(y.max(), y_hat.max())]
    ax.plot(lim, lim, 'k--', linewidth=1, label='ideal')
    if art['results'] and 'r2' in art['results']:
        ax.text(0.05, 0.95, fr"$R^2 = {art['results']['r2']:.2f}$",
                transform=ax.transAxes, fontsize=14, verticalalignment='top')
    ax.set_xlabel('true log-makespan'); ax.set_ylabel('predicted')
    ax.set_title('Predicted vs.\\ true'); ax.grid(True); ax.legend()
    fig.tight_layout()
    return _save(fig, output, 'fig5c_predictions_scatter')


def fig5d_uncertainty_vs_error(exp_dir: Path, output: Path) -> Path:
    """fig5d — predicted σ vs absolute error."""
    art = load_nig_artefacts(exp_dir)
    if art['predictions'] is None:
        raise FileNotFoundError(f"NIG predictions missing in {exp_dir}")
    p = art['predictions']
    sigma = p.get('stds')
    if sigma is None:
        raise KeyError("'stds' missing from NIG predictions")
    err = np.abs(p['targets'] - p['predictions'])
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.scatter(sigma, err, s=6, alpha=0.4, color=STYLE['colors']['secondary'])
    rho = float(np.corrcoef(sigma, err)[0, 1]) if len(sigma) > 1 else float('nan')
    ax.set_xlabel(r'predicted $\sigma$'); ax.set_ylabel(r'$|y - \hat{y}|$')
    ax.set_title(f'Uncertainty vs.\\ error ($\\rho = {rho:.2f}$)'); ax.grid(True)
    fig.tight_layout()
    return _save(fig, output, 'fig5d_uncertainty_vs_error')


def fig5e_calibration_curve(exp_dir: Path, output: Path) -> Path:
    """fig5e — reliability diagram with ECE annotation."""
    art = load_nig_artefacts(exp_dir)
    if art['predictions'] is None:
        raise FileNotFoundError(f"NIG predictions missing in {exp_dir}")
    p = art['predictions']
    y, mu, sigma = p['targets'], p['predictions'], p['stds']
    nominal = np.linspace(0.05, 0.99, 25)
    picp = []
    for c in nominal:
        z = _gauss_z(c)
        picp.append(np.mean((y >= mu - z * sigma) & (y <= mu + z * sigma)))
    picp = np.array(picp)
    ece = float(np.mean(np.abs(picp - nominal)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(nominal, picp, 'o-', color=STYLE['colors']['primary'], label='GraphPerf-RT')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='ideal')
    ax.fill_between(nominal, nominal, picp,
                    where=picp >= nominal, color=STYLE['colors']['positive'], alpha=0.15)
    ax.fill_between(nominal, nominal, picp,
                    where=picp <  nominal, color=STYLE['colors']['accent'],   alpha=0.15)
    ax.text(0.05, 0.92, f'ECE = {ece:.3f}', transform=ax.transAxes, fontsize=14)
    ax.set_xlabel('nominal coverage'); ax.set_ylabel('empirical coverage')
    ax.set_title('Calibration curve'); ax.grid(True); ax.legend()
    fig.tight_layout()
    return _save(fig, output, 'fig5e_calibration_curve')


def fig5f_ece_training(exp_dir: Path, output: Path) -> Path:
    """fig5f — ECE over training epochs."""
    art = load_nig_artefacts(exp_dir)
    if art['history'] is None or 'val_ece' not in art['history']:
        raise FileNotFoundError(f"val_ece missing in NIG history at {exp_dir}")
    h = art['history']; epochs = np.arange(1, len(h['val_ece']) + 1)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(epochs, h['val_ece'], color=STYLE['colors']['warning'])
    ax.set_xlabel('epoch'); ax.set_ylabel('ECE')
    ax.set_title('Calibration error during training'); ax.grid(True)
    fig.tight_layout()
    return _save(fig, output, 'fig5f_ece_training')


# ---------------------------------------------------------------------------
# Supplementary — Fig. 6: PICP / MPIW
# ---------------------------------------------------------------------------

def fig6a_picp_coverage(exp_dir: Path, output: Path) -> Path:
    """fig6a — PICP vs expected coverage (the conservative-calibration plot)."""
    art = load_nig_artefacts(exp_dir)
    if art['predictions'] is None:
        raise FileNotFoundError(f"NIG predictions missing in {exp_dir}")
    p = art['predictions']
    y, mu, sigma = p['targets'], p['predictions'], p['stds']
    nominal = np.linspace(0.05, 0.99, 25)
    picp = [np.mean((y >= mu - _gauss_z(c) * sigma) & (y <= mu + _gauss_z(c) * sigma))
            for c in nominal]
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.plot(nominal, picp, 'o-', color=STYLE['colors']['primary'], label='PICP')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='ideal')
    ax.set_xlabel('expected coverage'); ax.set_ylabel('PICP')
    ax.set_title('PICP vs.\\ expected coverage'); ax.grid(True); ax.legend()
    fig.tight_layout()
    return _save(fig, output, 'fig6a_picp_coverage')


def fig6b_mpiw_confidence(exp_dir: Path, output: Path) -> Path:
    """fig6b — MPIW grows with confidence level."""
    art = load_nig_artefacts(exp_dir)
    if art['predictions'] is None:
        raise FileNotFoundError(f"NIG predictions missing in {exp_dir}")
    p = art['predictions']
    sigma = p['stds']
    nominal = np.linspace(0.05, 0.99, 25)
    mpiw = [float(2 * _gauss_z(c) * sigma.mean()) for c in nominal]
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.plot(nominal, mpiw, 's-', color=STYLE['colors']['secondary'])
    ax.set_xlabel('confidence level'); ax.set_ylabel('MPIW')
    ax.set_title('MPIW vs.\\ confidence'); ax.grid(True)
    fig.tight_layout()
    return _save(fig, output, 'fig6b_mpiw_confidence')


# ---------------------------------------------------------------------------
# Supplementary — Fig. 7: uncertainty decomposition
# ---------------------------------------------------------------------------

def _aleatoric_epistemic(p: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (aleatoric_var, epistemic_var) from NIG NPZ predictions."""
    nu, alpha, beta = p.get('nu'), p.get('alpha'), p.get('beta')
    if nu is None or alpha is None or beta is None:
        raise KeyError("NIG predictions must carry 'nu', 'alpha', 'beta'")
    aleatoric = beta / (alpha - 1)
    epistemic = beta / (nu * (alpha - 1))
    return aleatoric, epistemic


def fig7a_uncertainty_error_trend(exp_dir: Path, output: Path) -> Path:
    """fig7a — uncertainty bins → mean error (positive trend)."""
    art = load_nig_artefacts(exp_dir)
    if art['predictions'] is None:
        raise FileNotFoundError(f"NIG predictions missing in {exp_dir}")
    p = art['predictions']
    sigma = p['stds']
    err = np.abs(p['targets'] - p['predictions'])
    edges = np.quantile(sigma, np.linspace(0, 1, 11))
    centres = 0.5 * (edges[1:] + edges[:-1])
    means = [err[(sigma >= edges[i]) & (sigma < edges[i + 1])].mean()
             for i in range(10)]
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.plot(centres, means, 'o-', color=STYLE['colors']['accent'])
    ax.set_xlabel('predicted $\\sigma$ (binned)'); ax.set_ylabel('mean |error|')
    ax.set_title('Uncertainty–error trend'); ax.grid(True)
    fig.tight_layout()
    return _save(fig, output, 'fig7a_uncertainty_error_trend')


def fig7b_uncertainty_pie(exp_dir: Path, output: Path) -> Path:
    """fig7b — aleatoric / epistemic share of total predictive variance."""
    art = load_nig_artefacts(exp_dir)
    if art['predictions'] is None:
        raise FileNotFoundError(f"NIG predictions missing in {exp_dir}")
    aleatoric, epistemic = _aleatoric_epistemic(art['predictions'])
    a, e = float(aleatoric.mean()), float(epistemic.mean())
    pct_a = a / (a + e); pct_e = e / (a + e)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.pie([pct_a, pct_e], labels=[f'aleatoric ({pct_a*100:.0f}%)',
                                    f'epistemic ({pct_e*100:.0f}%)'],
           colors=[STYLE['colors']['secondary'], STYLE['colors']['accent']],
           startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title('Uncertainty decomposition')
    fig.tight_layout()
    return _save(fig, output, 'fig7b_uncertainty_pie')


def fig7c_binned_calibration(exp_dir: Path, output: Path) -> Path:
    """fig7c — binned calibration: per-bin observed vs nominal coverage."""
    art = load_nig_artefacts(exp_dir)
    if art['predictions'] is None:
        raise FileNotFoundError(f"NIG predictions missing in {exp_dir}")
    p = art['predictions']
    y, mu, sigma = p['targets'], p['predictions'], p['stds']
    edges = np.quantile(sigma, np.linspace(0, 1, 11))
    centres = 0.5 * (edges[1:] + edges[:-1])
    cov = []
    for i in range(10):
        m = (sigma >= edges[i]) & (sigma < edges[i + 1])
        if m.sum() == 0:
            cov.append(np.nan); continue
        z = _gauss_z(0.95)
        in_int = (y[m] >= mu[m] - z * sigma[m]) & (y[m] <= mu[m] + z * sigma[m])
        cov.append(float(in_int.mean()))
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.bar(np.arange(10), cov, color=STYLE['colors']['secondary'],
           edgecolor='black', linewidth=0.8)
    ax.axhline(0.95, color='k', linestyle='--', label='95% target')
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels([f'{c:.2f}' for c in centres], rotation=45, ha='right')
    ax.set_xlabel('uncertainty bin'); ax.set_ylabel('observed coverage')
    ax.set_title('Binned calibration'); ax.legend(); ax.grid(True, axis='y')
    fig.tight_layout()
    return _save(fig, output, 'fig7c_binned_calibration')


# ---------------------------------------------------------------------------
# Supplementary attention heatmaps (no specific filename in the appendix)
# ---------------------------------------------------------------------------

def plot_attention(exp_dir: Path, output: Path, num_samples: int = 5) -> Optional[Path]:
    """Per-sample attention-weight heatmaps."""
    npz_path = exp_dir / 'attention_weights.npz'
    if not npz_path.is_file():
        log.warning("attention_weights.npz not found in %s — skipping", exp_dir)
        return None
    data = np.load(npz_path)
    weights = data['weights']
    benches = data['benchmarks']
    n = min(num_samples, weights.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4))
    if n == 1:
        axes = [axes]
    for i in range(n):
        a = weights[i].mean(axis=0)
        im = axes[i].imshow(a, cmap='viridis', vmin=0, vmax=a.max())
        axes[i].set_title(str(benches[i]))
        axes[i].set_xticks([]); axes[i].set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    return _save(fig, output, 'suppl_attention_heatmaps')


def plot_hyperparam_sweep(sweep_dir: Path, output: Path) -> Optional[Path]:
    """48-config NIG hyperparameter sweep summary (supplementary).

    Reads ``<sweep_dir>/leaderboard.csv`` if present, otherwise falls back to
    the bundled ``data/nig_hyperparam_sweep.csv`` snapshot of the paper run.
    """
    leaderboard = sweep_dir / 'leaderboard.csv'
    if not leaderboard.is_file():
        leaderboard = BUNDLED_SWEEP_CSV
        log.info("leaderboard.csv not found in %s — using bundled %s",
                 sweep_dir, leaderboard)
    if not leaderboard.is_file():
        log.warning("No sweep CSV available — skipping")
        return None
    df = pd.read_csv(leaderboard)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sc = ax.scatter(df['lambda_mse'], df['lambda_ns'],
                    c=df['r2'], s=80, cmap='viridis', edgecolors='black')
    ax.set_xlabel(r'$\lambda_{\mathrm{MSE}}$'); ax.set_ylabel(r'$\lambda_{\mathrm{NS}}$')
    ax.set_title('NIG hyperparameter sweep — colour: $R^2$')
    ax.set_yscale('log'); ax.grid(True)
    fig.colorbar(sc, ax=ax, label='$R^2$'); fig.tight_layout()
    return _save(fig, output, 'suppl_hyperparam_sweep')


# ---------------------------------------------------------------------------
# Supplementary §"Scalability Analysis" — Fig. 8
# ---------------------------------------------------------------------------

def fig8_scalability_scatter(per_benchmark_csv: Path, output: Path) -> Path:
    """fig8 — MAPE vs cyclomatic complexity, coloured by tier.

    Prefers ``gat/results/per_benchmark.csv`` (emitted by
    ``gat/scripts/scalability_analysis.py``).  If that file is absent, falls
    back to running the analysis on the bundled smoke-test fixture
    (``data/sample_test_predictions.npz``) joined with the bundled
    ``data/cfg_metrics.csv``.
    """
    if not per_benchmark_csv.is_file():
        log.info("per-benchmark CSV not found at %s — computing on the fly "
                 "from bundled fixtures", per_benchmark_csv)
        sys.path.insert(0, str(REPO_ROOT / 'gat' / 'scripts'))
        from scalability_analysis import aggregate_per_benchmark   # noqa: E402
        df = aggregate_per_benchmark(BUNDLED_NIG_NPZ,
                                     BUNDLED_DATA / 'cfg_metrics.csv')
    else:
        df = pd.read_csv(per_benchmark_csv)
    fig, ax = plt.subplots(figsize=(7, 5))
    for tier, c in (('Low', STYLE['colors']['low']),
                    ('Med', STYLE['colors']['med']),
                    ('High', STYLE['colors']['high'])):
        sub = df[df['tier'].str.startswith(tier[:3], na=False)]
        ax.scatter(sub['cyclomatic'], sub['mape'] * 100,
                   s=90, color=c, edgecolors='black', linewidths=0.7,
                   label=f'{tier} complexity')
    rho = df['cyclomatic'].corr(df['mape'], method='spearman')
    ax.set_xlabel('cyclomatic complexity'); ax.set_ylabel('MAPE (%)')
    ax.set_title(f'Scalability: complexity vs.\\ error (Spearman $\\rho = {rho:.2f}$)')
    ax.set_xscale('log'); ax.grid(True); ax.legend()
    fig.tight_layout()
    return _save(fig, output, 'fig8_scalability_scatter')


# ---------------------------------------------------------------------------
# Section 6 — RL integration (Fig. 5a/b, Fig. 8 (RL thermal variant))
# ---------------------------------------------------------------------------

def _stack_seed_curves(seed_runs: List[List[dict]], key: str) -> np.ndarray:
    """Pad / stack per-seed time-series into (seeds, episodes)."""
    n = min(len(r) for r in seed_runs)
    return np.stack([[ep.get(key, np.nan) for ep in r[:n]] for r in seed_runs])


def plot_rl_curves(results_dir: Path, output: Path,
                   methods: Sequence[str] = ('SAMFRL', 'SAMBRL',
                                             'MAMFRL_D3QN', 'MAMBRL_D3QN'),
                   seeds: Sequence[int] = (42, 123, 456, 789, 1024)) -> Path:
    """Episode-return learning curves with 95% CI shading (Fig. 5a)."""
    runs = load_rl_results(results_dir, methods, seeds)
    if not runs:
        raise FileNotFoundError(f"No RL results under {results_dir}")
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, seed_runs in runs.items():
        curves = _stack_seed_curves(seed_runs, 'episode_return')
        x = np.arange(curves.shape[1])
        mean = np.nanmean(curves, axis=0)
        ci   = np.nanstd(curves,  axis=0) / np.sqrt(curves.shape[0]) * 1.96
        _shade_ci(ax, x, mean, ci, METHOD_COLOURS.get(m, '#000'), label=m)
    ax.set_xlabel('episode'); ax.set_ylabel('return')
    ax.set_title('RL learning curves (5 seeds, 95% CI)')
    ax.grid(True); ax.legend(); fig.tight_layout()
    return _save(fig, output, 'fig5a_rl_curves')


def plot_rl_thermal(results_dir: Path, output: Path,
                    threshold_c: float = 80.0,
                    methods: Sequence[str] = ('SAMFRL', 'SAMBRL',
                                              'MAMFRL_D3QN', 'MAMBRL_D3QN'),
                    seeds: Sequence[int] = (42, 123, 456, 789, 1024)) -> Path:
    """Per-episode peak temperature vs threshold (paper Fig. 8)."""
    runs = load_rl_results(results_dir, methods, seeds)
    if not runs:
        raise FileNotFoundError(f"No RL results under {results_dir}")
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, seed_runs in runs.items():
        peaks = _stack_seed_curves(seed_runs, 'peak_temperature_c')
        x = np.arange(peaks.shape[1])
        mean = np.nanmean(peaks, axis=0)
        ci   = np.nanstd(peaks,  axis=0) / np.sqrt(peaks.shape[0]) * 1.96
        _shade_ci(ax, x, mean, ci, METHOD_COLOURS.get(m, '#000'), label=m)
    ax.axhline(threshold_c, color=STYLE['colors']['accent'],
               linestyle='--', label=f'thermal limit ({threshold_c:.0f} °C)')
    ax.set_xlabel('episode'); ax.set_ylabel('peak temperature (°C)')
    ax.set_title('RL thermal tracking (5 seeds, 95% CI)')
    ax.grid(True); ax.legend(); fig.tight_layout()
    return _save(fig, output, 'rl_thermal_tracking')


def plot_rl_energy(results_dir: Path, output: Path,
                   methods: Sequence[str] = ('SAMFRL', 'SAMBRL',
                                             'MAMFRL_D3QN', 'MAMBRL_D3QN'),
                   seeds: Sequence[int] = (42, 123, 456, 789, 1024)) -> Path:
    """Final-episode energy vs makespan scatter (Fig. 5b)."""
    runs = load_rl_results(results_dir, methods, seeds)
    if not runs:
        raise FileNotFoundError(f"No RL results under {results_dir}")
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, seed_runs in runs.items():
        E = [r[-1].get('energy_j',   np.nan) for r in seed_runs]
        T = [r[-1].get('makespan_s', np.nan) for r in seed_runs]
        ax.scatter(T, E, s=80, label=m,
                   color=METHOD_COLOURS.get(m, '#000'),
                   edgecolors='black', linewidths=0.7)
    ax.set_xlabel('makespan (s)'); ax.set_ylabel('energy (J)')
    ax.set_title('Energy / makespan trade-off')
    ax.grid(True); ax.legend(); fig.tight_layout()
    return _save(fig, output, 'fig5b_rl_energy')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

#: Maps a CLI command to (input-arg-attribute-name, callable).  The attribute
#: name is the argparse field that holds the input directory / file.
DISPATCH = {
    # Section 4
    'gat-training':            ('gat_exp',          plot_gat_training),
    # Supplementary Fig. 5a–f
    'fig5a-training-loss':     ('nig_exp',          fig5a_training_loss),
    'fig5b-r2-score':          ('nig_exp',          fig5b_r2_score),
    'fig5c-predictions':       ('nig_exp',          fig5c_predictions_scatter),
    'fig5d-uncertainty-error': ('nig_exp',          fig5d_uncertainty_vs_error),
    'fig5e-calibration':       ('nig_exp',          fig5e_calibration_curve),
    'fig5f-ece-training':      ('nig_exp',          fig5f_ece_training),
    # Supplementary Fig. 6a/b
    'fig6a-picp':              ('nig_exp',          fig6a_picp_coverage),
    'fig6b-mpiw':              ('nig_exp',          fig6b_mpiw_confidence),
    # Supplementary Fig. 7a–c
    'fig7a-uncertainty-trend': ('nig_exp',          fig7a_uncertainty_error_trend),
    'fig7b-uncertainty-pie':   ('nig_exp',          fig7b_uncertainty_pie),
    'fig7c-binned-calibration':('nig_exp',          fig7c_binned_calibration),
    # Supplementary Fig. 8 (scalability scatter, distinct from RL Fig. 8)
    'fig8-scalability':        ('per_benchmark_csv', fig8_scalability_scatter),
    # Other supplementary
    'attention':               ('nig_exp',          plot_attention),
    'hyperparam-sweep':        ('sweep',            plot_hyperparam_sweep),
    # Section 6 — RL
    'rl-curves':               ('rl',               plot_rl_curves),
    'rl-thermal':              ('rl',               plot_rl_thermal),
    'rl-energy':               ('rl',               plot_rl_energy),
}

#: Every supplementary appendix figure with a fixed filename.
SUPPLEMENTARY_FIGS = [
    'fig5a-training-loss', 'fig5b-r2-score', 'fig5c-predictions',
    'fig5d-uncertainty-error', 'fig5e-calibration', 'fig5f-ece-training',
    'fig6a-picp', 'fig6b-mpiw',
    'fig7a-uncertainty-trend', 'fig7b-uncertainty-pie', 'fig7c-binned-calibration',
    'fig8-scalability',
]
NIG_FIGS = SUPPLEMENTARY_FIGS[:11]   # everything except fig8-scalability


def _resolve_arg(args, attr: str) -> Path:
    return getattr(args, attr.replace('-', '_'))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description='Regenerate every Python-driven figure in the paper '
                    'and supplementary appendix.')
    p.add_argument('command',
                   choices=list(DISPATCH.keys()) + ['nig-all', 'supplementary', 'all'],
                   help='which figure to regenerate (or a group)')
    p.add_argument('--gat-exp',          type=Path, default=DEFAULT_GAT_EXPERIMENT,
                   help=f'GAT experiment directory (default: {DEFAULT_GAT_EXPERIMENT})')
    p.add_argument('--nig-exp',          type=Path, default=DEFAULT_NIG_EXPERIMENT,
                   help=f'NIG experiment directory (default: {DEFAULT_NIG_EXPERIMENT})')
    p.add_argument('--sweep',            type=Path, default=REPO_ROOT / 'nig' / 'experiments',
                   help='sweep directory (must contain leaderboard.csv)')
    p.add_argument('--rl',               type=Path, default=DEFAULT_RL_RESULTS,
                   help=f'RL results directory (default: {DEFAULT_RL_RESULTS})')
    p.add_argument('--per-benchmark-csv', type=Path, default=DEFAULT_SCALABILITY,
                   help=f'CSV from gat/scripts/scalability_analysis.py '
                        f'(default: {DEFAULT_SCALABILITY})')
    p.add_argument('--output',           type=Path, default=DEFAULT_OUTPUT,
                   help=f'output directory (default: {DEFAULT_OUTPUT})')
    args = p.parse_args(argv)

    configure_style()

    if   args.command == 'nig-all':       commands = NIG_FIGS
    elif args.command == 'supplementary': commands = SUPPLEMENTARY_FIGS
    elif args.command == 'all':           commands = list(DISPATCH.keys())
    else:                                 commands = [args.command]

    failures = 0
    for cmd in commands:
        attr, fn = DISPATCH[cmd]
        try:
            log.info("[%s] reading %s", cmd, _resolve_arg(args, attr))
            fn(_resolve_arg(args, attr), args.output)
        except Exception as exc:                    # noqa: BLE001
            failures += 1
            log.error("[%s] failed: %s", cmd, exc)
    if failures:
        log.error("%d figure(s) failed", failures)
        return 1
    log.info("All requested figures written to %s", args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
