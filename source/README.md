# GraphPerf-RT — Reproducibility Source Release

This directory contains every script needed to reproduce the numerical claims
in the camera-ready submission:

> **GraphPerf-RT: Graph-Driven Performance Modeling with Calibrated Uncertainty for OpenMP Scheduling on Heterogeneous Embedded SoCs.**
> Mohammad Pivezhandi, Mahdi Banisharif, Saeed Bakhshan, Abusayeed Saifullah, Ali Jannesari.
> *Proceedings of the 35th International Joint Conference on Artificial Intelligence — 29th European Conference on Artificial Intelligence (IJCAI-ECAI 2026), AI4Tech Track.*

Public mirror (datasets, checkpoints, large logs):

> https://github.com/pivezhan/GraphPerf-RT_IJCAI2026

---

## 1. Layout (what lives where, and which paper section it serves)

```
source/
├── README.md                  ← you are here  (the only Markdown file in the release)
├── requirements.txt           pinned Python dependencies
│
├── data/                      bundled reproducibility data (small, version-controlled)
│   ├── cfg_metrics.csv               per-benchmark cyclomatic / loops / branches
│   ├── benchmark_inventory.csv       42 benchmarks with suite + parallelism
│   ├── dvfs_frequencies.json         per-platform DVFS tables (TX2 / Orin NX / RUBIK Pi)
│   ├── nig_hyperparam_sweep.csv      48-config sweep leaderboard
│   ├── sample_test_predictions.npz   smoke-test fixture (synthetic NIG output)
│   ├── sample_training_history.json  smoke-test fixture (loss / R² / ECE curves)
│   ├── sample_results.json           smoke-test fixture (one-shot summary metrics)
│   ├── generate_sample_predictions.py    re-emits the NPZ fixture
│   └── generate_sample_history.py        re-emits the JSON fixture
│
├── gat/                       §4 — GAT performance surrogate
│   ├── configs/               three YAML configs (GAT-log, MLP baseline, preprocessing)
│   ├── scripts/               numbered preprocessing pipeline + train + evaluate + inference
│   ├── src/                   library code (data / models / training / utils)
│   └── tests/                 split-validation utility
│
├── nig/                       §5 — Normal-Inverse-Gamma evidential head
│   ├── model.py               GAT backbone + NIG head (γ, ν, α, β)
│   ├── loss.py                evidential NLL + non-saturating regularizer
│   ├── train.py               single-config trainer
│   ├── hyperparam_configs.py  emits 48 configs for the sweep
│   ├── aggregate_sweep.py     reduces a sweep folder → leaderboard.csv
│   └── slurm_*.sh             SLURM templates (single / array / multi-GPU)
│
├── rl/                        §6 — RL integration
│   ├── thermal_violation_analysis.py   Table 4
│   ├── convergence_analysis.py         Fig. 5d learning-curve convergence test
│   ├── lobo_generalization.py          Table 5  (Leave-One-Benchmark-Out)
│   ├── world_model_ablation.py         Table 6  (GraphPerf-RT vs FCN world model)
│   ├── evaluate_baselines.py           aggregator → rl_summary.json
│   ├── offline_analysis.py             cross-platform transfer
│   └── on_device/                      Jetson TCP server / client
│       ├── client.py                   benchmark runner (Jetson side)
│       ├── server_combined.py          all four methods in one job (paper main result)
│       ├── server_{mambrl,mamfrl,sambrl,samfrl}.py   per-method drivers
│       ├── server_world_model.py       world-model ablation server
│       ├── server_lobo.py              LOBO sweep server
│       ├── tasks.py                    benchmark / state / action definitions
│       └── evaluate_on_device.py       TX2-specific aggregator
│
├── benchmarks/                workload runners (not bundled C source)
│   ├── omptasks/              BOTS task-parallel benchmarks (12 kernels)
│   └── polytasks/             PolyBench/C loop kernels (30 kernels)
│
└── figures/
    └── plot_figures.py        single CLI that regenerates every Python-driven
                                figure in the paper (Fig. 4 / 5 / 6 / 7 / 8
                                + supplementary attention + sweep)
```

**76 files, ~1.5 MB.**  No data, no checkpoints, no vendored toolchains —
those live in the public mirror linked above.

---

## 2. Reproducibility quick-start

### 2.1 What is and is not bundled

The `data/` directory ships **eight small files** (≈ 200 KB total) that are
required, or strongly useful, for reproducibility:

| File | Purpose | Source of truth |
| --- | --- | --- |
| `cfg_metrics.csv` | per-benchmark cyclomatic / loops / branches | extracted from the supplementary `tab:per-benchmark-scalability` (28 rows transcribed verbatim, then extended to all 42 benchmarks) |
| `benchmark_inventory.csv` | suite / kind / parallelism for every benchmark | curated for this release |
| `dvfs_frequencies.json` | per-platform DVFS frequency tables, thermal zones, governor defaults | measured on each Jetson / RUBIK Pi (see supplementary §"Hardware Platform Details") |
| `nig_hyperparam_sweep.csv` | 48-config leaderboard ranked by $R^2$ | from the production run of `nig/train.py` |
| `sample_test_predictions.npz`, `sample_training_history.json`, `sample_results.json` | **synthetic** smoke-test fixtures matching the schema of `nig/train.py` outputs | regenerated locally by `data/generate_sample_*.py` |
| `generate_sample_predictions.py` / `generate_sample_history.py` | re-emit the NPZ / JSON fixtures (deterministic seed) | this release |

**Not bundled** (live in the public mirror at https://github.com/pivezhan/GraphPerf-RT_IJCAI2026):

- The 47 000 raw profiling rows under `data/raw/` (split per platform).
- Normalised CSVs and `metadata.json` from preprocessing.
- Training checkpoints (`*.pth`) and per-experiment logs.
- All on-device RL trajectories (`save_model/`).
- Vendored LLVM / SWEET / OmpTG toolchains.

### 2.2 Smoke-testing the plot pipeline (no training required)

Verify that the plotting infrastructure works before kicking off a full
training run:

```bash
export GRAPHPERF_RT_ROOT="$(pwd)"
python data/generate_sample_predictions.py        # writes data/sample_test_predictions.npz
python data/generate_sample_history.py            # writes data/sample_training_history.json
python figures/plot_figures.py supplementary      # falls back to bundled fixtures
ls figures/output/                                # PNG + PDF for each of the 12 figures
```

The plot script auto-detects missing `nig/experiments/.../*.npz` and falls
back to the bundled fixtures, so the command produces real plots even on a
fresh checkout.

### 2.3 Environment

```bash
export GRAPHPERF_RT_ROOT="$(pwd)"           # the directory containing this README
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

All scripts honour `$GRAPHPERF_RT_ROOT` and never read absolute developer
paths — that was a deliberate sanitisation pass before release.

### 2.4 Mapping each main-paper claim to a runnable command

| Paper artefact | Numerical claim | Command |
| --- | --- | --- |
| Table 1, Fig. 4 | $R^2 = 0.81$, Spearman $\rho = 0.95$ on log-makespan | `python gat/scripts/train.py --config gat/configs/gat_log_config.yaml --experiment exp_v2_gat_log` |
| Table 3, Fig. 7 | PICP $= 99.9\%$ at 95% confidence | `python nig/train.py --lambda_mse 20 --lambda_ns 0.001 --lr 5e-4 --hidden_dim 128` |
| Fig. 5, Table 2 | RL learning curves (5 seeds × 200 episodes) | `python rl/on_device/server_combined.py --host 0.0.0.0 --port 7777 --seed 42` |
| Table 4, Fig. 8 | Zero thermal violations, peak-temperature stats | `python rl/thermal_violation_analysis.py --results-dir rl/on_device/save_model --output rl/results/thermal` |
| Table 5 | LOBO generalisation | `python rl/lobo_generalization.py --output rl/results/lobo` |
| Table 6 | World-model ablation (GraphPerf-RT vs FCN) | `python rl/world_model_ablation.py --output rl/results/wm` |
| All Python figures | Regenerate from saved artefacts | `python figures/plot_figures.py all` |

### 2.5 Mapping every **supplementary appendix** artefact to a runnable command

The supplementary references twelve PNGs by exact filename and ~12 result
tables.  Each row below names the section in the appendix, the artefact, and
the script that produces it.

#### Figures (filenames match `\includegraphics{}` calls in the appendix)

| Appendix § | Filename | Command |
| --- | --- | --- |
| NIG Results | `fig5a_training_loss.png` | `python figures/plot_figures.py fig5a-training-loss` |
| NIG Results | `fig5b_r2_score.png` | `python figures/plot_figures.py fig5b-r2-score` |
| NIG Results | `fig5c_predictions_scatter.png` | `python figures/plot_figures.py fig5c-predictions` |
| NIG Results | `fig5d_uncertainty_vs_error.png` | `python figures/plot_figures.py fig5d-uncertainty-error` |
| NIG Results | `fig5e_calibration_curve.png` | `python figures/plot_figures.py fig5e-calibration` |
| NIG Results | `fig5f_ece_training.png` | `python figures/plot_figures.py fig5f-ece-training` |
| PICP / MPIW | `fig6a_picp_coverage.png` | `python figures/plot_figures.py fig6a-picp` |
| PICP / MPIW | `fig6b_mpiw_confidence.png` | `python figures/plot_figures.py fig6b-mpiw` |
| Decomposition | `fig7a_uncertainty_error_trend.png` | `python figures/plot_figures.py fig7a-uncertainty-trend` |
| Decomposition | `fig7b_uncertainty_pie.png` | `python figures/plot_figures.py fig7b-uncertainty-pie` |
| Decomposition | `fig7c_binned_calibration.png` | `python figures/plot_figures.py fig7c-binned-calibration` |
| Scalability | `fig8_scalability_scatter.png` | `python figures/plot_figures.py fig8-scalability` |

Run all twelve at once: `python figures/plot_figures.py supplementary`.

#### Tables in the appendix

| Appendix § | Table | Script that produces it |
| --- | --- | --- |
| Per-Platform Dataset Statistics | `tab:per-platform-stats` | `gat/scripts/dataset_statistics.py` |
| Per-Platform Performance | `tab:per-platform-perf` | `gat/scripts/evaluate.py --per-platform` |
| All-Platforms Performance | `tab:all-platforms-perf` | `gat/scripts/evaluate.py` |
| Cross-Platform Transfer | `tab:cross-platform-transfer` | `rl/offline_analysis.py --task cross_platform` |
| Held-Out Benchmark | `tab:held-out-bench` | `rl/offline_analysis.py --task benchmark_holdout` |
| Wilcoxon Significance | `tab:wilcoxon` | `rl/offline_analysis.py --task statistics` |
| Extended RL Baselines | `tab:rl-baseline-extended` | `rl/evaluate_baselines.py` |
| Thermal Safety | `tab:thermal-safety` | `rl/thermal_violation_analysis.py` |
| LOBO Evaluation | `tab:lobo-results` | `rl/lobo_generalization.py` |
| Per-Benchmark Scalability | `tab:per-benchmark-scalability` | `gat/scripts/scalability_analysis.py` |
| Inference Latency | `tab:inference-latency` | `gat/scripts/inference_benchmark.py --mode platform` |
| Method Comparison (Evidential / Ensemble / MC-Dropout) | `tab:latency-comparison` | `gat/scripts/inference_benchmark.py --mode methods` |
| World-Model Ablation | `tab:world-model-ablation` | `rl/world_model_ablation.py` |
| NIG Best-Configuration Metrics | `tab:nig-best-metrics` | `nig/train.py` then `nig/aggregate_sweep.py` |
| Hyperparameter Sensitivity | `tab:nig-sensitivity` | `nig/aggregate_sweep.py --search_id <id>` |

### 2.6 GAT pipeline (data → train → evaluate)

```bash
cd gat
python scripts/01_process_data.py     --config configs/preprocess_config.yaml
python scripts/02_normalize_data.py   --config configs/preprocess_config.yaml
python scripts/03_split_data.py
python scripts/04_create_log_splits.py
python scripts/train.py    --config configs/gat_log_config.yaml --experiment exp_v2_gat_log
python scripts/evaluate.py --checkpoint experiments/exp_v2_gat_log/best.pth
```

The whole sequence is wrapped in `gat/scripts/run_preprocessing.sh`. A
CPU-friendly debug variant of the trainer lives in `gat/scripts/run_train_cpu.sh`.

### 2.7 NIG sweep (optional — only if you want the full hyperparameter table)

```bash
cd nig
python hyperparam_configs.py       # writes hp_configs/*.json
sbatch slurm_array.sh              # SLURM cluster, 48 array tasks
python aggregate_sweep.py          # → leaderboard.csv
```

If you don't have SLURM, the array script is a thin loop — adapt with GNU
`parallel` or a plain `for cfg in hp_configs/*.json; do python train.py --config $cfg; done`.

### 2.8 RL on-device training

Hardware: NVIDIA Jetson TX2 (primary), Jetson Orin NX, RUBIK Pi.

```bash
# host laptop (the GPU box):
export GRAPHPERF_RT_ROOT="$(pwd)"
python rl/on_device/server_combined.py --host 0.0.0.0 --port 7777 --seed 42

# Jetson:
python rl/on_device/client.py --host <HOST_IP> --port 7777
```

The `<HOST_IP>` placeholder is the IP of the host laptop reachable from the
Jetson's LAN; the public release contains **no hard-coded IP addresses**.
Repeat with seeds 42, 123, 456, 789, 1024 to reproduce the paper's CIs.

### 2.9 Offline RL analysis (no hardware required)

Once trajectories are saved under `rl/on_device/save_model/`, every analysis
script in `rl/` can run on a laptop:

```bash
python rl/thermal_violation_analysis.py --results-dir rl/on_device/save_model
python rl/convergence_analysis.py       --results-dir rl/on_device/save_model
python rl/lobo_generalization.py        --results-dir rl/on_device/save_model
python rl/world_model_ablation.py       --results-dir rl/on_device/save_model
python rl/evaluate_baselines.py         --results-dir rl/on_device/save_model
```

### 2.10 Figure regeneration

```bash
# Main paper
python figures/plot_figures.py gat-training         # Fig. 4
python figures/plot_figures.py rl-curves            # Fig. 5a
python figures/plot_figures.py rl-energy            # Fig. 5b
python figures/plot_figures.py rl-thermal           # Fig. 5d / Fig. 8

# Supplementary appendix (twelve fixed-name PNGs at once)
python figures/plot_figures.py supplementary

# Or one at a time:
python figures/plot_figures.py fig5a-training-loss
python figures/plot_figures.py fig5b-r2-score
python figures/plot_figures.py fig5c-predictions
python figures/plot_figures.py fig5d-uncertainty-error
python figures/plot_figures.py fig5e-calibration
python figures/plot_figures.py fig5f-ece-training
python figures/plot_figures.py fig6a-picp
python figures/plot_figures.py fig6b-mpiw
python figures/plot_figures.py fig7a-uncertainty-trend
python figures/plot_figures.py fig7b-uncertainty-pie
python figures/plot_figures.py fig7c-binned-calibration
python figures/plot_figures.py fig8-scalability    # needs gat/results/per_benchmark.csv

# Optional supplementary visualizations
python figures/plot_figures.py attention
python figures/plot_figures.py hyperparam-sweep

python figures/plot_figures.py all                  # every figure above
```

Plots use **Libertinus Serif** when available (matches the IJCAI typography).
Set `LIBERTINUS_REGULAR` and `LIBERTINUS_BOLD` to point at the font files;
otherwise matplotlib's default serif is used as a fallback.

---

## 3. Hardware

| Platform | Cores | Frequency range | Used for |
| --- | --- | --- | --- |
| NVIDIA Jetson TX2     | 4× A57 + 2× Denver2 | 0.34–2.04 GHz | RL on-device training (primary) |
| NVIDIA Jetson Orin NX | 8× A78AE            | 0.115–1.984 GHz | DVFS scheduling experiments |
| RUBIK Pi (RB3)        | 4× A78 + 4× A55     | 0.30–2.30 GHz | Cross-platform GAT validation |

Only on-device RL needs a Jetson. The GAT and NIG pipelines run end-to-end
on a CPU laptop (~4–6× slower than an A100).

---

## 4. Software versions tested

| Component | Version |
| --- | --- |
| Python | 3.10.12 |
| PyTorch | 2.1.0 (CPU + CUDA 11.8 / 12.1) |
| PyTorch Geometric | 2.4.0 |
| torch-scatter / torch-sparse | 2.1.2 / 0.6.18 |
| numpy / pandas / scikit-learn | 1.26.x / 2.1.x / 1.3.x |
| matplotlib / seaborn | 3.8.x / 0.13 |
| GCC / Clang for benchmarks | GCC 11.4 / Clang 14 |
| OpenMP runtime | libgomp 11.4 |

Pinned versions live in [requirements.txt](requirements.txt).

---

## 5. Sanitisation notes

Before release we removed:

- **Absolute developer paths**: `/home/<user>/...` and `/work/hdd/...` HPC
  paths replaced with `$GRAPHPERF_RT_ROOT` (shell scripts) or
  `os.environ.get('GRAPHPERF_RT_ROOT', ...)` with sane fallbacks (Python).
- **Hard-coded LAN IPs**: replaced with `<JETSON_IP>` / `<HOST_IP>`
  placeholders supplied at runtime through `--host` / `--port`.
- **Personal references** in comments and docstrings.
- **Author email addresses, SSH keys, credentials, API tokens** — none were
  present; we re-checked with `grep -rE`.

If you find a residual leak, please open an issue at the public mirror.

---

## 6. Citation

```bibtex
@inproceedings{pivezhandi2026graphperfrt,
  title     = {GraphPerf-RT: Graph-Driven Performance Modeling with Calibrated
               Uncertainty for OpenMP Scheduling on Heterogeneous Embedded SoCs},
  author    = {Pivezhandi, Mohammad and Banisharif, Mahdi and Bakhshan, Saeed and
               Saifullah, Abusayeed and Jannesari, Ali},
  booktitle = {Proceedings of the 35th International Joint Conference on
               Artificial Intelligence and the 29th European Conference on
               Artificial Intelligence (IJCAI-ECAI), AI4Tech Track},
  year      = {2026}
}
```
