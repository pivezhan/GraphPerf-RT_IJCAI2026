# GraphPerf-RT — IJCAI-ECAI 2026 Public Release

Camera-ready artefacts for:

> **GraphPerf-RT: Graph-Driven Performance Modeling with Calibrated Uncertainty for OpenMP Scheduling on Heterogeneous Embedded SoCs.**
> Mohammad Pivezhandi, Mahdi Banisharif, Saeed Bakhshan, Abusayeed Saifullah, Ali Jannesari.
> *Proceedings of the 35th International Joint Conference on Artificial Intelligence — 29th European Conference on Artificial Intelligence (IJCAI-ECAI 2026), AI4Tech Track.*

---

## Repository layout

```
GraphPerf-RT_IJCAI2026/
├── README.md                  ← you are here
├── ijcai_main.pdf             camera-ready paper (9 pages)
├── ijcai_supplementary.pdf    supplementary appendix (17 pages)
└── source/                    sanitized, end-to-end runnable code
    ├── README.md              full reproducibility guide
    ├── requirements.txt       pinned Python dependencies
    ├── data/                  bundled reproducibility data (cfg metrics,
    │                          DVFS tables, sweep leaderboard, fixtures)
    ├── gat/                   §4 — Graph Attention Network surrogate
    ├── nig/                   §5 — Normal-Inverse-Gamma evidential head
    ├── rl/                    §6 — RL integration (analysis + on-device)
    ├── benchmarks/            BOTS + PolyBench/C runners
    └── figures/               unified plot regenerator (`plot_figures.py`)
```

## Quick start

```bash
cd source
export GRAPHPERF_RT_ROOT="$(pwd)"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Smoke-test the plot pipeline (uses bundled fixtures, no training required)
python figures/plot_figures.py supplementary
```

This regenerates all twelve supplementary figures from the bundled smoke-test
fixtures — see [source/README.md](source/README.md) for the full
reproducibility procedure (data preparation, GAT training, NIG sweep,
on-device RL, figure / table generation).

## Citation

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

## Funding

The work was supported by the US Office of Naval Research through grant
**N00014-23-1-2151** and the US National Science Foundation through grants
**CNS-2601685** and **CNS-2602744**.

