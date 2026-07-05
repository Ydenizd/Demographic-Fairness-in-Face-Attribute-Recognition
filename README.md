# Demographic Fairness in Face-Attribute Recognition under Data Scarcity

Modern face models report very high **average** accuracy — but does that average hide systematically worse performance on demographic groups that are **rare in the training data**? This project measures that hidden gap rigorously, shows exactly where it lives, and tests three cheap ways to shrink it.

**Project 16 · Summer 2026 · Kaggle · InsightFace (ArcFace R100) · scikit-learn**

---

## Table of Contents
- [Overview](#overview)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Results Summary](#results-summary)
- [Setup and Running](#setup-and-running)
- [Notebook Structure](#notebook-structure)
- [Limitations](#limitations)
- [License and Attribution](#license-and-attribution)

---

## Overview

Face-attribute systems run in moderation, analytics, and access control. If accuracy silently drops for under-represented groups, the system encodes a measurable demographic bias even when its headline number looks excellent. This notebook quantifies that gap with defensible statistics, locates it (the age extremes), and establishes what **head-only** interventions can and cannot fix — without overclaiming.

The core idea: keep the heavy part (the backbone) **frozen** and learn only a tiny part. This makes the whole study fast, reproducible on a single Kaggle GPU, and isolates the question of *representation fairness*.

## Methodology

1. **Backbone (frozen):** Every face image is passed once through **ArcFace R100** (InsightFace's `buffalo_l` recognition model), yielding a **512-dimensional embedding** per face. The network is never fine-tuned — it is a fixed feature extractor. Because the embeddings are computed once, all downstream tasks share them for free.
2. **Heads (learned):** On top of the frozen embeddings, three small **MLP classifiers** (256 → 64, ReLU) are trained, one per attribute:
   - **Gender** — binary (Female / Male)
   - **Age** — 9 ordinal bins (0-2 … "more than 70")
   - **Race** — 7 classes (Black, East Asian, Indian, Latino_Hispanic, Middle Eastern, Southeast Asian, White)
3. **Fairness axis:** Every face is grouped into an intersectional **`race × age` cell** (7 × 9 = 63 cells), and each cell is labeled **under-** or **well-represented** by how often it appears in *training*. Fairness is defined as the **accuracy gap between the under- and well-represented cohorts**.
4. **Robust measurement:** The gap is reported two ways — (a) a **pooled, instance-level** gap on the natural distribution with a **bootstrap 95% confidence interval**, and (b) a **balanced** gap computed on a re-split where every subgroup contributes exactly 50 evaluation samples. Per-cell training frequency is also correlated with per-cell accuracy (**Spearman**) to test the scarcity hypothesis directly.
5. **Joint task (4th result):** For each face, we also ask whether **all three** attributes are correct at once ("3/3"). This compounds per-task errors and is the most operationally meaningful fairness view.
6. **Three mitigations**, each evaluated on the same balanced split so improvements are comparable:
   - **(V2) Reweighting** — oversample rare cells during training.
   - **(ii) Flip Test-Time Augmentation (TTA)** — average the original and horizontally-flipped embedding.
   - **(iii) Age-region group-expert heads** — a gate routes each face to a head specialized for its age region (young / middle / old), i.e. a Mixture-of-Experts.

### Why these choices
- **Frozen backbone, light heads:** The research question is about the *representation*, not maximum accuracy; freezing keeps the experiment cheap and the comparison clean.
- **Scarcity, not gender balance, as the axis:** FairFace is ~50/50 on gender, so a gender-only fairness axis carries almost no signal. The real imbalance is in the `race × age` cells.
- **Balanced + bootstrap:** The under-represented cohort is small and noisy; without these robustness steps a real gap can be buried under measurement noise (the notebook shows exactly this happening).

## Dataset

[**FairFace**](https://github.com/dchen236/FairFace) (~98k images), downloaded from the official Google Drive links via `gdown`. The notebook uses the `padding=0.25` version of the `dchen236/FairFace` release along with the corresponding `train`/`val` label CSVs. Downloading happens inside the notebook (CELL 06) automatically — no manual data prep is required.

## Results Summary

On frozen ArcFace embeddings, lightweight MLP heads reach high average accuracy on the easy task and modest accuracy on the hard ones — **gender 90.7%, age 47.5%, race 57.5%** overall. But the averages hide a **statistically significant scarcity bias**, and the bias is **unequal across tasks** and **compounds** when all three attributes must be correct at once.

**Finding 1 — The scarcity gap is real and significant** (balanced split, 50 samples per cell):

| Task | Under acc | Well acc | Gap (95% CI) |
|------|-----------|----------|--------------|
| Gender | 0.843 | 0.896 | **0.053** [0.028, 0.080] |
| Age | 0.380 | 0.443 | **0.063** [0.025, 0.100] |
| Race | 0.490 | 0.586 | **0.096** [0.060, 0.133] |
| Joint (3/3) | 0.136 | 0.238 | **0.101** [0.074, 0.130] |

None of the confidence intervals include zero. The Spearman correlation between a cell's training frequency and its accuracy is significant for all three tasks (gender ρ=+0.30; age ρ=+0.38; race ρ=+0.27) — direct evidence that *more training data → higher accuracy* at the subgroup level.

**Finding 2 — Fairness is a task × subgroup interaction.** The gap grows with task difficulty: gender (0.053) < age (0.063) < race (0.096) < joint (0.101).

**Finding 3 — The bias is located at the age extremes** (overwhelmingly elderly 50-69 and infant 0-2 faces).

**Finding 4 — Mitigations:** Age-region expert heads are the strongest intervention; on the joint task they lifted under-represented accuracy from 0.136 → 0.211 (**+55% relative**) and roughly halved the gap (0.101 → 0.042). Reweighting mostly *levels down* (improving "fairness" by degrading the strong cohort); Flip-TTA gives a genuine but partial gain.

> The numbers above are from this notebook's run; re-running reproduces them up to small sampling variation.

## Setup and Running

The notebook is designed to run on **Kaggle**.

**Kaggle settings:**
- Accelerator: `GPU T4 x2` or `P100`
- Internet: `On`
- Persistence: `Files`

**Dependencies** (installed by the notebook in CELL 03):

```bash
pip install insightface onnxruntime-gpu==1.19.2 opencv-python-headless gdown fairlearn
```

It also uses `torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `seaborn` (pre-installed in the Kaggle image).

**GPU note:** InsightFace executes ONNX models. If the CPU build of `onnxruntime` is present, it *silently shadows* the GPU build, and embedding extraction for ~98k images crawls on CPU (10-20× slower). That is why `onnxruntime-gpu==1.19.2` is pinned and `torch` is imported **before** `onnxruntime`. If the provider check fails: run the install cell, then **Run → Restart Kernel**, then continue.

Just run the cells in order from top to bottom.

## Notebook Structure

| Section | Content |
|---------|---------|
| Section 0 | Environment setup and GPU verification |
| Section 1 | Data, EDA, and the scarcity definition (`race × age` subgroups) |
| Section 2 | ArcFace embedding extraction (GPU, shared by all tasks) |
| Section 3 | Three baseline MLP heads (gender, age, race) |
| Section 4 | Scarcity-aware fairness toolkit + baseline evaluation |
| Section 5 | Joint correctness: all three attributes at once (Result #4) |
| Section 5b | Balanced subgroup evaluation (leakage-free re-split) |
| Section 6 | Mitigation V2: scarcity reweighting |
| Section 7 / 7b / 7c | Group thresholding, Flip-TTA, age-region expert heads (MoE) |
| Section 8 | Results comparison |
| Section 9 | Error analysis and limitations |
| Section 11 | Conclusion and detailed findings |

## Limitations

A high headline accuracy can coexist with a **significant, located, task-dependent** demographic bias driven by data scarcity at the age extremes. Cheap, **head-only** interventions can meaningfully reduce it, but they are **bounded** by the frozen backbone and by the gate's own accuracy (~84%). Mitigation here is *possible and worthwhile, not complete*: closing the remaining gap would likely require touching the representation itself (e.g. partially fine-tuning the backbone on reweighted data) or a stronger gate.

## License and Attribution

- Backbone: [InsightFace](https://github.com/deepinsight/insightface) — ArcFace R100 (`buffalo_l`).
- Dataset: [FairFace](https://github.com/dchen236/FairFace) (Kärkkäinen & Joo, 2021). Check the repository's license/terms of use before using.

Consider adding a license (e.g. MIT) to this repository.
