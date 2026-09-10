# HCV Genotype-4 Drug-Resistance Prediction — ESM-2 / ESMFold / AlphaFold2

Predicts HCV genotype-4 drug resistance from sequence and predicted 3-D
structure. Three feature families — sequence k-mers, ESM-2 protein language-model
embeddings, and structure from ESMFold and AlphaFold2 (ColabFold) — are compared
under one shared cohort, one shared split scheme, and one shared classifier grid,
so every number in the results table is directly comparable.

> **Dataset:** [🔗 ADD DATASET LINK HERE]
> (drop the CSV as `hcv_enriched.csv` next to the notebooks, or as a Kaggle dataset input.)

---

## Contents

| File | Purpose |
|---|---|
| `hcv-final.ipynb` | Main pipeline — ESM-2 fine-tuning, ESMFold, ColabFold folding, feature extraction, classifier grid, XAI |
| `alphafold-final (1).ipynb` | Standalone AlphaFold2 (ColabFold) arm — fold, parse structural features, evaluate, explain |
| `hcv_pipeline.py` | Shared module: cohort construction, feature extractors, evaluation harness, XAI |
| `hcv_enriched.csv` | Input table (see dataset link above) |
| `requirements.txt` | Python dependencies |

Outputs produced at runtime: `grid_results.csv`, `colabfold_results.csv`,
`explanations.csv`, `colabfold_explanations.csv`, `fig_*.png`,
`esm2_ft/`, `esmfold_grid_n*.parquet`, `esm2_grid_n*.parquet`, `cf_out/`.

---

## Overview

The project studies how much predicted structure adds over sequence-only models
for HCV genotype-4 resistance calls. It runs two self-contained notebooks:

- **`hcv-final.ipynb`** builds the cohort, fine-tunes ESM-2 on the corpus,
  folds structures with ESMFold and ColabFold, extracts features, trains five
  classifiers on five feature sources, and produces the results tables and
  figures.
- **`alphafold-final (1).ipynb`** isolates the AlphaFold2 arm: fold, parse
  structural features, evaluate, and explain, so AlphaFold2 can be studied on
  its own without the rest of the pipeline interfering.

Both notebooks share `hcv_pipeline.py`, which contains the cohort builder,
feature extractors, classifier grid, metrics, and explainability helpers.

---

## Pipeline

```
hcv_enriched.csv
      │
      ├─► build_common_cohort()  ─► one shared cohort (binary + 4-class)
      │
      ├─► add_identity_clusters()  ─► groups similar sequences together
      │
      ▼
┌──────────────────────┬───────────────────────┬────────────────────────┐
│ Sequence features    │ ESM-2 embeddings      │ Structure features     │
│  • base composition  │  • continued MLM      │  • ESMFold 3B          │
│  • 3-mer frequencies │    fine-tune (T33)    │  • ColabFold / AF2     │
│                      │  • 1280-d mean pool   │  • pLDDT, Cα geometry  │
└──────────┬───────────┴───────────┬───────────┴────────────┬───────────┘
           │                       │                        │
           └─────────────┬─────────┴────────────┬───────────┘
                         ▼                      ▼
              run_grid(...)            explain_grid(...)
              ─ logreg                ─ SHAP (trees)
              ─ random forest         ─ coefficients (linear)
              ─ xgboost               ─ permutation (MLP)
              ─ lightgbm              ─ per-residue occlusion (ESM-2)
              ─ mlp                   ─ named structural importance
                         │
                         ▼
          Per-fold mean ± std + out-of-fold metrics + 95% CIs
          (AUROC, PR-AUC, F1, MCC, balanced acc, sensitivity,
          specificity, accuracy — Wilson and bootstrap intervals)
```

---

## Quickstart

### Google Colab (recommended)

1. **Runtime → Change runtime type → GPU** (T4 is enough for the default profile).
2. Upload `hcv-final.ipynb`, `alphafold-final (1).ipynb`, and `hcv_enriched.csv`
   into `/content/`.
3. **Runtime → Run all** in either notebook. The first cell writes
   `hcv_pipeline.py` to disk, so no manual module setup is needed.
4. When the run finishes, `grid_results.csv` and every `fig_*.png` are in
   `/content/`.

### Local

```bash
git clone <this-repo>
cd <repo>
pip install -r requirements.txt
jupyter notebook hcv-final.ipynb
```

---

## Configuration

All runtime knobs live in the **CONFIG** cell of each notebook:

| Flag | Default | Meaning |
|---|---|---|
| `RUN_ESM2` | `True` | Compute frozen ESM-2 embeddings (1280-d) |
| `FINETUNE_ESM2` | `True` | Continued MLM pretraining on the corpus |
| `RUN_ESMFOLD` | `True` | Load ESMFold features from cache (or fold live if `ESMFOLD_LIVE`) |
| `ESMFOLD_LIVE` | `False` | Fold live in the current kernel (3B model — needs ≥ 16 GB VRAM) |
| `RUN_COLABFOLD` | `True` | Fold with AlphaFold2 / ColabFold |
| `CF_MAX_SEQS` | `60` | Cap the number of sequences folded per run (set to `None` for all) |
| `CF_MSA_MODE` | `"single_sequence"` | No MSA server → fast, no `PENDING`/`RUNNING` stalls |
| `CF_NUM_RECYCLE` | `1` | AlphaFold2 recycling depth |
| `CF_TIMEOUT_MIN` | `45` | Hard wall-clock cap on ColabFold |
| `N_SPLITS` | `5` | Cross-validation folds |
| `CLUSTER_TH` | `0.90` | 6-mer Jaccard threshold for grouping similar sequences |
| `GRID_CLASSIFIERS` | `["logreg","randomforest","xgboost","lightgbm","mlp"]` | Classifier grid |

Every heavy stage is gated: no GPU, no internet, or a missing cache simply
skips that feature source instead of crashing the notebook. Folding and
fine-tuning operate on unique accessions only; every row in the cohort inherits
structural features by `group_id`.

---

## Evaluation protocol

- **Cohort.** One shared set of sequences, one label definition
  (binary = any resistance-associated call), one split scheme.
- **Splitting.** `StratifiedGroupKFold` on `cluster_group`, where clusters come
  from greedy single-linkage grouping by 6-mer Jaccard ≥ 0.90.
- **Metrics.** Per-fold mean ± std plus pooled out-of-fold: accuracy with
  Wilson 95% CI, AUROC with percentile bootstrap 95% CI, PR-AUC, macro-F1,
  MCC, balanced accuracy, sensitivity, specificity.
- **Classifiers.** Logistic regression, random forest, XGBoost, LightGBM, and
  an MLP, on identical inputs.

---

## Explainability

| Feature source | Method | What it tells you |
|---|---|---|
| Sequence 3-mers | SHAP (TreeExplainer) | Which **k-mer motifs** drive the call |
| ESM-2 embeddings | Per-residue occlusion | Which **residues**, when masked, most change the prediction |
| ESMFold / ColabFold | SHAP on **named** structural columns (pLDDT, Cα geometry, pLDDT-at-RAS-position) | Which **structural-confidence signals** matter |
| Logistic probe | Coefficients | Linear direction of influence |

Latent embedding dimensions are reported by influence only — they are not
interpreted as motifs.

---

## Results (regenerated on every run)

The two notebooks emit:

- `grid_results.csv` — every feature source × classifier with mean ± std AUROC, PR-AUC, F1, MCC, balanced accuracy, sensitivity, specificity, accuracy, and 95% CIs.
- `colabfold_results.csv` — the AlphaFold2-only table.
- `explanations.csv`, `colabfold_explanations.csv` — top features per model.
- `fig_grid_auroc.png`, `fig_grid_best_curves.png`, `fig_xai_top.png`,
  `fig_xai_shap_seqkmer3.png`, `fig_xai_structural.png`,
  `fig_xai_esm2_positions.png`, `fig_colabfold_*.png`.

---

## Limitations

- **Silver-standard labels.** Resistance calls come from Geno2pheno[HCV], a
  computational interpreter — not clinical treatment outcomes. Every result
  describes agreement with an in-silico tool, not validated clinical resistance.
  Prospective validation against SVR / treatment-failure data is required
  before any clinical use.
- **Class imbalance.** Rare-class metrics carry wide confidence intervals by
  construction; the reported CIs reflect that.
- **Structure model.** ESMFold and AlphaFold2 predict structure from sequence
  alone. Structural features reflect predicted confidence and geometry, not
  experimental structures.
- **Fine-tuning corpus.** Continued ESM-2 pretraining uses the corpus sequences;
  swap in a broader reference FASTA (e.g. LANL) for strict reproduction of
  published numbers.

---

## Reproducibility

- All random seeds are fixed (`seed=42` by default) and threaded through the
  splitter, classifiers, and bootstrap CIs.
- Every heavy stage writes a cache tagged with the cohort size
  (`esmfold_grid_n{N}.parquet`, `esm2_grid_n{N}.parquet`) so a stale cache
  cannot silently misalign with a resized cohort.

---

## Environment

```
python     >= 3.10
torch      >= 2.1   (CUDA ≥ 11.8 for ESMFold / ColabFold)
transformers >= 4.38
scikit-learn, xgboost, lightgbm, shap, biopython
colabfold[alphafold]  (installed on demand by the notebook)
```

See `requirements.txt` for pinned versions.

---

## Citation

```bibtex
@misc{hcv_g4_resistance,
  title  = {HCV Genotype-4 Drug-Resistance Prediction with ESM-2, ESMFold, and AlphaFold2},
  author = {[AUTHORS]},
  year   = {2026},
  url    = {https://github.com/[USER]/[REPO]}
}
```

---

## Dataset

**https://drive.google.com/file/d/1IqcqPuJhdNL74kbO4Z0XYzg3anYYujLr/view?usp=sharing**

Place the file as `hcv.csv` next to the notebooks (Colab, local) or
as a Kaggle dataset input (the loader globs `/kaggle/input/**/hcv_enriched.csv`).

---
