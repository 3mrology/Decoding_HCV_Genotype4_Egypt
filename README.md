# HCV Genotype-4 Drug-Resistance Prediction: ESM-2 / ESMFold / AlphaFold2

Predicts HCV genotype-4 drug resistance from sequence and predicted 3-D
structure. Three feature families, namely sequence k-mers, ESM-2 protein
language-model embeddings, and structure from ESMFold and AlphaFold2
(ColabFold), are compared under one shared cohort, one shared split scheme, and
one shared classifier grid, so every number in the results table is directly
comparable.

This repository accompanies the article **"Decoding Drug Resistance in Egyptian
HCV Genotype 4: A Comparative Evaluation of Sequence- and Structure-Based
Protein Foundation Models"** (Babylonian Journal of Artificial Intelligence,
manuscript BJAI-2026-0104).

> **Dataset:** https://doi.org/10.5281/zenodo.24280872
> Download the curated cohort from the Zenodo record and place it as `hcv.csv`
> next to the notebooks, or supply it as a Kaggle dataset input.

---

## Contents

| File | Purpose |
|---|---|
| `hcv-final.ipynb` | Main pipeline: ESM-2 fine-tuning, ESMFold, ColabFold folding, feature extraction, classifier grid, XAI |
| `alphafold.ipynb` | Standalone AlphaFold2 (ColabFold) arm: fold, parse structural features, evaluate, explain |
| `supplementary/` | Released data and results that back the manuscript (see below) |
| `hcv.csv` | Input table, obtained from the Zenodo record above; not stored in this repository |

**Not stored in this repository.** The following are written into the working
directory when you run the notebooks, and are regenerated on every run:
`grid_results.csv`, `colabfold_results.csv`, `explanations.csv`,
`colabfold_explanations.csv`, `fig_*.png`, `esm2_ft/` (the fine-tuned ESM-2
checkpoint), `esmfold_grid_n*.parquet` and `esm2_grid_n*.parquet` (feature
caches), and `cf_out/` (the predicted structures). They are excluded because
they are derived artefacts, several of them large, and because the frozen
results behind the published tables and figures are released in
[`supplementary/`](supplementary/) instead. In particular,
`supplementary/result_grid_25_configurations.csv` is the grid that Table II was
built from, and it is the authoritative version.

---

## Supplementary data and results

The files released with the manuscript are in [`supplementary/`](supplementary/).
These are the frozen artefacts behind the published tables and figures, as
distinct from the runtime outputs above, which are regenerated on every run.

| File | Contents |
|---|---|
| `accessions.csv` | 1,890 records: accession, identity-cluster group, binary RAS label |
| `cohort_audit.json` | Cohort audit; cluster sizes on record and accession bases |
| `preprocessing_manifest.csv` | Stage-by-stage record and accession counts |
| `ram_coverage.csv` | Per-coordinate evaluable n and consensus departure rate, 60 positions |
| `classifier_hyperparameters.json` | Search grids and class weighting, all five classifiers |
| `result_grid_25_configurations.csv` | All 25 configurations, in Table II column order |
| `pr_curves.csv` | Accession-level precision-recall curves |
| `reproducibility_manifest.json` | Executed configuration, split, library versions, accession list |
| `requirements.txt` | Environment as executed (full `pip freeze`) |
| `checksums.txt` | SHA-256 of every file above |

Integrity check, run from inside `supplementary/`:

```bash
sha256sum -c checksums.txt
```

### Correspondence to the manuscript

| Manuscript item | File |
|---|---|
| Section 3.1, preprocessing stages | `preprocessing_manifest.csv` |
| Section 3.3, four-category counts | `cohort_audit.json` |
| Section 3.8, coverage and departure rates | `ram_coverage.csv` |
| Section 3.10, accession listing | `accessions.csv` |
| Section 3.10, SHA-256 hashes | `checksums.txt`, and the `sha256` block of `reproducibility_manifest.json` |
| Section 3.10, classifier hyperparameters | `classifier_hyperparameters.json` |
| Table II, complete 25-configuration grid | `result_grid_25_configurations.csv` |
| Fig. 10, precision-recall curves | `pr_curves.csv` |

### Notes on individual files

**`ram_coverage.csv`, coverage basis.** Coverage is `n_evaluable / n_records`
and is gene-relative: `n_records` is 285 for NS5A rows and 1,490 for NS5B rows,
meaning the records that cover the gene at all. The cohort arithmetic underlying
Section 3.8 is therefore

```
records that do NOT extend to NS5A 93      = 1890 - 285  + (285  - 178)  = 1712
records that do NOT extend to NS5B 282/316 = 1890 - 1490 + (1490 - 1490) = 400
```

Six RAM coordinates depart from consensus at 1.000, namely NS5A 28, 30, 31, 93
and NS5B 282, 316.

**`preprocessing_manifest.csv`, duplicate stage and `accessions_removed`.** The
duplicate stage removes one record, the second of the single exact duplicate
pair described in Section 3.2. The accession-purge row withholds 72 accessions,
after which 20,797 records remain; the label-blind filter stages below it remove
18,907 records in total, leaving 1,890 evaluable records.

`accessions_removed` is non-additive by construction. It is nonzero only on the
purge row (72) and the duplicate row (1). The purge withholds accessions from
the pretraining copy, while the evaluation cohort retains all 361. Filter stages
operate on records and do not remove accessions, hence 0 on those rows.

**`pr_curves.csv`.** Ten curves, one per feature source and classifier. Rows run
in descending recall, with the highest precision first inside any recall tie,
matching how `average_precision_score` integrates. `pr_auc` is the
accession-level average precision of that curve's own rows and reproduces from
them by the rectangle rule. Fig. 10 plots the XGBoost curve of each source; the
logistic-regression rows are retained for completeness but are not plotted, so
the two RAM rows are not a pair. `baseline` is the chance precision,
43 / 361 = 0.1191.

**`result_grid_25_configurations.csv`.** Eleven columns in Table II order:
`features`, `classifier`, `auroc`, `auroc_sd`, `pr_auc`, `f1`, `mcc`,
`sensitivity`, `specificity`, `balanced_accuracy`, `accuracy`. Balanced accuracy
is `(sensitivity + specificity) / 2`, rounded half-up to three decimals, and
sits between specificity and accuracy.

**Notebook sub-manifest.** `results/cf_sub_manifest.csv` was written by
`colabfold_batch` during the folding run and is named in
`reproducibility_manifest.json` under `notebook_manifest`. It is not part of
this release and is not referenced in the manuscript.

---

## Overview

The project studies how much predicted structure adds over sequence-only models
for HCV genotype-4 resistance calls. It runs two self-contained notebooks:

- **`hcv-final.ipynb`** builds the cohort, fine-tunes ESM-2 on the corpus, folds
  structures with ESMFold and ColabFold, extracts features, trains five
  classifiers on five feature sources, and produces the results tables and
  figures.
- **`alphafold.ipynb`** isolates the AlphaFold2 arm: fold, parse structural
  features, evaluate, and explain, so AlphaFold2 can be studied on its own
  without the rest of the pipeline interfering.

Both notebooks share `hcv_pipeline.py`, which contains the cohort builder,
feature extractors, classifier grid, metrics, and explainability helpers. The
first cell of each notebook writes that module to disk, so no manual module
setup is needed.

---

## Pipeline

```
hcv.csv
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
          specificity, accuracy; Wilson and bootstrap intervals)
```

---

## Quickstart

### Google Colab (recommended)

1. **Runtime → Change runtime type → GPU.** A T4 is enough for the default profile.
2. Upload `hcv-final.ipynb`, `alphafold.ipynb`, and `hcv.csv` into `/content/`.
3. **Runtime → Run all** in either notebook.
4. When the run finishes, `grid_results.csv` and every `fig_*.png` are in `/content/`.

---

## Configuration

All runtime knobs live in the **CONFIG** cell of each notebook. The right-hand
column gives the value used for the published run where it differs from the
repository default.

| Flag | Default | Meaning | Published run |
|---|---|---|---|
| `RUN_ESM2` | `True` | Compute frozen ESM-2 embeddings (1280-d) | `True` |
| `FINETUNE_ESM2` | `True` | Continued MLM pretraining on the corpus | `True` |
| `RUN_ESMFOLD` | `True` | Load ESMFold features from cache, or fold live if `ESMFOLD_LIVE` | `True` |
| `ESMFOLD_LIVE` | `False` | Fold live in the current kernel (3B model, needs 16 GB VRAM or more) | `False` |
| `RUN_COLABFOLD` | `True` | Fold with AlphaFold2 / ColabFold | `True` |
| `CF_MAX_SEQS` | `60` | Cap on sequences folded per run; `None` folds all | `None` (all 361 accessions) |
| `CF_MSA_MODE` | `"single_sequence"` | No MSA retrieved, so folding is alignment-free | `"single_sequence"` |
| `CF_NUM_RECYCLE` | `1` | AlphaFold2 recycling depth | `1` |
| `CF_TIMEOUT_MIN` | `45` | Hard wall-clock cap on ColabFold | `45` |
| `N_SPLITS` | `5` | Cross-validation folds | `5` |
| `CLUSTER_TH` | `0.90` | 6-mer Jaccard threshold for grouping similar sequences | `0.90` |
| `GRID_CLASSIFIERS` | `["logreg","randomforest","xgboost","lightgbm","mlp"]` | Classifier grid | all five |

Every heavy stage is gated: no GPU, no internet, or a missing cache simply skips
that feature source instead of crashing the notebook. Folding and fine-tuning
operate on unique accessions only; every row in the cohort inherits structural
features by `group_id`.

### Folding configuration as executed

AlphaFold2 structures were produced with ColabFold using `model_type
alphafold2_ptm`, `msa_mode single_sequence`, `num_recycle 1`, one model and one
seed. No external alignment was retrieved, so the ColabFold and ESMFold arms are
both alignment-free and differ in the folding network rather than in the use of
evolutionary information. Folding ran once per accession at roughly 11.4 s,
giving about 68.5 minutes for the 361 accessions on a single Tesla T4. The same
configuration is recorded in `supplementary/reproducibility_manifest.json` and
in Sections 3.7 and 3.10 of the manuscript.

To reproduce the published folding run, set `CF_MAX_SEQS = None`. The repository
default of `60` caps folding for quick inspection and does not fold the full
cohort.

---

## Evaluation protocol

- **Cohort.** One shared set of 1,890 records across 361 accessions, one label
  definition (binary, meaning any resistance-associated call), one split scheme.
- **Splitting.** `StratifiedGroupKFold` on `cluster_group`, where clusters come
  from greedy single-linkage grouping by 6-mer Jaccard at 0.90 or above. This
  merges the 361 accessions into 324 clusters, and no cluster spans a fold
  boundary.
- **Metrics.** Per-fold mean ± std plus pooled out-of-fold: accuracy with Wilson
  95% CI, AUROC with percentile bootstrap 95% CI over 2,000 draws, PR-AUC,
  macro-F1, MCC, balanced accuracy, sensitivity, specificity.
- **Classifiers.** Logistic regression, random forest, XGBoost, LightGBM, and an
  MLP, on identical inputs. Grids and class weighting are in
  `supplementary/classifier_hyperparameters.json`.
- **Labels.** Geno2pheno[HCV] version 0.92.

---

## Explainability

| Feature source | Method | What it tells you |
|---|---|---|
| Sequence 3-mers | SHAP (TreeExplainer) | Which **k-mer motifs** drive the call |
| ESM-2 embeddings | Per-residue occlusion | Which **residues**, when masked, most change the prediction |
| ESMFold / ColabFold | SHAP on **named** structural columns (pLDDT, Cα geometry, pLDDT at RAM coordinates) | Which **structural-confidence signals** matter |
| Logistic probe | Coefficients | Linear direction of influence |

Latent embedding dimensions are reported by influence only, and are not
interpreted as motifs. RAM here denotes the canonical resistance-associated
mutation coordinates NS5A 28, 30, 31, 93 and NS5B 282, 316.

---

## Results

The two notebooks emit, on every run:

- `grid_results.csv`, every feature source by classifier with mean ± std AUROC,
  PR-AUC, F1, MCC, balanced accuracy, sensitivity, specificity, accuracy, and
  95% CIs.
- `colabfold_results.csv`, the AlphaFold2-only table.
- `explanations.csv` and `colabfold_explanations.csv`, top features per model.
- `fig_grid_auroc.png`, `fig_grid_best_curves.png`, `fig_xai_top.png`,
  `fig_xai_shap_seqkmer3.png`, `fig_xai_structural.png`,
  `fig_xai_esm2_positions.png`, `fig_colabfold_*.png`.

The frozen grid behind the published Table II is
`supplementary/result_grid_25_configurations.csv`. A fresh run reproduces it up
to library and hardware variation; `supplementary/reproducibility_manifest.json`
records the exact environment used.

---

## Limitations

- **Silver-standard labels.** Resistance calls come from Geno2pheno[HCV], a
  computational interpreter, not from clinical treatment outcomes. Every result
  describes agreement with an in-silico tool, not validated clinical resistance.
  Prospective validation against SVR or treatment-failure data is required
  before any clinical use.
- **Class imbalance.** Rare-class metrics carry wide confidence intervals by
  construction, and the reported CIs reflect that.
- **Predicted structure.** ESMFold and AlphaFold2 predict structure from
  sequence alone, and in this study both do so without any alignment.
  Structural features reflect predicted confidence and geometry, not
  experimental structures.
- **Fine-tuning corpus.** Continued ESM-2 pretraining draws on the Los Alamos
  corpus sequences described in Section 3.2 of the manuscript, and should be
  understood as domain-adaptive pretraining. Its composition and filtering are
  documented in the manuscript and in `supplementary/preprocessing_manifest.csv`.

---

## Reproducibility

- All random seeds are fixed at 42 by default and threaded through the splitter,
  the classifiers, and the bootstrap CIs.
- Every heavy stage writes a cache tagged with the cohort size
  (`esmfold_grid_n{N}.parquet`, `esm2_grid_n{N}.parquet`), so a stale cache
  cannot silently misalign with a resized cohort.
- `supplementary/reproducibility_manifest.json` records the executed
  configuration, the split, the library versions, and the full accession list.
- `supplementary/checksums.txt` carries the SHA-256 of every released file.

---

## Environment

```
python       >= 3.10
torch        >= 2.1   (CUDA 11.8 or later for ESMFold / ColabFold)
transformers >= 4.38
scikit-learn, xgboost, lightgbm, shap, biopython
colabfold[alphafold]  (installed on demand by the notebook)
```

The environment as actually executed, pinned, is
`supplementary/requirements.txt`. The published run used Python 3.13.16, numpy
2.1.3, scikit-learn 1.6.1, torch 2.11.0+cu130, transformers 5.20.2, xgboost
3.4.1 and LightGBM 4.6.0, on CUDA 13.0 with a single NVIDIA Tesla T4.

---

## Citation

If you use this code or dataset, please cite the article and the dataset record.

```bibtex
@article{shaarawy2026hcv,
  title   = {Decoding Drug Resistance in Egyptian HCV Genotype 4: A Comparative
             Evaluation of Sequence- and Structure-Based Protein Foundation Models},
  author  = {Shaarawy, Amr and ElGharbawy, Yomna and Helmy, Ayman and
             Fouad, Mohamed Mostafa},
  journal = {Babylonian Journal of Artificial Intelligence},
  year    = {2026},
  note    = {Manuscript BJAI-2026-0104}
}

@dataset{shaarawy2026hcv_data,
  title     = {Curated Egyptian HCV Genotype 4 NS5A/NS5B Sequence Cohort with
               Geno2pheno[HCV] Resistance Annotations},
  author    = {Shaarawy, Amr and ElGharbawy, Yomna and Helmy, Ayman and
               Fouad, Mohamed Mostafa},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24280872},
  url       = {https://doi.org/10.5281/zenodo.24280872}
}
```

---

## Dataset

**https://doi.org/10.5281/zenodo.24280872**

Download the curated cohort from the Zenodo record and place it as `hcv.csv`
next to the notebooks, or supply it as a Kaggle dataset input.

---

## Ethics and data provenance

Sequences derive from publicly available records in NCBI GenBank and the Los
Alamos HCV Sequence Database. No direct human subjects or private clinical data
were involved, so institutional ethical approval was not required.
