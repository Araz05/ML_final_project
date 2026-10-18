# ML_final_project
# Decision Tree and SVM from scratch on bina.az listings (MLE-AI-201 final project)

Two models written **from scratch in NumPy**, applied to the real, messy
[bina.az sale-listings dump](https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project)
and benchmarked against scikit-learn:

| | Task A: regression | Task B: classification |
|---|---|---|
| Target | `price` (AZN) | price tier: premium (1) vs standard (0), split at the **training-set median** |
| Metrics | RMSE, MAE, R² | precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix |
| Our models | `DecisionTree` (MSE) | `DecisionTree` (Gini / entropy), `PegasosSVM` (linear), `RBFPegasosSVM` (random Fourier features) |
| Baselines (comparison only) | `DecisionTreeRegressor` | `DecisionTreeClassifier`, `SGDClassifier(hinge)` and `SVC(rbf)` inside a `Pipeline` with `StandardScaler` |

scikit-learn is used **only** as the baseline and for sanity checks in `tests/`.
Every model we report as "ours" is plain NumPy.

## One-command reproduction

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt      # pinned versions (tested on Python 3.12)

# 1) get the data (see data/README.md): put the CSV at data/bina_az_sale.csv
# 2) reproduce every number and figure (about 6 minutes)
python -m src.run_all
```

This writes `report/results.txt` / `report/results.json` (all tables and
sweeps) and every figure to `report/figures/`. Everything is seeded
(`SEED = 42`), so two runs give identical numbers.

Other commands:

```bash
python -m src.run_all --quick        # ~20 s smoke run on a 6,000-row subsample
python -m src.data_prep              # inspect the raw dump and the cleaned splits
python -m pytest -q                  # all tests (sanity checks against sklearn)
```

## Getting the dataset

The raw CSV is **not** committed (see `.gitignore`). Download it from Kaggle
(`sehriyarmemmedli/binaaz-sale-project`) and save it as
`data/bina_az_sale.csv`. Details and gotchas are in `data/README.md`.
Respect the dataset licence and bina.az's terms; do not redistribute personal data.

## Repository layout

```
.
├── README.md
├── requirements.txt            # pinned dependencies
├── contribution_report.md      # who did what
├── data/README.md              # how to obtain the dataset (CSV is git-ignored)
├── src/
│   ├── data_prep.py            # cleaning, leakage removal, features, split, tier label, scaling
│   ├── decision_tree.py        # DecisionTree: Gini / entropy / MSE, 4 stopping rules, predict_proba
│   ├── svm.py                  # PegasosSVM (+ random Fourier features for the RBF kernel)
│   ├── evaluate.py             # metrics from scratch + comparison table
│   └── run_all.py              # ONE entry point for every number and figure
├── tests/                      # pytest suite (sanity checks vs scikit-learn)
├── report/                     # IEEE report (report.tex), figures/, results.txt/json
└── presentation/               # slides
```

## What the pipeline does (and why)

1. **Cleaning** (`data_prep.clean`, every step logs its row count):
   latest scrape per listing (the dump holds the same listing several times),
   leakage and identifier columns dropped (`unit_price`, `total_price`,
   owner/shop names, addresses, ids, urls, free text), currency normalised to
   AZN, `sot` land areas converted to m², `"7 / 9"` floors split into floor and
   total floors, fixed-bound outlier rules, impossible coordinates set to missing.
2. **Leakage-safe preprocessing:** rows are split first (stratified on price
   quartiles, 70/15/15). Category levels, imputation medians, scaler
   statistics and the tier threshold are all learned on the **training rows only**.
3. **Models:** the tree and SVM are tuned on the validation set only; the test
   set is touched once for the final numbers. Test metrics carry a bootstrap
   standard deviation, and we also report 5-fold cross-validated mean ± std.
4. **Error analysis:** tree impurity-decrease importances, SVM weights,
   errors by price quartile / property category / sparse districts.

## Headline results (held-out test set, ± bootstrap std)

| Task | Model | Main metrics |
|---|---|---|
| A | Ours: tree | RMSE 342,525 ± 19,432 · MAE 102,958 · R² 0.533 ± 0.031 |
| A | sklearn tree | RMSE 342,515 · MAE 102,938 · R² 0.533 |
| B | Ours: tree (Gini, depth 14) | F1 0.922 ± 0.003 · ROC-AUC 0.924 |
| B | sklearn tree | F1 0.919 · ROC-AUC 0.920 |
| B | Ours: Pegasos SVM (linear) | F1 0.850 ± 0.004 · ROC-AUC 0.931 |
| B | sklearn SGD hinge (Pipeline) | F1 0.850 · ROC-AUC 0.930 |
| B | Ours: Pegasos + RFF (RBF) | F1 0.849 · ROC-AUC 0.932 |
| B | sklearn SVC RBF (Pipeline, 8,000-row subsample) | F1 0.855 · ROC-AUC 0.936 |

The full tables (including cross-validation, timing and every sweep) are in
`report/results.txt` after running `python -m src.run_all`.

## Design notes

* **Tree:** exhaustive threshold search over sorted feature values; criteria
  Gini, entropy (information gain) and MSE reduction; `max_depth`,
  `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease` each change
  the tree (tested). Matches scikit-learn's splits on tiny, fully determined examples.
* **SVM:** Pegasos mini-batch sub-gradient descent on the regularised hinge
  loss; bias folded in as a constant feature; `lambda_`, `n_iters` / `n_epochs`
  and `lr_schedule` (`pegasos`, `inv_sqrt`, `constant`) are exposed; reports
  ‖w‖, the margin 2/‖w‖ and the support vectors; the primal objective is
  tracked and decreases. The RBF variant uses random Fourier features
  (chosen over kernelised Pegasos to avoid an n×n kernel matrix on ~45k rows).
* **Known limitations:** `SVC` is run on an 8,000-row subsample because it is
  too slow on the full training set; the tier threshold is the training median,
  so the tier is balanced by construction; Task A is evaluated on raw prices,
  which are heavy-tailed (a log-price target is reported as a variant).
