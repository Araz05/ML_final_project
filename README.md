# Decision Tree and SVM from scratch on bina.az listings

**MLE-AI-201 Machine Learning — Final Project (AI Academy, National AI Center)**

Two models written **from scratch in NumPy**, applied to the real, messy
[bina.az sale-listings dump](https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project)
and benchmarked against scikit-learn.

| | Task A: regression | Task B: classification |
|---|---|---|
| Target | `price` (AZN), plus a `log(price)` variant | price tier: premium (1) vs standard (0), split at the **training-set median** (210,000 AZN) |
| Metrics | RMSE, MAE, R² | accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix |
| Our models | `DecisionTree` (MSE) | `DecisionTree` (Gini / entropy), `PegasosSVM` (linear), `RBFPegasosSVM` (random Fourier features) |
| Baselines (comparison only) | `DecisionTreeRegressor` | `DecisionTreeClassifier`, `SGDClassifier(hinge)` and `SVC(rbf)`, each in a `Pipeline` with `StandardScaler` |

scikit-learn is used **only** for the baselines and for sanity checks in `tests/`.
Every model reported as "ours" is plain NumPy.

**Deliverables:** report → [`report/report.pdf`](report/report.pdf) (source `report/report.tex`) ·
slides → [`presentation/slides.pdf`](presentation/slides.pdf) ·
contributions → [`contribution_report.md`](contribution_report.md)

---

## One-command reproduction

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt      # pinned versions

# 1) get the data (see "Getting the dataset" below): data/bina_az_sale.csv
# 2) reproduce every number and figure (~7 minutes on a laptop)
python -m src.run_all
```

This writes `report/results.txt` and `report/results.json` (every table and
sweep) and all 16 figures to `report/figures/`. All randomness is seeded
(`SEED = 42`), so two runs give identical numbers.

Other commands:

```bash
python -m src.run_all --quick        # ~20 s smoke run on a 6,000-row subsample
python -m src.run_all --no-cv        # skip the 5-fold cross-validation
python -m src.data_prep              # inspect the raw dump and the cleaned splits
python -m pytest -q                  # all tests (incl. sanity checks against sklearn)

cd report && pdflatex report.tex && pdflatex report.tex   # rebuild the report PDF
```

## Getting the dataset

The raw CSV is **not** committed (see `.gitignore`). Download it from Kaggle
(`sehriyarmemmedli/binaaz-sale-project`) and save it as
`data/bina_az_sale.csv`. Details and gotchas are in `data/README.md`.
Respect the dataset licence and bina.az's terms; do not redistribute personal data.

## Headline results

Held-out test set (9,637 listings), mean ± bootstrap std over 200 resamples.

**Task A — price regression**

| Model | RMSE (AZN) | MAE (AZN) | R² |
|---|---|---|---|
| Ours: tree (MSE, depth 8, min_samples_leaf 50) | 342,525 ± 19,432 | 102,958 ± 2,944 | 0.533 ± 0.031 |
| sklearn: DecisionTreeRegressor (same settings) | 342,515 ± 19,432 | 102,938 ± 2,943 | 0.533 ± 0.031 |
| Ours: tree on log(price) | 365,036 ± 24,456 | 98,963 ± 3,283 | 0.470 ± 0.029 |

**Task B — price tier (premium vs standard)**

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Ours: tree (Gini, depth 14) | **0.922** | **0.922** | 0.924 | 0.882 |
| sklearn: DecisionTreeClassifier | 0.920 | 0.919 | 0.920 | 0.874 |
| Ours: Pegasos SVM (linear, λ = 1e-3) | 0.850 | 0.850 | 0.931 | 0.922 |
| sklearn: SGDClassifier hinge [Pipeline] | 0.850 | 0.850 | 0.930 | 0.920 |
| Ours: Pegasos + RFF (RBF, D = 300) | 0.849 | 0.849 | 0.932 | 0.927 |
| sklearn: SVC RBF [Pipeline, 8,000-row subsample] | 0.854 | 0.855 | **0.936** | **0.931** |

Bootstrap std is ≈ 0.003–0.004 for every Task B metric. 5-fold cross-validation
(train+val) gives the same ranking, e.g. Task A R² 0.588 ± 0.024 for both trees.

Key observations:

* **Our tree ≈ sklearn's tree.** Same depth and 119 leaves on Task A
  (prediction correlation 0.99999); on Task B 1,715 vs 1,712 leaves with
  99.1% identical test predictions. Remaining gaps come from tie-breaking
  between equally good splits.
* **Our SVM ≈ SGDClassifier** to the third decimal. RFF brings it within
  0.005 AUC of an exact RBF `SVC` while staying linear in the number of rows.
* **Tree vs SVM:** the tree has the best F1 (747 vs 1,444 test errors), the
  SVMs rank listings better (higher ROC-AUC).
* **Where models fail:** land plots (60.8% median abs. error) and commercial
  objects (46.9%) vs 14–15% for flats; luxury quartile (MAE 268k AZN);
  sparsely listed districts (49.6% vs 16.7%); listings within ±15% of the
  tier threshold (24.3% vs 4.0% tier error).

The full tables (precision/recall, CV, timing, every sweep) are in
`report/results.txt`.

## Repository layout

```
.
├── README.md
├── requirements.txt            # pinned dependencies
├── contribution_report.md      # who did what
├── PROJECT_STATEMENT.pdf       # the brief
├── data/README.md              # how to obtain the dataset (CSV is git-ignored)
├── src/
│   ├── data_prep.py            # cleaning, leakage removal, features, split, tier label, scaling
│   ├── decision_tree.py        # DecisionTree: Gini / entropy / MSE, 4 stopping rules, predict_proba
│   ├── svm.py                  # PegasosSVM (+ random Fourier features for the RBF kernel)
│   ├── evaluate.py             # metrics from scratch + comparison table
│   └── run_all.py              # ONE entry point for every number and figure
├── tests/                      # pytest suite (data pipeline, tree, SVM; sanity checks vs sklearn)
├── report/
│   ├── report.tex / report.pdf # IEEE two-column report
│   ├── results.txt / .json     # every number in the report
│   └── figures/                # every figure in the report and slides
└── presentation/slides.pdf     # demo slides
```

## What the pipeline does (and why)

1. **Cleaning** (`data_prep.clean`, every step logs its row count):
   Azerbaijani column names translated (`Sahə`→`area`, `Otaq sayı`→`rooms`, …);
   only the latest scrape per listing kept (the dump holds ~36k repeated scrapes);
   leakage and identifier columns dropped (`unit_price`, `total_price`,
   owner/shop names, addresses, ids, urls, free text); currency normalised to
   AZN; `sot` land areas converted to m²; `"7 / 9"` floors split into floor and
   total floors; fixed-bound outlier rules (price 5k–10M AZN, area 10–100,000 m²,
   rooms 1–15); impossible coordinates set to missing.
   Result: **64,247 listings, 91 features**.
2. **Leakage-safe preprocessing:** rows are split first (70/15/15 =
   44,973 / 9,637 / 9,637, stratified on price quartiles). Category levels,
   imputation medians, scaler statistics and the tier threshold are all learned
   on the **training rows only** (and re-learned inside every CV fold).
3. **Tuning:** tree depth / criterion / `min_samples_leaf`, SVM λ, RBF γ,
   training length and learning-rate schedule are all chosen on the
   validation split. The test split is used once, for the final numbers.
4. **Evaluation:** test metrics with bootstrap std, 5-fold CV mean ± std,
   fit/predict times.
5. **Error analysis:** tree impurity-decrease importances, SVM weights, errors
   by property category, price quartile, sparse districts and distance to the
   tier threshold.

## Implementation notes

* **Tree** (`src/decision_tree.py`): exhaustive threshold search over sorted
  feature values with cumulative sums (all candidate splits of a feature in one
  vectorised pass); midpoint thresholds like scikit-learn; criteria Gini,
  entropy (information gain) and MSE reduction; `max_depth`,
  `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease` each change
  the tree (tested); `predict_proba` from leaf frequencies; impurity-decrease
  `feature_importances_`. Matches scikit-learn's splits on tiny, fully
  determined examples (tested).
* **SVM** (`src/svm.py`): Pegasos mini-batch sub-gradient descent on the
  regularised hinge-loss primal; bias folded in as a constant feature;
  projection onto ‖w‖ ≤ 1/√λ and averaging of the second half of the
  iterates; `lambda_`, `n_iters` / `n_epochs` and `lr_schedule`
  (`pegasos`, `inv_sqrt`, `constant`) exposed; reports ‖w‖, the margin
  2/‖w‖ and the support vectors; the primal objective is tracked and
  decreases. The RBF variant uses random Fourier features, chosen over
  kernelised Pegasos to avoid kernel evaluations against all ~45k training rows.

## Known limitations

* `SVC` runs on an 8,000-row subsample because it is too slow on the full
  training set, so the RFF-vs-SVC comparison mixes kernel approximation and data size.
* The tier threshold is the training median, so the tier is balanced by
  construction and listings near the threshold are inherently hard.
* Task A is evaluated on raw prices, which are heavy-tailed; RMSE is dominated
  by a few luxury listings (the log-price variant is reported alongside).
* Free-text descriptions are not used; our tree is 5–18× slower to fit than
  scikit-learn's Cython implementation.

## AI-assistance disclosure

An AI assistant (Claude, Anthropic) helped draft the README
and the slides from our results files and figures. All numbers are produced by
the code in this repository.
