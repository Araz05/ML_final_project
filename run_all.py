"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all              # full run (a few minutes)
    python -m src.run_all --quick      # small smoke run on a subsample

What it does
------------
  1. load + clean the data, build features, make the splits, impute and
     standardize with TRAIN statistics only (SVM needs standardized features),
  2. EDA figures,
  3. Task A (regression, price):   our DecisionTree  vs  sklearn baseline,
  4. Task B (classification, tier): our DecisionTree, our Pegasos SVM (linear
     and RBF via random Fourier features)  vs  sklearn baselines,
  5. hyper-parameter sweeps (depth, criterion, min_samples_leaf, lambda, gamma)
     chosen on the VALIDATION set only,
  6. final test-set numbers + k-fold cross-validated mean +- std,
  7. error analysis (feature importances, SVM weights, errors by segment),
  8. every figure is saved to report/figures/, every number to
     report/results.json and report/results.txt.

Everything is deterministic (fixed seeds, see SEED).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                       # no display needed
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402

from . import data_prep, evaluate as ev     # noqa: E402
from .decision_tree import DecisionTree     # noqa: E402
from .svm import PegasosSVM, RBFPegasosSVM  # noqa: E402

SEED = data_prep.SEED
REPORT_DIR = Path("report")
FIG_DIR = REPORT_DIR / "figures"

# --- experiment grids (full run / quick run) --------------------------------
FULL = dict(
    depths=[2, 4, 6, 8, 10, 12, 14, 18],
    leafs=[1, 5, 20, 50],
    lambdas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    rbf_lambdas=[1e-5, 1e-4, 1e-3],
    rbf_gamma_mult=[0.25, 1.0],
    epochs=[0.1, 0.5, 1, 3, 10, 30],
    svm_iters=20_000,
    n_components=300,
    svc_n=8_000,
    folds=5,
    max_rows=None,
)
QUICK = dict(
    depths=[2, 4, 6],
    leafs=[1, 20],
    lambdas=[1e-4, 1e-2],
    rbf_lambdas=[1e-4],
    rbf_gamma_mult=[1.0],
    epochs=[0.5, 3],
    svm_iters=3_000,
    n_components=100,
    svc_n=1_500,
    folds=3,
    max_rows=6_000,
)


# =============================================================================
# small utilities
# =============================================================================
def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  [figure] report/figures/{name}")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def reg_metrics(y, pred) -> dict:
    return {"RMSE": ev.rmse(y, pred), "MAE": ev.mae(y, pred), "R2": ev.r2(y, pred)}


def clf_metrics(y, pred, score) -> dict:
    p, r, f = ev.precision_recall_f1(y, pred)
    return {"Accuracy": ev.accuracy(y, pred), "Precision": p, "Recall": r, "F1": f,
            "ROC-AUC": ev.roc_auc(y, score), "PR-AUC": ev.average_precision(y, score)}


def with_uncertainty(point: dict, resampled_fn, n: int, n_boot: int = 200) -> dict:
    """Attach a bootstrap standard deviation to every metric of one model on the
    held-out TEST set: resample the test rows with replacement `n_boot` times,
    recompute the metrics, and return {metric: (value, std)}."""
    rng = np.random.default_rng(SEED)
    draws = [resampled_fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    return {k: (point[k], float(np.std([d[k] for d in draws]))) for k in point}


def _tree_score(model, X):
    """P(premium) from a tree (ours or sklearn's)."""
    return model.predict_proba(X)[:, 1]


# =============================================================================
# 1. data
# =============================================================================
def prepare_data(cfg) -> dict:
    _section("1. DATA  (load -> clean -> split -> features -> impute -> standardize)")
    df = data_prep.clean(data_prep.load_raw())
    if cfg["max_rows"] and len(df) > cfg["max_rows"]:
        df = df.sample(n=cfg["max_rows"], random_state=SEED).reset_index(drop=True)
        print(f"  [quick] subsampled to {len(df):,} rows")

    # rows are split FIRST; category levels are learned on the train rows only
    S = data_prep.split_and_featurize(df)
    names, ytr, yva, yte = S["names"], S["ytr"], S["yva"], S["yte"]
    # keep the un-imputed train+val block for cross-validation later
    Xtv_raw, ytv = np.vstack([S["Xtr"], S["Xva"]]), np.concatenate([ytr, yva])
    Xtr, Xva, Xte = data_prep.impute_median(S["Xtr"], S["Xva"], S["Xte"])

    tier_tr, thr = data_prep.make_tier_label(ytr)          # threshold from TRAIN only
    tier_va, _ = data_prep.make_tier_label(yva, thr)
    tier_te, _ = data_prep.make_tier_label(yte, thr)
    Ztr, Zva, Zte = data_prep.standardize(Xtr, Xva, Xte)   # for the SVM

    print(f"  rows after cleaning : {len(df):,}")
    print(f"  features            : {len(names)}")
    print(f"  split train/val/test: {len(ytr):,} / {len(yva):,} / {len(yte):,}")
    print(f"  tier threshold (TRAIN median price): {thr:,.0f} AZN")
    for nm, t in (("train", tier_tr), ("val", tier_va), ("test", tier_te)):
        print(f"  premium share in {nm:<5}: {100 * t.mean():.1f}%")

    return dict(df=df, names=names, thr=thr, Xtv_raw=Xtv_raw, ytv=ytv,
                Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte,
                Ttr=tier_tr, Tva=tier_va, Tte=tier_te,
                Ztr=Ztr, Zva=Zva, Zte=Zte)


# =============================================================================
# 2. EDA
# =============================================================================
def eda_figures(D) -> None:
    _section("2. EDA FIGURES")
    df, rng = D["df"], np.random.default_rng(SEED)
    price = df["price"].to_numpy()

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].hist(price / 1e3, bins=80, color="#4c72b0")
    ax[0].set(title="Price (right-skewed)", xlabel="price, thousand AZN", ylabel="listings")
    ax[1].hist(np.log10(price), bins=80, color="#55a868")
    ax[1].set(title="log10(price) (near-symmetric)", xlabel="log10(price in AZN)")
    _save(fig, "eda_price_distribution.png")

    idx = rng.choice(len(df), size=min(8000, len(df)), replace=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.scatter(df["area"].to_numpy()[idx], price[idx], s=3, alpha=.3)
    ax.set(xscale="log", yscale="log", xlabel="area, m² (log)", ylabel="price, AZN (log)",
           title="Area vs price")
    _save(fig, "eda_area_vs_price.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ok = (df["lat"].notna() & df["lng"].notna()).to_numpy()
    sc = ax.scatter(df["lng"].to_numpy()[ok], df["lat"].to_numpy()[ok],
                    c=np.log10(price[ok]), s=2, cmap="viridis", alpha=.5)
    fig.colorbar(sc, label="log10(price)")
    ax.set(xlabel="longitude", ylabel="latitude", title="Location effect")
    _save(fig, "eda_location.png")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    g = df.groupby("category")["price"].agg(["size", "median"]).sort_values("size", ascending=False)
    ax[0].bar(g.index, g["median"] / 1e3, color="#c44e52")
    ax[0].set(title="Median price by category", ylabel="thousand AZN")
    ax[0].tick_params(axis="x", rotation=30)
    rooms = df["rooms"].dropna().astype(int).value_counts().sort_index()
    ax[1].bar(rooms.index, rooms.values, color="#8172b2")
    ax[1].set(title="Rooms", xlabel="rooms", ylabel="listings")
    _save(fig, "eda_category_rooms.png")

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    shares = [100 * D[k].mean() for k in ("Ttr", "Tva", "Tte")]
    ax.bar(["train", "val", "test"], shares, color="#4c72b0")
    ax.axhline(50, color="k", ls="--", lw=.8)
    ax.set(ylabel="premium share, %", ylim=(0, 100), title="Price-tier balance")
    _save(fig, "eda_tier_balance.png")


# =============================================================================
# 3. Task A — regression
# =============================================================================
def task_a(D, cfg) -> dict:
    _section("3. TASK A — REGRESSION (price in AZN)")
    from sklearn.tree import DecisionTreeRegressor
    Xtr, ytr, Xva, yva, Xte, yte = (D[k] for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte"))

    # --- depth sweep (bias-variance) ---------------------------------------
    sweep = []
    for d in cfg["depths"]:
        m, t = _timed(lambda: DecisionTree(task="regression", criterion="mse",
                                           max_depth=d).fit(Xtr, ytr))
        sk = DecisionTreeRegressor(max_depth=d, random_state=SEED).fit(Xtr, ytr)
        row = dict(depth=d, train_R2=ev.r2(ytr, m.predict(Xtr)),
                   val_R2=ev.r2(yva, m.predict(Xva)),
                   val_RMSE=ev.rmse(yva, m.predict(Xva)),
                   sk_val_R2=ev.r2(yva, sk.predict(Xva)), leaves=m.get_n_leaves(), fit_s=t)
        sweep.append(row)
        print(f"  depth={d:>2}  train R2={row['train_R2']:.3f}  val R2={row['val_R2']:.3f}  "
              f"(sklearn {row['sk_val_R2']:.3f})  leaves={row['leaves']:>5}  {t:.1f}s")
    best_d = min(sweep, key=lambda r: r["val_RMSE"])["depth"]

    # --- min_samples_leaf sweep at the best depth -----------------------------
    leaf_sweep = []
    for l in cfg["leafs"]:
        m = DecisionTree(task="regression", criterion="mse", max_depth=best_d,
                         min_samples_leaf=l).fit(Xtr, ytr)
        leaf_sweep.append(dict(min_samples_leaf=l, val_RMSE=ev.rmse(yva, m.predict(Xva)),
                               val_R2=ev.r2(yva, m.predict(Xva))))
        print(f"  depth={best_d} min_samples_leaf={l:>3}  val R2={leaf_sweep[-1]['val_R2']:.3f}")
    best_l = min(leaf_sweep, key=lambda r: r["val_RMSE"])["min_samples_leaf"]
    print(f"  -> chosen on VALIDATION: max_depth={best_d}, min_samples_leaf={best_l}")

    # --- final models (test) -------------------------------------------------
    own, t_own = _timed(lambda: DecisionTree(task="regression", criterion="mse",
                                             max_depth=best_d, min_samples_leaf=best_l).fit(Xtr, ytr))
    sk, t_sk = _timed(lambda: DecisionTreeRegressor(
        max_depth=best_d, min_samples_leaf=best_l, random_state=SEED).fit(Xtr, ytr))
    # extra: tree on log(price), predictions mapped back to AZN
    own_log = DecisionTree(task="regression", criterion="mse", max_depth=best_d,
                           min_samples_leaf=best_l).fit(Xtr, np.log(ytr))

    pred, pred_t = {}, {}
    for name, f in (("Ours: tree", lambda: own.predict(Xte)),
                    ("sklearn: tree", lambda: sk.predict(Xte)),
                    ("Ours: tree (log-price target)", lambda: np.exp(own_log.predict(Xte)))):
        pred[name], pred_t[name] = _timed(f)
    results = {}
    for name, p_ in pred.items():
        results[name] = with_uncertainty(
            reg_metrics(yte, p_), lambda idx, p_=p_: reg_metrics(yte[idx], p_[idx]), len(yte))
        results[name]["pred_s"] = pred_t[name]
    results["Ours: tree"]["fit_s"], results["sklearn: tree"]["fit_s"] = t_own, t_sk
    print("\n  TEST set (metric ± bootstrap std over 200 resamples of the test set):")
    ev.compare(results)

    # structure / prediction parity with sklearn (Problem 5a)
    po, ps = pred["Ours: tree"], pred["sklearn: tree"]
    parity = dict(ours_depth=own.get_depth(), sk_depth=sk.get_depth(),
                  ours_leaves=own.get_n_leaves(), sk_leaves=sk.get_n_leaves(),
                  prediction_correlation=float(np.corrcoef(po, ps)[0, 1]),
                  identical_prediction_share=float(np.mean(np.isclose(po, ps))))
    print(f"\n  Parity with sklearn: depth {parity['ours_depth']} vs {parity['sk_depth']}, "
          f"leaves {parity['ours_leaves']} vs {parity['sk_leaves']}, "
          f"prediction corr {parity['prediction_correlation']:.4f}, "
          f"identical predictions {100 * parity['identical_prediction_share']:.1f}%")

    # --- figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ds = [r["depth"] for r in sweep]
    ax.plot(ds, [r["train_R2"] for r in sweep], "o-", label="train (ours)")
    ax.plot(ds, [r["val_R2"] for r in sweep], "s-", label="validation (ours)")
    ax.plot(ds, [r["sk_val_R2"] for r in sweep], "^--", label="validation (sklearn)")
    ax.set(xlabel="max_depth", ylabel="R²", title="Task A: bias–variance (decision tree)")
    ax.legend()
    _save(fig, "task_a_depth_curve.png")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    p = pred["Ours: tree"]
    ax[0].scatter(yte, np.maximum(p, 1.0), s=3, alpha=.3)
    lim = [yte.min(), yte.max()]
    ax[0].plot(lim, lim, "r--", lw=1)
    ax[0].set(xscale="log", yscale="log", xlabel="actual price", ylabel="predicted price",
              title="Test: predicted vs actual")
    q = data_prep.price_strata(yte)
    mdape = [np.median(np.abs(p[q == g] - yte[q == g]) / yte[q == g]) * 100 for g in range(4)]
    ax[1].bar(["Q1 (cheap)", "Q2", "Q3", "Q4 (luxury)"], mdape, color="#c44e52")
    ax[1].set(ylabel="median abs. % error", title="Error by price quartile")
    _save(fig, "task_a_errors.png")

    return dict(sweep=sweep, leaf_sweep=leaf_sweep, best=dict(max_depth=best_d, min_samples_leaf=best_l),
                test=results, own_model=own, test_pred=p, parity=parity)


# =============================================================================
# 4. Task B — classification
# =============================================================================
def _svm_row(m, Z, T, extra=None):
    pred, score = m.predict(Z), m.decision_function(Z)
    row = clf_metrics(T, pred, score)
    row.update(extra or {})
    return row


def task_b(D, cfg) -> dict:
    _section("4. TASK B — CLASSIFICATION (price tier: premium = 1)")
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    Xtr, Xva, Xte = D["Xtr"], D["Xva"], D["Xte"]
    Ztr, Zva, Zte = D["Ztr"], D["Zva"], D["Zte"]
    Ttr, Tva, Tte = D["Ttr"], D["Tva"], D["Tte"]
    d_feat = Xtr.shape[1]

    # ---------------- decision tree: criterion x depth ---------------------
    print("\n  4a. Decision tree: Gini vs entropy x max_depth")
    tree_sweep = []
    for crit in ("gini", "entropy"):
        for d in cfg["depths"]:
            m = DecisionTree(task="classification", criterion=crit, max_depth=d).fit(Xtr, Ttr)
            ftr = ev.precision_recall_f1(Ttr, m.predict(Xtr))[2]
            fva = ev.precision_recall_f1(Tva, m.predict(Xva))[2]
            auc = ev.roc_auc(Tva, _tree_score(m, Xva))
            tree_sweep.append(dict(criterion=crit, depth=d, train_F1=ftr, val_F1=fva, val_AUC=auc))
            print(f"  {crit:<8} depth={d:>2}  train F1={ftr:.3f}  val F1={fva:.3f}  val AUC={auc:.3f}")
    best_t = max(tree_sweep, key=lambda r: (round(r["val_F1"], 4), -r["depth"]))
    leaf_sweep = []
    for l in cfg["leafs"]:
        m = DecisionTree(task="classification", criterion=best_t["criterion"],
                         max_depth=best_t["depth"], min_samples_leaf=l).fit(Xtr, Ttr)
        leaf_sweep.append(dict(min_samples_leaf=l,
                               val_F1=ev.precision_recall_f1(Tva, m.predict(Xva))[2]))
        print(f"  {best_t['criterion']} depth={best_t['depth']} min_samples_leaf={l:>3}  "
              f"val F1={leaf_sweep[-1]['val_F1']:.3f}")
    best_leaf = max(leaf_sweep, key=lambda r: (round(r["val_F1"], 4), -r["min_samples_leaf"]))["min_samples_leaf"]
    tree_cfg = dict(criterion=best_t["criterion"], max_depth=best_t["depth"],
                    min_samples_leaf=best_leaf)
    print(f"  -> chosen on VALIDATION: {tree_cfg}")

    # ---------------- linear SVM: lambda sweep ----------------------------
    print("\n  4b. Pegasos SVM: lambda sweep (margin vs error trade-off)")
    svm_sweep = []
    for lam in cfg["lambdas"]:
        m = PegasosSVM(lambda_=lam, n_iters=cfg["svm_iters"], random_state=SEED).fit(Ztr, Ttr)
        r = _svm_row(m, Zva, Tva)
        r.update(lam=lam, margin=m.margin_width(), w_norm=m.weight_norm(),
                 n_support=m.n_support_, objective=m.objective_history_[-1][1],
                 history=m.objective_history_)
        svm_sweep.append(r)
        print(f"  lambda={lam:<7.0e} val F1={r['F1']:.3f} AUC={r['ROC-AUC']:.3f}  "
              f"margin={r['margin']:.3f}  ||w||={r['w_norm']:.2f}  SVs={r['n_support']:>6}  "
              f"J={r['objective']:.4f}")
    best_lam = max(svm_sweep, key=lambda r: (round(r["F1"], 4), r["lam"]))["lam"]

    # ---------------- RBF SVM via random Fourier features -----------------
    print("\n  4c. RBF-SVM (random Fourier features): lambda x gamma")
    rbf_sweep = []
    for lam in cfg["rbf_lambdas"]:
        for gm in cfg["rbf_gamma_mult"]:
            gamma = gm / d_feat
            m = RBFPegasosSVM(lambda_=lam, n_iters=cfg["svm_iters"], gamma=gamma,
                              n_components=cfg["n_components"], random_state=SEED).fit(Ztr, Ttr)
            r = _svm_row(m, Zva, Tva)
            r.update(lam=lam, gamma=gamma, gamma_mult=gm)
            rbf_sweep.append(r)
            print(f"  lambda={lam:<7.0e} gamma={gamma:.4f} (x{gm})  val F1={r['F1']:.3f}  "
                  f"AUC={r['ROC-AUC']:.3f}")
    best_rbf = max(rbf_sweep, key=lambda r: (round(r["F1"], 4), r["lam"]))
    print(f"  -> chosen on VALIDATION: linear lambda={best_lam:g}; "
          f"RBF lambda={best_rbf['lam']:g}, gamma={best_rbf['gamma']:.4f}")

    # ---------------- epochs / learning-rate schedule / decision boundary ---
    study = svm_study(D, cfg, best_lam, names=D["names"])

    # ---------------- final models on the TEST set -------------------------
    print("\n  4d. Final models on the TEST set (matched hyper-parameters)")
    n = len(Ttr)
    models = {}

    tree, t = _timed(lambda: DecisionTree(task="classification", **tree_cfg).fit(Xtr, Ttr))
    models["Ours: tree"] = (tree, "tree", t)
    skt, t = _timed(lambda: DecisionTreeClassifier(**tree_cfg, random_state=SEED).fit(Xtr, Ttr))
    models["sklearn: tree"] = (skt, "tree", t)

    svm, t = _timed(lambda: PegasosSVM(lambda_=best_lam, n_iters=cfg["svm_iters"],
                                       random_state=SEED).fit(Ztr, Ttr))
    models["Ours: SVM (Pegasos, linear)"] = (svm, "svm", t)
    # sklearn baselines: scaling INSIDE a Pipeline, hinge loss, alpha = lambda
    sgd, t = _timed(lambda: make_pipeline(
        StandardScaler(), SGDClassifier(loss="hinge", alpha=best_lam, max_iter=100,
                                        tol=None, random_state=SEED)).fit(Xtr, Ttr))
    models["sklearn: SGD hinge [Pipeline]"] = (sgd, "pipe", t)

    rbf, t = _timed(lambda: RBFPegasosSVM(
        lambda_=best_rbf["lam"], n_iters=cfg["svm_iters"], gamma=best_rbf["gamma"],
        n_components=cfg["n_components"], random_state=SEED).fit(Ztr, Ttr))
    models["Ours: SVM (Pegasos + RFF, RBF)"] = (rbf, "svm", t)
    sub = np.random.default_rng(SEED).choice(n, size=min(cfg["svc_n"], n), replace=False)
    skrbf, t = _timed(lambda: make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=1.0 / (best_rbf["lam"] * len(sub)),
                              gamma=best_rbf["gamma"])).fit(Xtr[sub], Ttr[sub]))
    models[f"sklearn: SVC RBF [Pipeline, {len(sub):,} rows]"] = (skrbf, "pipe", t)

    results, scores, preds = {}, {}, {}
    for name, (m, kind, t) in models.items():
        X = Zte if kind == "svm" else Xte            # ours-SVM gets pre-scaled features
        t0 = time.perf_counter()
        pred = m.predict(X)
        pred_s = time.perf_counter() - t0
        score = _tree_score(m, X) if kind == "tree" else m.decision_function(X)
        results[name] = with_uncertainty(
            clf_metrics(Tte, pred, score),
            lambda idx, pred=pred, score=score: clf_metrics(Tte[idx], pred[idx], score[idx]),
            len(Tte))
        results[name]["fit_s"], results[name]["pred_s"] = t, pred_s
        scores[name], preds[name] = score, pred
    print("  TEST set (metric ± bootstrap std over 200 resamples of the test set):")
    ev.compare(results)

    po, ps = preds["Ours: tree"], preds["sklearn: tree"]
    parity = dict(ours_depth=tree.get_depth(), sk_depth=skt.get_depth(),
                  ours_leaves=tree.get_n_leaves(), sk_leaves=skt.get_n_leaves(),
                  identical_prediction_share=float(np.mean(po == ps)))
    print(f"\n  Parity with sklearn tree: depth {parity['ours_depth']} vs {parity['sk_depth']}, "
          f"leaves {parity['ours_leaves']} vs {parity['sk_leaves']}, "
          f"identical predictions {100 * parity['identical_prediction_share']:.1f}%")

    _task_b_figures(D, tree_sweep, svm_sweep, scores, preds)

    return dict(tree_sweep=tree_sweep, tree_leaf_sweep=leaf_sweep, tree_cfg=tree_cfg,
                svm_sweep=[{k: v for k, v in r.items() if k != "history"} for r in svm_sweep],
                rbf_sweep=rbf_sweep, best_lambda=best_lam,
                best_rbf=dict(lam=best_rbf["lam"], gamma=best_rbf["gamma"]),
                svm_study=study, parity=parity,
                test=results, models=models, scores=scores, preds=preds)


def svm_study(D, cfg, lam, names) -> dict:
    """Problem 3(b): show that the number of epochs and the learning-rate
    schedule each change the objective and the margin, and draw a 2-D decision
    boundary with its margin lines."""
    print("\n  4b'. Pegasos SVM: epochs, learning-rate schedule, decision boundary "
          f"(lambda={lam:g})")
    Ztr, Zva, Ttr, Tva = D["Ztr"], D["Zva"], D["Ttr"], D["Tva"]

    def row(m, **extra):
        r = dict(J=m.objective_history_[-1][1], val_F1=ev.precision_recall_f1(
            Tva, m.predict(Zva))[2], margin=m.margin_width(), w_norm=m.weight_norm(),
            epochs=m.epochs_, steps=m.n_steps_)
        r.update(extra)
        return r

    epoch_rows = []
    for ep in cfg["epochs"]:
        m = PegasosSVM(lambda_=lam, n_epochs=ep, random_state=SEED).fit(Ztr, Ttr)
        epoch_rows.append(row(m))
        r = epoch_rows[-1]
        print(f"  epochs={ep:<5g} steps={r['steps']:>6}  J={r['J']:.4f}  val F1={r['val_F1']:.3f}  "
              f"margin={r['margin']:.3f}  ||w||={r['w_norm']:.2f}")

    schedules = [("pegasos", 1.0, "1/(λt) (Pegasos)"), ("constant", 0.01, "constant 0.01"),
                 ("inv_sqrt", 1.0, "1/√t")]
    sched_rows, histories = [], {}
    for name, eta0, label in schedules:
        m = PegasosSVM(lambda_=lam, n_iters=cfg["svm_iters"], lr_schedule=name, eta0=eta0,
                       random_state=SEED).fit(Ztr, Ttr)
        sched_rows.append(row(m, schedule=name, eta0=eta0))
        histories[label] = m.objective_history_
        r = sched_rows[-1]
        print(f"  schedule={name:<9} eta0={eta0:<5g} J={r['J']:.4f}  val F1={r['val_F1']:.3f}  "
              f"margin={r['margin']:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    eps = [r["epochs"] for r in epoch_rows]
    ax[0].loglog(eps, [r["J"] for r in epoch_rows], "o-", color="#4c72b0")
    ax[0].set(xlabel="epochs (passes over the data)", ylabel="final objective J",
              title="Training length")
    ax[1].semilogx(eps, [r["margin"] for r in epoch_rows], "o-", color="#c44e52", label="margin 2/‖w‖")
    ax[1].set(xlabel="epochs", ylabel="margin width", title="Margin vs epochs")
    ax1b = ax[1].twinx()
    ax1b.semilogx(eps, [r["val_F1"] for r in epoch_rows], "s--", color="k", label="val F1")
    ax1b.set_ylabel("validation F1")
    for label, hist in histories.items():
        steps, vals = zip(*hist)
        ax[2].semilogy(steps, vals, label=label)
    ax[2].set(xlabel="step", ylabel="primal objective J", title="Learning-rate schedule")
    ax[2].legend(fontsize=7)
    _save(fig, "svm_epochs_schedule.png")

    # ---- 2-D decision boundary on (area, rooms), standardized ----------------
    # area is extremely right-skewed (plots up to 10 ha), so use log(area) here
    ia, ir = names.index("area"), names.index("rooms")
    la_tr, la_va = np.log(D["Xtr"][:, ia]), np.log(D["Xva"][:, ia])
    mu, sd = la_tr.mean(), la_tr.std()                         # TRAIN statistics
    Z2tr = np.c_[(la_tr - mu) / sd, Ztr[:, ir]]
    Z2va = np.c_[(la_va - mu) / sd, Zva[:, ir]]
    rng = np.random.default_rng(SEED)
    pts = rng.choice(len(Tva), size=min(2500, len(Tva)), replace=False)
    lo = np.percentile(Z2va[:, 0], 1), np.percentile(Z2va[:, 0], 99)
    boundary = []
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for a, lam2 in zip(ax, (1e-4, 1e-1)):
        m = PegasosSVM(lambda_=lam2, n_iters=cfg["svm_iters"], random_state=SEED).fit(Z2tr, Ttr)
        a.scatter(Z2va[pts, 0], Z2va[pts, 1], c=Tva[pts], cmap="coolwarm", s=5, alpha=.4)
        x = np.linspace(*lo, 50)
        for c, ls in ((0, "-"), (1, "--"), (-1, "--")):
            if abs(m.w[1]) > 1e-9:
                a.plot(x, (c - m.b - m.w[0] * x) / m.w[1], "k" + ls, lw=1.2 if c == 0 else .8)
        a.set(xlim=lo, ylim=(np.percentile(Z2va[:, 1], 1) - .5, np.percentile(Z2va[:, 1], 99) + .5),
              xlabel="log area (standardized)",
              title=f"λ={lam2:g}: margin={m.margin_width():.2f}, SVs={m.n_support_:,}")
        boundary.append(dict(lam=lam2, margin=m.margin_width(), n_support=m.n_support_))
    ax[0].set_ylabel("rooms (standardized)")
    _save(fig, "svm_decision_boundary.png")

    return dict(epochs=epoch_rows, schedules=sched_rows, boundary=boundary)


def _task_b_figures(D, tree_sweep, svm_sweep, scores, preds) -> None:
    Tte = D["Tte"]

    # tree: criterion x depth
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for crit, mk in (("gini", "o"), ("entropy", "s")):
        rows = [r for r in tree_sweep if r["criterion"] == crit]
        ax.plot([r["depth"] for r in rows], [r["val_F1"] for r in rows], mk + "-",
                label=f"{crit} (validation)")
    rows = [r for r in tree_sweep if r["criterion"] == "gini"]
    ax.plot([r["depth"] for r in rows], [r["train_F1"] for r in rows], "k:", label="gini (train)")
    ax.set(xlabel="max_depth", ylabel="F1 (premium)", title="Task B: tree depth & criterion")
    ax.legend()
    _save(fig, "task_b_tree_depth_criterion.png")

    # SVM lambda sweep
    lam = [r["lam"] for r in svm_sweep]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].semilogx(lam, [r["F1"] for r in svm_sweep], "o-", label="F1")
    ax[0].semilogx(lam, [r["ROC-AUC"] for r in svm_sweep], "s-", label="ROC-AUC")
    ax[0].set(xlabel="λ", title="Validation metrics")
    ax[0].legend()
    ax[1].loglog(lam, [r["margin"] for r in svm_sweep], "o-", color="#c44e52")
    ax[1].set(xlabel="λ", ylabel="2/‖w‖", title="Margin width")
    ax[2].semilogx(lam, [r["n_support"] for r in svm_sweep], "o-", color="#8172b2")
    ax[2].set(xlabel="λ", ylabel="# support vectors", title="Support vectors")
    _save(fig, "svm_lambda_sweep.png")

    # convergence of the primal objective
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for r in svm_sweep:
        steps, vals = zip(*r["history"])
        ax.semilogy(steps, vals, label=f"λ={r['lam']:g}")
    ax.set(xlabel="Pegasos step", ylabel="primal objective J(w, b)",
           title="Convergence of the hinge-loss objective")
    ax.legend(fontsize=7)
    _save(fig, "svm_convergence.png")

    # confusion matrices
    show = ["Ours: tree", "Ours: SVM (Pegasos, linear)", "Ours: SVM (Pegasos + RFF, RBF)"]
    fig, ax = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.2))
    for a, name in zip(np.atleast_1d(ax), show):
        cm = ev.confusion_matrix(Tte, preds[name])
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, f"{v:,}", ha="center", va="center",
                   color="white" if v > cm.max() / 2 else "black")
        a.set(title=name.replace("Ours: ", ""), xticks=[0, 1], yticks=[0, 1],
              xticklabels=["standard", "premium"], yticklabels=["standard", "premium"],
              xlabel="predicted", ylabel="true")
    _save(fig, "task_b_confusion_matrices.png")

    # ROC curves
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, s in scores.items():
        fpr, tpr, _ = ev.roc_curve(Tte, s)
        short = name.replace("sklearn: ", "sk: ").replace("Ours: ", "")
        ax.plot(fpr, tpr, ls="-" if name.startswith("Ours") else "--",
                label=f"{short} ({ev.roc_auc(Tte, s):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=.8)
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate", title="ROC (test)")
    ax.legend(fontsize=6)
    _save(fig, "task_b_roc.png")


# =============================================================================
# 5. cross-validation (mean +- std)
# =============================================================================
def cross_validate(D, cfg, A, B) -> dict:
    _section(f"5. {cfg['folds']}-FOLD CROSS-VALIDATION on train+val  (mean ± std)")
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    Xtv, ytv = D["Xtv_raw"], D["ytv"]
    k = cfg["folds"]
    folds = np.array_split(np.random.default_rng(SEED).permutation(len(ytv)), k)
    a, tcfg = A["best"], B["tree_cfg"]
    lam, rbf = B["best_lambda"], B["best_rbf"]
    acc = {}

    def add(task, name, metrics):
        acc.setdefault(task, {}).setdefault(name, []).append(metrics)

    for i in range(k):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        Xtr, Xva = data_prep.impute_median(Xtv[tr], Xtv[va])       # fold-train statistics
        ytr, yva = ytv[tr], ytv[va]
        Ttr, thr = data_prep.make_tier_label(ytr)                    # fold-train median
        Tva, _ = data_prep.make_tier_label(yva, thr)
        Ztr, Zva = data_prep.standardize(Xtr, Xva)

        # Task A
        m = DecisionTree(task="regression", criterion="mse", **a).fit(Xtr, ytr)
        add("A", "Ours: tree", reg_metrics(yva, m.predict(Xva)))
        s = DecisionTreeRegressor(**a, random_state=SEED).fit(Xtr, ytr)
        add("A", "sklearn: tree", reg_metrics(yva, s.predict(Xva)))

        # Task B
        m = DecisionTree(task="classification", **tcfg).fit(Xtr, Ttr)
        add("B", "Ours: tree", clf_metrics(Tva, m.predict(Xva), _tree_score(m, Xva)))
        s = DecisionTreeClassifier(**tcfg, random_state=SEED).fit(Xtr, Ttr)
        add("B", "sklearn: tree", clf_metrics(Tva, s.predict(Xva), _tree_score(s, Xva)))

        m = PegasosSVM(lambda_=lam, n_iters=cfg["svm_iters"], random_state=SEED).fit(Ztr, Ttr)
        add("B", "Ours: SVM (linear)", clf_metrics(Tva, m.predict(Zva), m.decision_function(Zva)))
        s = make_pipeline(StandardScaler(), SGDClassifier(
            loss="hinge", alpha=lam, max_iter=100, tol=None, random_state=SEED)).fit(Xtr, Ttr)
        add("B", "sklearn: SGD hinge [Pipeline]",
            clf_metrics(Tva, s.predict(Xva), s.decision_function(Xva)))

        m = RBFPegasosSVM(lambda_=rbf["lam"], n_iters=cfg["svm_iters"], gamma=rbf["gamma"],
                          n_components=cfg["n_components"], random_state=SEED).fit(Ztr, Ttr)
        add("B", "Ours: SVM (RBF, RFF)", clf_metrics(Tva, m.predict(Zva), m.decision_function(Zva)))
        sub = np.random.default_rng(SEED).choice(len(Ttr), size=min(cfg["svc_n"], len(Ttr)),
                                                 replace=False)
        s = make_pipeline(StandardScaler(), SVC(
            kernel="rbf", C=1.0 / (rbf["lam"] * len(sub)), gamma=rbf["gamma"])
        ).fit(Xtr[sub], Ttr[sub])
        add("B", "sklearn: SVC RBF [Pipeline, subsample]",
            clf_metrics(Tva, s.predict(Xva), s.decision_function(Xva)))
        print(f"  fold {i + 1}/{k} done")

    summary = {}
    for task, models in acc.items():
        summary[task] = {}
        for name, rows in models.items():
            summary[task][name] = {met: (float(np.mean([r[met] for r in rows])),
                                         float(np.std([r[met] for r in rows])))
                                   for met in rows[0]}
    print("\n  Task A (regression), CV mean ± std:")
    ev.compare(summary["A"])
    print("\n  Task B (classification), CV mean ± std:")
    ev.compare(summary["B"])
    return summary


# =============================================================================
# 6. error analysis
# =============================================================================
def error_analysis(D, A, B) -> dict:
    _section("6. ERROR ANALYSIS")
    names, Xte, yte = D["names"], D["Xte"], D["yte"]
    out = {}

    # --- feature importance (tree) and SVM weights ---------------------------
    reg_imp = A["own_model"].feature_importances_
    clf_imp = B["models"]["Ours: tree"][0].feature_importances_
    w = B["models"]["Ours: SVM (Pegasos, linear)"][0].w          # standardized features

    def top(v, k=12):
        return np.argsort(-np.abs(v))[:k]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, v, title, col in ((ax[0], reg_imp, "Tree importance — Task A", "#4c72b0"),
                             (ax[1], clf_imp, "Tree importance — Task B", "#55a868"),
                             (ax[2], w, "SVM weights (standardized X)", "#c44e52")):
        idx = top(v)[::-1]
        a.barh([names[i][:28] for i in idx], v[idx], color=col)
        a.set_title(title, fontsize=9)
        a.tick_params(axis="y", labelsize=7)
    _save(fig, "feature_importance.png")
    out["top_tree_A"] = [(names[i], float(reg_imp[i])) for i in top(reg_imp, 8)]
    out["top_tree_B"] = [(names[i], float(clf_imp[i])) for i in top(clf_imp, 8)]
    out["top_svm_w"] = [(names[i], float(w[i])) for i in top(w, 8)]
    print("  Top tree features (Task A):", ", ".join(f"{n} ({v:.2f})" for n, v in out["top_tree_A"][:5]))
    print("  Top tree features (Task B):", ", ".join(f"{n} ({v:.2f})" for n, v in out["top_tree_B"][:5]))
    print("  Largest SVM weights       :", ", ".join(f"{n} ({v:+.2f})" for n, v in out["top_svm_w"][:5]))

    # --- regression errors by segment (decode category from the one-hot) -----
    cat_cols = [i for i, n in enumerate(names) if n.startswith("category_")]
    cats = np.array([names[cat_cols[j]][len("category_"):]
                     for j in np.argmax(Xte[:, cat_cols], axis=1)])
    pred = A["test_pred"]
    ape = np.abs(pred - yte) / yte
    rows = []
    print(f"\n  {'segment':<24}{'n':>7}{'median price':>14}{'MAE':>12}{'median |err| %':>16}")
    for c in sorted(set(cats), key=lambda c: -np.sum(cats == c)):
        m = cats == c
        row = dict(segment=c, n=int(m.sum()), median_price=float(np.median(yte[m])),
                   MAE=float(np.mean(np.abs(pred[m] - yte[m]))),
                   MdAPE=float(np.median(ape[m]) * 100))
        rows.append(row)
        print(f"  {c:<24}{row['n']:>7,}{row['median_price']:>14,.0f}{row['MAE']:>12,.0f}"
              f"{row['MdAPE']:>16.1f}")
    out["by_category"] = rows

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar([r["segment"] for r in rows], [r["MdAPE"] for r in rows], color="#8172b2")
    ax.set(ylabel="median absolute % error", title="Task A: error by property category")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, "task_a_error_by_category.png")

    # --- luxury outliers: error by price quartile ----------------------------
    q = data_prep.price_strata(yte)
    qrows = []
    print(f"\n  {'price quartile':<24}{'median price':>14}{'MAE':>12}{'median |err| %':>16}")
    for g, lab in enumerate(("Q1 (cheapest)", "Q2", "Q3", "Q4 (luxury)")):
        m = q == g
        row = dict(segment=lab, median_price=float(np.median(yte[m])),
                   MAE=float(np.mean(np.abs(pred[m] - yte[m]))),
                   MdAPE=float(np.median(ape[m]) * 100))
        qrows.append(row)
        print(f"  {lab:<24}{row['median_price']:>14,.0f}{row['MAE']:>12,.0f}{row['MdAPE']:>16.1f}")
    out["by_price_quartile"] = qrows

    # --- sparse districts: rare `location` levels were merged into "other" ----
    wrong = B["preds"]["Ours: tree"] != D["Tte"]
    if "location_other" in names:
        sparse = Xte[:, names.index("location_other")] == 1
        out["sparse_districts"] = dict(
            share_of_test=float(sparse.mean()),
            regression_MdAPE_sparse=float(np.median(ape[sparse]) * 100) if sparse.any() else None,
            regression_MdAPE_common=float(np.median(ape[~sparse]) * 100),
            tier_error_sparse=float(wrong[sparse].mean()) if sparse.any() else None,
            tier_error_common=float(wrong[~sparse].mean()))
        d = out["sparse_districts"]
        print(f"\n  Sparse districts ({100 * d['share_of_test']:.0f}% of test listings, "
              f"location merged into 'other'): Task A median |err| "
              f"{d['regression_MdAPE_sparse']:.1f}% vs {d['regression_MdAPE_common']:.1f}% elsewhere; "
              f"Task B tree error {100 * d['tier_error_sparse']:.1f}% vs "
              f"{100 * d['tier_error_common']:.1f}%.")

    # --- classification: where are the mistakes? -----------------------------
    near = np.abs(yte - D["thr"]) / D["thr"] < 0.15
    out["tree_B_error_near_threshold"] = float(wrong[near].mean())
    out["tree_B_error_far_from_threshold"] = float(wrong[~near].mean())
    print(f"\n  Task B tree error rate: {100 * wrong[near].mean():.1f}% for listings within ±15% of the "
          f"threshold vs {100 * wrong[~near].mean():.1f}% elsewhere "
          f"({100 * near.mean():.0f}% of test listings are near the threshold).")
    return out


# =============================================================================
# main
# =============================================================================
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Reproduce every number and figure.")
    ap.add_argument("--quick", action="store_true", help="small, fast smoke run")
    ap.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    args = ap.parse_args(argv)
    cfg = QUICK if args.quick else FULL
    t0 = time.perf_counter()

    D = prepare_data(cfg)
    eda_figures(D)
    A = task_a(D, cfg)
    B = task_b(D, cfg)
    cv = {} if args.no_cv else cross_validate(D, cfg, A, B)
    err = error_analysis(D, A, B)

    # ---- final comparison tables -------------------------------------------
    _section("FINAL RESULTS (held-out TEST set, hyper-parameters chosen on validation)")
    print("Task A — regression (price, AZN)")
    ev.compare(A["test"])
    print("\nTask B — classification (premium vs standard)")
    ev.compare(B["test"])

    REPORT_DIR.mkdir(exist_ok=True)
    payload = dict(
        seed=SEED, quick=args.quick, rows=len(D["df"]), features=len(D["names"]),
        tier_threshold=D["thr"],
        split=dict(train=len(D["ytr"]), val=len(D["yva"]), test=len(D["yte"])),
        tier_share=dict(train=float(D["Ttr"].mean()), val=float(D["Tva"].mean()),
                        test=float(D["Tte"].mean())),
        task_a=dict(best=A["best"], depth_sweep=A["sweep"], leaf_sweep=A["leaf_sweep"],
                    test=A["test"], parity_with_sklearn=A["parity"]),
        task_b=dict(tree_cfg=B["tree_cfg"], best_lambda=B["best_lambda"], best_rbf=B["best_rbf"],
                    tree_sweep=B["tree_sweep"], tree_leaf_sweep=B["tree_leaf_sweep"],
                    svm_sweep=B["svm_sweep"], rbf_sweep=B["rbf_sweep"], svm_study=B["svm_study"],
                    parity_with_sklearn=B["parity"], test=B["test"]),
        cross_validation=cv, error_analysis=err,
        runtime_seconds=time.perf_counter() - t0,
    )
    name = "results_quick" if args.quick else "results"
    (REPORT_DIR / f"{name}.json").write_text(json.dumps(_jsonable(payload), indent=2),
                                             encoding="utf-8")
    text = ["Task A — regression (test)", ev.format_table(A["test"]), "",
            "Task B — classification (test)", ev.format_table(B["test"])]
    if cv:
        text += ["", f"Task A — {cfg['folds']}-fold CV mean ± std", ev.format_table(cv["A"]), "",
                 f"Task B — {cfg['folds']}-fold CV mean ± std", ev.format_table(cv["B"])]
    (REPORT_DIR / f"{name}.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"\nSaved report/{name}.json and report/{name}.txt  "
          f"(total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
