"""Model zoo + imbalance-strategy comparison.

Run from the repo root:
  python -m src.compare zoo        [--set B] [--sample 50000]
  python -m src.compare imbalance  [--set A] [--sample 50000]

zoo        LogReg / Naive Bayes / Random Forest / MLP / LightGBM on one
           feature set, with ROC and PR curves (reports/roc_pr_<set>.png).
imbalance  none vs class weights vs under-sampling vs SMOTE (LogReg).

All preprocessing is fit on the training split only; the test split keeps
the natural class balance. Metrics are threshold-free (ROC-AUC, PR-AUC) plus
recall among the 20% highest-risk applicants, so models with differently
calibrated probabilities can be compared fairly.
"""
import argparse
import os
import time

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import TARGET, feature_columns, load_set_a, variant_tag

SEED = 42


# ---------------------------------------------------------------- helpers
def load_split(feature_set, sample, compliant=False):
    if feature_set == "B":
        from src.aggregate import load_set_b
        df = load_set_b()
    else:  # A and C both live on the application table
        df = load_set_a()
    if sample:
        df, _ = train_test_split(df, train_size=sample, stratify=df[TARGET],
                                 random_state=SEED)
    cols = feature_columns(df, feature_set, compliant)
    return train_test_split(df[cols], df[TARGET], test_size=0.2,
                            stratify=df[TARGET], random_state=SEED)


def recall_at_top(y, p, frac=0.2):
    y = np.asarray(y)
    top = np.argsort(-p)[: int(len(y) * frac)]
    return y[top].sum() / y.sum()


def score(y, p):
    return {"roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "recall_top20pct": recall_at_top(y, p)}


def make_dense(X_tr, X_te):
    """Median-impute + scale numerics, one-hot categoricals (train-fit only)."""
    cat = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
    num = [c for c in X_tr.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                              sparse_output=False, dtype=np.float32), cat),
    ])

    def as_str(X):
        X = X.copy()
        X[cat] = X[cat].astype(str)
        return X

    A = pre.fit_transform(as_str(X_tr)).astype(np.float32)
    B = pre.transform(as_str(X_te)).astype(np.float32)
    print(f"preprocessed: {A.shape[1]} columns after imputing/encoding")
    return A, B


def fit_lgbm(X_tr, y_tr, X_te):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=SEED)
    gbm = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.7,
        scale_pos_weight=(y_fit == 0).sum() / (y_fit == 1).sum(),
        random_state=SEED, n_jobs=-1, verbose=-1)
    gbm.fit(X_fit, y_fit, eval_X=X_val, eval_y=y_val, eval_metric="auc",
            callbacks=[lgb.early_stopping(50, first_metric_only=True,
                                          verbose=False)])
    return gbm.predict_proba(X_te)[:, 1]


def save(df, name):
    os.makedirs("reports", exist_ok=True)
    df.to_csv(f"reports/{name}.csv", index=False)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(df.to_string(index=False))


# ------------------------------------------------------------------- zoo
def zoo(feature_set, sample, compliant=False):
    tag = variant_tag(feature_set, compliant)
    X_tr, X_te, y_tr, y_te = load_split(feature_set, sample, compliant)
    A_tr, A_te = make_dense(X_tr, X_te)
    A_bal, y_bal = RandomUnderSampler(random_state=SEED).fit_resample(A_tr, y_tr)

    # (name, estimator, use balanced-by-undersampling data?)
    models = [
        ("Logistic Regression",
         LogisticRegression(class_weight="balanced", max_iter=1000), False),
        ("Naive Bayes", GaussianNB(), True),
        ("Random Forest",
         RandomForestClassifier(n_estimators=200, max_depth=12,
                                min_samples_leaf=50, max_features="sqrt",
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED), False),
        ("MLP", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                              early_stopping=True, n_iter_no_change=5,
                              max_iter=50, random_state=SEED), True),
    ]
    rows, probs = [], {}
    for name, est, balanced in models:
        t0 = time.time()
        Xf, yf = (A_bal, y_bal) if balanced else (A_tr, y_tr)
        est.fit(Xf, yf)
        probs[name] = est.predict_proba(A_te)[:, 1]
        rows.append({"model": name, **score(y_te, probs[name]),
                     "seconds": time.time() - t0})
        print(f"done {name} ({rows[-1]['seconds']:.0f}s)", flush=True)

    t0 = time.time()
    probs["LightGBM"] = fit_lgbm(X_tr, y_tr, X_te)
    rows.append({"model": "LightGBM", **score(y_te, probs["LightGBM"]),
                 "seconds": time.time() - t0})

    res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    res.insert(0, "feature_set", tag)
    save(res, f"model_comparison_{tag}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_te, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax[1].plot(rc, pr, label=f"{name} ({average_precision_score(y_te, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(title=f"ROC (set {tag})", xlabel="False positive rate",
              ylabel="True positive rate")
    ax[1].axhline(y_te.mean(), color="k", ls="--", lw=1)
    ax[1].set(title=f"Precision-recall (set {tag})",
              xlabel="Recall", ylabel="Precision")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"reports/roc_pr_{tag}.png", dpi=150)


# ------------------------------------------------------------- imbalance
def imbalance(feature_set, sample, compliant=False):
    tag = variant_tag(feature_set, compliant)
    X_tr, X_te, y_tr, y_te = load_split(feature_set, sample, compliant)
    A_tr, A_te = make_dense(X_tr, X_te)

    strategies = {
        "none": (None, None),
        "class weights": (None, "balanced"),
        "under-sampling": (RandomUnderSampler(random_state=SEED), None),
        "SMOTE": (SMOTE(random_state=SEED), None),
    }
    rows = []
    for name, (sampler, cw) in strategies.items():
        t0 = time.time()
        Xf, yf = (sampler.fit_resample(A_tr, y_tr) if sampler else (A_tr, y_tr))
        clf = LogisticRegression(class_weight=cw, max_iter=1000).fit(Xf, yf)
        p = clf.predict_proba(A_te)[:, 1]
        rows.append({"strategy": name, "train_rows": len(yf), **score(y_te, p),
                     "seconds": time.time() - t0})
        print(f"done {name}", flush=True)
    res = pd.DataFrame(rows)
    res.insert(0, "feature_set", tag)
    save(res, f"imbalance_comparison_{tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["zoo", "imbalance"])
    ap.add_argument("--set", dest="fset", choices=["A", "B", "C"])
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--compliant", action="store_true",
                    help="drop protected attributes (CODE_GENDER)")
    a = ap.parse_args()
    if a.task == "zoo":
        zoo(a.fset or "B", a.sample, a.compliant)
    else:
        imbalance(a.fset or "A", a.sample, a.compliant)
