"""Baseline models on feature set A vs C.

Run from the repo root:  python -m src.baseline
Optional:                python -m src.baseline --sample 50000
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, recall_score,
                             precision_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import TARGET, feature_columns, load_set_a, variant_tag

SEED = 42


def report(name, y_true, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "recall@thr": recall_score(y_true, pred),
        "precision@thr": precision_score(y_true, pred, zero_division=0),
    }


def run(feature_set, df, compliant=False):
    cols = feature_columns(df, feature_set, compliant)
    X, y = df[cols], df[TARGET]

    # Split FIRST (stratified); test set keeps the natural class balance.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=SEED)

    cat = [c for c in cols if str(X[c].dtype) == "category"]
    num = [c for c in cols if c not in cat]
    rows = []

    # 1) Dummy baseline
    dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    rows.append(report("Dummy", y_te, dummy.predict_proba(X_te)[:, 1]))

    # 2) Logistic regression (imputer/scaler/encoder fit on train only)
    def to_str(d):
        d = d.copy()
        d[cat] = d[cat].astype(str)
        return d

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
    ])
    lr = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(class_weight="balanced",
                                              max_iter=1000))])
    lr.fit(to_str(X_tr), y_tr)
    rows.append(report("LogReg (balanced)", y_te,
                       lr.predict_proba(to_str(X_te))[:, 1], threshold=0.5))

    # 3) LightGBM with scale_pos_weight + early stopping
    spw = (y_fit == 0).sum() / (y_fit == 1).sum()
    gbm = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        scale_pos_weight=spw, random_state=SEED, n_jobs=-1, verbose=-1)
    gbm.fit(X_fit, y_fit, eval_X=X_val, eval_y=y_val, eval_metric="auc",
            callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
    print(f"[{feature_set}] LightGBM stopped at iteration {gbm.best_iteration_}")
    rows.append(report("LightGBM", y_te, gbm.predict_proba(X_te)[:, 1],
                       threshold=0.5))

    out = pd.DataFrame(rows)
    out.insert(0, "feature_set", variant_tag(feature_set, compliant))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="stratified subsample size for quick dev runs")
    ap.add_argument("--sets", nargs="+", default=["A", "C", "B"],
                    help="feature sets to run (B needs src.aggregate first)")
    ap.add_argument("--compliant", action="store_true",
                    help="drop protected attributes (CODE_GENDER)")
    args = ap.parse_args()

    def get_data(s):
        if s == "B":
            from src.aggregate import load_set_b
            data = load_set_b()
        else:
            data = load_set_a()
        if args.sample:  # same seed + row order => same rows for every set
            data, _ = train_test_split(data, train_size=args.sample,
                                       stratify=data[TARGET], random_state=SEED)
        return data

    results = pd.concat([run(s, get_data(s), args.compliant) for s in args.sets],
                        ignore_index=True)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(results.to_string(index=False))

    import os
    os.makedirs("reports", exist_ok=True)
    suffix = "_compliant" if args.compliant else ""
    results.to_csv(f"reports/baseline_results{suffix}.csv", index=False)
