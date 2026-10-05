"""Optuna tuning for LightGBM, then save the final model.

Run from the repo root:
  python -m src.tune --set B --trials 30 --timeout 30   # timeout in minutes
  python -m src.tune --set C --trials 10                # thin-file model

Trained WITHOUT scale_pos_weight so predicted probabilities stay roughly
calibrated (the imbalance study showed ranking quality is unaffected), which
the cost-based threshold analysis in src.explain relies on.
Trial 0 is the untuned baseline config, so tuning can only match or improve it.
"""
import argparse
import json
import os

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.compare import SEED, load_split, score
from src.features import variant_tag

BASELINE = dict(num_leaves=31, learning_rate=0.05, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1e-3, reg_lambda=1e-3)


def split_train_val(X, y):
    return train_test_split(X, y, test_size=0.15, stratify=y,
                            random_state=SEED)


def fit(params, X_fit, y_fit, X_val, y_val):
    gbm = lgb.LGBMClassifier(n_estimators=3000, subsample_freq=1,
                             random_state=SEED, n_jobs=-1, verbose=-1,
                             **params)
    gbm.fit(X_fit, y_fit, eval_X=X_val, eval_y=y_val, eval_metric="auc",
            callbacks=[lgb.early_stopping(100, first_metric_only=True,
                                          verbose=False)])
    return gbm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="fset", default="B", choices=["A", "B", "C"])
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=30, help="minutes")
    ap.add_argument("--compliant", action="store_true",
                    help="drop protected attributes (CODE_GENDER)")
    a = ap.parse_args()
    tag = variant_tag(a.fset, a.compliant)

    X_tr, X_te, y_tr, y_te = load_split(a.fset, a.sample, a.compliant)
    X_fit, X_val, y_fit, y_val = split_train_val(X_tr, y_tr)

    def objective(trial):
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.8),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )
        gbm = fit(params, X_fit, y_fit, X_val, y_val)
        trial.set_user_attr("best_iter", int(gbm.best_iteration_))
        return roc_auc_score(y_val, gbm.predict_proba(X_val)[:, 1])

    def progress(study, trial):
        print(f"trial {trial.number:>3}  val AUC {trial.value:.4f}  "
              f"best {study.best_value:.4f}", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(BASELINE)
    study.optimize(objective, n_trials=a.trials, timeout=a.timeout * 60,
                   callbacks=[progress])

    best = study.best_params
    base_val = study.trials[0].value
    print(f"\nbaseline val AUC {base_val:.4f} -> tuned {study.best_value:.4f}")

    # Final model: best params, early-stopped on the same validation split.
    gbm = fit({**BASELINE, **best}, X_fit, y_fit, X_val, y_val)
    p = gbm.predict_proba(X_te)[:, 1]
    res = score(y_te, p)
    print("TEST:", {k: round(v, 4) for k, v in res.items()},
          f"| mean predicted prob {p.mean():.3f} vs base rate {y_te.mean():.3f}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    joblib.dump({"model": gbm, "feature_set": a.fset, "compliant": a.compliant,
                 "sample": a.sample,
                 "params": best}, f"models/lgbm_{tag}.joblib")
    with open(f"reports/best_params_{tag}.json", "w") as f:
        json.dump({**best, "n_trees": int(gbm.best_iteration_),
                   "val_auc_baseline": base_val,
                   "val_auc_tuned": study.best_value, "test": res}, f, indent=2)
    trials = study.trials_dataframe()
    trials.to_csv(f"reports/optuna_trials_{tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(trials["number"], trials["value"], "o", alpha=0.5, label="trial")
    ax.plot(trials["number"], trials["value"].cummax(), "r-", label="best so far")
    ax.set(xlabel="trial", ylabel="validation ROC-AUC",
           title=f"Optuna search (set {tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"reports/optuna_history_{tag}.png", dpi=150)


if __name__ == "__main__":
    main()
