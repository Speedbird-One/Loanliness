"""SHAP explanations + cost-based decision threshold for the tuned model.

Run from the repo root (after src.tune):
  python -m src.explain --set B --cost-ratio 5

Outputs (reports/): shap_summary_<set>.png, shap_importance_<set>.csv,
shap_by_source_<set>.csv, threshold_analysis_<set>.csv,
threshold_cost_curve_<set>.png, threshold_<set>.json (used by the demo app).

Cost model: rejecting a good borrower costs 1 unit (lost margin); approving
a defaulter costs R units (lost principal). R is an assumption, so a sweep is
reported. Thresholds are chosen on the validation split and reported on test.
With calibrated probabilities the theoretical optimum is 1 / (1 + R).
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.compare import SEED, load_split
from src.features import variant_tag
from src.tune import split_train_val

SOURCES = [("BUREAU_", "bureau (+bureau_balance)"), ("PREV_", "previous_application"),
           ("POS_", "POS_CASH_balance"), ("INST_", "installments_payments"),
           ("CC_", "credit_card_balance"), ("EXT_SOURCE", "EXT_SOURCE_* scores")]


def source_of(col):
    for prefix, name in SOURCES:
        if col.startswith(prefix):
            return name
    return "application (other)"


def shap_analysis(gbm, X_te, fs, n):
    Xs = X_te.sample(min(n, len(X_te)), random_state=SEED)
    # LightGBM's own exact TreeSHAP; last column is the bias term.
    contrib = gbm.predict(Xs, pred_contrib=True)[:, :-1]
    imp = (pd.DataFrame({"feature": Xs.columns,
                         "mean_abs_shap": np.abs(contrib).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False))
    imp.to_csv(f"reports/shap_importance_{fs}.csv", index=False)
    print("\nTop 15 features by mean |SHAP|:")
    print(imp.head(15).to_string(index=False))

    imp["source"] = imp["feature"].map(source_of)
    by_src = imp.groupby("source")["mean_abs_shap"].sum()
    by_src = (by_src / by_src.sum()).sort_values(ascending=False).rename("share")
    by_src.to_csv(f"reports/shap_by_source_{fs}.csv")
    print("\nShare of total attribution by source:")
    print(by_src.map("{:.1%}".format).to_string())

    Xp = Xs.copy()  # colour the beeswarm with category codes
    for c in Xp.columns:
        if str(Xp[c].dtype) == "category":
            Xp[c] = Xp[c].cat.codes.astype(float).replace(-1, np.nan)
    shap.summary_plot(contrib, Xp, max_display=20, show=False)
    plt.savefig(f"reports/shap_summary_{fs}.png", dpi=150, bbox_inches="tight")
    plt.close()


def cost_at(y, p, t, R):
    pred = p >= t                      # predicted 1 = reject
    fn = int(((~pred) & (y == 1)).sum())   # approved a defaulter
    fp = int((pred & (y == 0)).sum())      # rejected a good borrower
    tp = int((pred & (y == 1)).sum())
    n = len(y)
    return {"reject_rate": pred.mean(), "defaulters_caught": tp / max(y.sum(), 1),
            "precision": tp / max(pred.sum(), 1),
            "cost_per_applicant": (R * fn + fp) / n}


def best_threshold(y, p, R, grid):
    costs = [R * (((p < t) & (y == 1)).sum()) + ((p >= t) & (y == 0)).sum()
             for t in grid]
    return float(grid[int(np.argmin(costs))]), np.array(costs) / len(y)


def threshold_analysis(gbm, X_tr, y_tr, X_te, y_te, fs, R_main):
    _, X_val, _, y_val = split_train_val(X_tr, y_tr)
    yv, yt = y_val.to_numpy(), y_te.to_numpy()
    pv, pt = gbm.predict_proba(X_val)[:, 1], gbm.predict_proba(X_te)[:, 1]
    print(f"\nCalibration check: mean predicted {pt.mean():.3f} vs actual rate {yt.mean():.3f}")

    grid = np.linspace(0.005, 0.8, 400)
    rows, main = [], None
    for R in sorted({2, 5, 10, 20, R_main}):
        t, curve = best_threshold(yv, pv, R, grid)
        if R == R_main:
            main = (t, curve)
        approve_all = R * yt.mean()
        for label, thr in [("cost-optimal", t), ("default 0.5", 0.5)]:
            m = cost_at(yt, pt, thr, R)
            rows.append({"cost_ratio": R, "rule": label, "threshold": thr,
                         **m, "approve_all_cost": approve_all,
                         "saving_vs_approve_all":
                             1 - m["cost_per_applicant"] / approve_all})
    res = pd.DataFrame(rows)
    res.to_csv(f"reports/threshold_analysis_{fs}.csv", index=False)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(res.to_string(index=False))

    t, curve = main
    print(f"\nChosen threshold for R={R_main}: {t:.3f} "
          f"(theory for calibrated model: {1/(1+R_main):.3f})")
    with open(f"reports/threshold_{fs}.json", "w") as f:
        json.dump({"cost_ratio": R_main, "threshold": t}, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(grid, curve)
    ax.axvline(t, color="r", ls="--", label=f"optimum {t:.2f}")
    ax.axvline(0.5, color="gray", ls=":", label="default 0.5")
    ax.set(xlabel="reject if P(default) >= threshold",
           ylabel="validation cost per applicant",
           title=f"Cost curve (miss costs {R_main}x, set {fs})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"reports/threshold_cost_curve_{fs}.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="fset", default="B", choices=["A", "B", "C"])
    ap.add_argument("--cost-ratio", type=int, default=5)
    ap.add_argument("--shap-n", type=int, default=5000)
    ap.add_argument("--compliant", action="store_true",
                    help="explain the model trained without CODE_GENDER")
    a = ap.parse_args()
    tag = variant_tag(a.fset, a.compliant)

    art = joblib.load(f"models/lgbm_{tag}.joblib")
    X_tr, X_te, y_tr, y_te = load_split(a.fset, art["sample"], a.compliant)
    shap_analysis(art["model"], X_te, tag, a.shap_n)
    threshold_analysis(art["model"], X_tr, y_tr, X_te, y_te, tag, a.cost_ratio)


if __name__ == "__main__":
    main()
