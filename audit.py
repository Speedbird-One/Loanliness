"""Gender audit: research model (uses CODE_GENDER) vs compliant model (does not).

Run from the repo root, after tuning BOTH variants of a feature set:
  python -m src.tune --set B                 &  python -m src.tune --set B --compliant
  python -m src.audit --set B --cost-ratio 5

Answers three questions:
  1. What does dropping gender cost in accuracy? (paired-bootstrap AUC difference)
  2. Do outcomes still differ by gender without the column? (reject rates etc.)
  3. Is gender still inferable from the remaining features? (proxy check)

Both models use the same cost-based threshold 1/(1+R); the models are
well calibrated (see src.explain). No group-specific thresholds are used.
"""
import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.compare import SEED, load_split, recall_at_top
from src.features import variant_tag


def load_artifact(fs, compliant):
    return joblib.load(f"models/lgbm_{variant_tag(fs, compliant)}.joblib")


def overall_row(name, y, p, thr, R):
    pred = p >= thr
    fn = int(((~pred) & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return {"variant": name, "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "recall_top20pct": recall_at_top(y, p),
            "reject_rate": pred.mean(), "cost_per_applicant": (R * fn + fp) / len(y)}


def group_rows(name, y, p, g, thr):
    rows = []
    for grp in ("F", "M"):
        m = g == grp
        yy, pp = y[m], p[m]
        pred = pp >= thr
        rows.append({"variant": name, "gender": grp, "n": int(m.sum()),
                     "actual_default_rate": yy.mean(),
                     "mean_predicted_risk": pp.mean(),
                     "approval_rate": 1 - pred.mean(),
                     "defaulters_rejected": pred[yy == 1].mean(),
                     "good_borrowers_rejected": pred[yy == 0].mean(),
                     "auc_within_group": roc_auc_score(yy, pp)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="fset", default="B", choices=["A", "B", "C"])
    ap.add_argument("--cost-ratio", type=int, default=5)
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    R, thr = a.cost_ratio, 1 / (1 + a.cost_ratio)

    # Full split (with CODE_GENDER) so we can audit by gender; the compliant
    # model simply never looks at that column.
    res_art, cmp_art = load_artifact(a.fset, False), load_artifact(a.fset, True)
    assert res_art["sample"] == cmp_art["sample"], "variants tuned on different samples"
    X_tr, X_te, y_tr, y_te = load_split(a.fset, res_art["sample"], compliant=False)
    y = y_te.to_numpy()
    g = X_te["CODE_GENDER"].astype(object).to_numpy()
    res_m, cmp_m = res_art["model"], cmp_art["model"]
    p_res = res_m.predict_proba(X_te[res_m.feature_name_])[:, 1]
    p_cmp = cmp_m.predict_proba(X_te[cmp_m.feature_name_])[:, 1]
    print(f"threshold {thr:.3f} (R={R}); test n={len(y):,}, "
          f"gender known for {np.isin(g, ['F', 'M']).mean():.1%}")

    # 1) accuracy cost of dropping gender
    overall = pd.DataFrame([overall_row("research (with gender)", y, p_res, thr, R),
                            overall_row("compliant (no gender)", y, p_cmp, thr, R)])
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(a.boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum():
            diffs.append(roc_auc_score(y[i], p_res[i]) - roc_auc_score(y[i], p_cmp[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n1) Overall performance")
    print(overall.to_string(index=False))
    print(f"AUC cost of dropping gender: {np.mean(diffs):.4f} "
          f"(95% bootstrap CI {lo:.4f} to {hi:.4f})")

    # 2) outcomes by gender
    known = np.isin(g, ["F", "M"])
    grp = pd.DataFrame(group_rows("research (with gender)", y[known], p_res[known], g[known], thr)
                       + group_rows("compliant (no gender)", y[known], p_cmp[known], g[known], thr))
    print("\n2) Outcomes by gender")
    print(grp.to_string(index=False))
    ratios = {}
    for name, d in grp.groupby("variant"):
        f, m = d[d.gender == "F"].iloc[0], d[d.gender == "M"].iloc[0]
        ratios[name] = {
            "approval_ratio_f_m": float(f.approval_rate / m.approval_rate),
            "good_rejection_gap_f_m":
                float(f.good_borrowers_rejected - m.good_borrowers_rejected)}
        print(f"{name}: approval ratio F/M = {ratios[name]['approval_ratio_f_m']:.3f} | "
              f"good-borrower rejection gap F-M = "
              f"{ratios[name]['good_rejection_gap_f_m']:+.3f}")

    # 3) proxy check: can gender be predicted from the compliant features?
    feats = cmp_m.feature_name_
    ktr = np.isin(X_tr["CODE_GENDER"].astype(object).to_numpy(), ["F", "M"])
    gtr = (X_tr["CODE_GENDER"].astype(object).to_numpy() == "M")
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, verbose=-1,
                             n_jobs=-1, random_state=SEED)
    clf.fit(X_tr[feats][ktr], gtr[ktr])
    gte = g[known] == "M"
    auc_g = roc_auc_score(gte, clf.predict_proba(X_te[feats][known])[:, 1])
    top = (pd.Series(clf.booster_.feature_importance("gain"), index=feats)
           .sort_values(ascending=False).head(6))
    print(f"\n3) Proxy check: gender predicted from the compliant features, "
          f"AUC = {auc_g:.3f} (0.5 = not inferable)")
    print("   strongest proxies:", ", ".join(top.index))
    risk_auc = roc_auc_score(gte, p_cmp[known])
    print(f"   compliant risk score vs gender (AUC): {risk_auc:.3f}")

    overall.to_csv(f"reports/audit_overall_{a.fset}.csv", index=False)
    grp.to_csv(f"reports/audit_gender_{a.fset}.csv", index=False)
    with open(f"reports/audit_summary_{a.fset}.json", "w") as f:
        json.dump({"cost_ratio": R, "threshold": thr,
                   "auc_cost": float(np.mean(diffs)),
                   "auc_cost_ci": [float(lo), float(hi)],
                   "proxy_auc": float(auc_g), "proxies": list(top.index),
                   "risk_score_gender_auc": float(risk_auc),
                   "by_variant": ratios}, f, indent=2)


if __name__ == "__main__":
    main()
