"""Collect every result in reports/ into one markdown file: RESULTS.md (repo root)

Run from the repo root after the experiments:  python -m src.summarize
Sections whose input files do not exist yet are skipped with a note, so it
is safe to run at any point. Copy tables from RESULTS.md into the slides and
write-up instead of retyping numbers.
"""
import json
from pathlib import Path

import pandas as pd

REPORTS = Path("reports")
OUT = Path("RESULTS.md")  # repo root, easy to find
SETS = [("C", "C: thin-file"), ("A", "A: application only"),
        ("B", "B: application + aggregated tables")]
VARIANTS = [("", "with gender"), ("_compliant", "without gender")]


def md(df, digits=3):
    """DataFrame -> GitHub markdown table (no tabulate dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.{digits}f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |"
              for r in df.itertuples(index=False)]
    return "\n".join(lines)


def csv(name):
    p = REPORTS / name
    return pd.read_csv(p) if p.exists() else None


def js(name):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def section(title, body, note=""):
    out = [f"## {title}", ""]
    if note:
        out += [note, ""]
    out += [body if body else "_Not available yet (input files missing)._", ""]
    return out


def ladder():
    rows = []
    for code, label in SETS:
        for suf, vlabel in VARIANTS:
            j = js(f"best_params_{code}{suf}.json")
            if j:
                t = j["test"]
                rows.append({"feature set": label, "variant": vlabel,
                             "test ROC-AUC": t["roc_auc"], "PR-AUC": t["pr_auc"],
                             "recall in riskiest 20%": t["recall_top20pct"],
                             "trees": j.get("n_trees", ""),
                             "val AUC untuned": j["val_auc_baseline"],
                             "val AUC tuned": j["val_auc_tuned"]})
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    text = [md(df), ""]
    for _, vlabel in VARIANTS:
        sub = df[df.variant == vlabel]
        d = pd.Series(sub["test ROC-AUC"].to_numpy(), index=sub["feature set"].str[0])
        if {"A", "B", "C"} <= set(d.index):
            text.append(f"- {vlabel}: A vs C = {d['A'] - d['C']:+.3f} AUC, "
                        f"B vs A = {d['B'] - d['A']:+.3f} AUC")
    return "\n".join(text)


def untuned():
    parts = []
    for suf, vlabel in VARIANTS:
        df = csv(f"baseline_results{suf}.csv")
        if df is not None:
            parts.append(f"**{vlabel}**\n\n" + md(df))
    return "\n\n".join(parts)


def model_comparison():
    frames = [csv(f"model_comparison_{c}.csv") for c, _ in SETS]
    frames = [f for f in frames if f is not None]
    if not frames:
        return ""
    df = pd.concat(frames)
    order = [c for c, _ in SETS if c in set(df.feature_set)]
    out = []
    for metric in ("roc_auc", "recall_top20pct"):
        pv = df.pivot(index="model", columns="feature_set", values=metric)[order]
        out += [f"**{metric}**", "", md(pv.sort_values(order[-1], ascending=False)
                                       .reset_index()), ""]
    out += ["**All metrics**", "", md(df)]
    return "\n".join(out)


def imbalance():
    frames = [csv(f"imbalance_comparison_{c}.csv") for c, _ in SETS]
    frames = [f for f in frames if f is not None]
    return md(pd.concat(frames)) if frames else ""


def thresholds():
    rows, sweep = [], []
    for code, label in SETS:
        for suf, vlabel in VARIANTS:
            df = csv(f"threshold_analysis_{code}{suf}.csv")
            if df is None:
                continue
            opt = df[df.rule == "cost-optimal"]
            base = (df[df.rule == "default 0.5"][["cost_ratio", "saving_vs_approve_all"]]
                    .rename(columns={"saving_vs_approve_all": "saving at 0.5"}))
            m = opt.merge(base, on="cost_ratio")
            m.insert(0, "variant", vlabel)
            m.insert(0, "feature set", label)
            keep = ["feature set", "variant", "cost_ratio", "threshold",
                    "reject_rate", "defaulters_caught", "precision",
                    "saving_vs_approve_all", "saving at 0.5"]
            rows.append(m[m.cost_ratio == 5][keep])
            if suf == "":
                sweep.append(m[keep])
    if not rows:
        return ""
    return ("**Headline: R = 5** (a missed defaulter costs 5x a rejected good borrower)\n\n"
            + md(pd.concat(rows)) + "\n\n**Sensitivity to R (with-gender models)**\n\n"
            + md(pd.concat(sweep)))


def shap_tables():
    tags = [f"{c}{suf}" for c, _ in SETS for suf, _ in VARIANTS]
    top, share = {}, {}
    for t in tags:
        imp, src = csv(f"shap_importance_{t}.csv"), csv(f"shap_by_source_{t}.csv")
        if imp is not None:
            top[t] = imp.feature.head(10).tolist()
        if src is not None:
            share[t] = src.set_index("source")["share"]
    if not top:
        return ""
    t1 = pd.DataFrame(top, index=range(1, 11)).rename_axis("rank").reset_index()
    t2 = pd.DataFrame(share).rename_axis("source").reset_index()
    return ("**Top 10 features by mean |SHAP|**\n\n" + md(t1)
            + "\n\n**Share of total attribution by source table**\n\n" + md(t2))


def audit():
    over, grp, summ = [], [], []
    for code, _ in SETS:
        o, g, j = (csv(f"audit_overall_{code}.csv"), csv(f"audit_gender_{code}.csv"),
                   js(f"audit_summary_{code}.json"))
        if o is not None:
            o.insert(0, "set", code); over.append(o)
        if g is not None:
            g.insert(0, "set", code); grp.append(g)
        if j:
            by = j["by_variant"]
            summ.append({"set": code, "AUC cost of dropping gender": j["auc_cost"],
                         "95% CI": f"{j['auc_cost_ci'][0]:+.4f} to {j['auc_cost_ci'][1]:+.4f}",
                         "gender predictable from remaining features (AUC)": j["proxy_auc"],
                         "strongest proxies": ", ".join(j["proxies"][:4]),
                         "approval ratio F/M (with gender)":
                             next((v["approval_ratio_f_m"] for k, v in by.items() if "research" in k), float("nan")),
                         "approval ratio F/M (without gender)":
                             next((v["approval_ratio_f_m"] for k, v in by.items() if "compliant" in k), float("nan"))})
    if not over:
        return ""
    out = []
    if summ:
        out += ["**Summary**", "", md(pd.DataFrame(summ), 4), ""]
    out += ["**Overall (threshold 1/(1+R))**", "", md(pd.concat(over)), "",
            "**By gender**", "", md(pd.concat(grp))]
    return "\n".join(out)


def main():
    lines = ["# Results summary", "",
             "_Generated by `python -m src.summarize` from the files in `reports/`. "
             "All test metrics use the same stratified 80/20 split (seed 42); the "
             "test set keeps the natural ~8% default rate._", ""]
    lines += section("1. Feature-set ladder (tuned LightGBM, test set)", ladder(),
                     "C = no EXT_SOURCE / bureau-enquiry columns (thin-file), "
                     "A = application table, B = A + aggregated auxiliary tables.")
    lines += section("2. Untuned baselines", untuned())
    lines += section("3. Model comparison", model_comparison(),
                     "LogReg / RF / LightGBM use class weights; Naive Bayes and MLP "
                     "train on an under-sampled balanced copy.")
    lines += section("4. Imbalance strategies (logistic regression)", imbalance())
    lines += section("5. Cost-based decision threshold", thresholds(),
                     "'saving' = reduction in expected cost per applicant versus "
                     "approving everyone.")
    lines += section("6. SHAP feature attribution", shap_tables())
    lines += section("7. Gender audit", audit())
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
