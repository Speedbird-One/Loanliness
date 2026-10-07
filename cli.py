"""Command-line version of the demo (same models and logic as app.py).

Run from the repo root:
  python cli.py assess                      # score the default example applicant
  python cli.py assess --income 90000 --credit 400000 --annuity 22000 \
        --goods 360000 --age 29 --years-employed 2 --education "Higher education"
  python cli.py assess -i                   # prompt for every field
  python cli.py assess --variant compliant --cost-ratio 10
  python cli.py replay --kind defaulter --seed 3   # real held-out applicant
  python cli.py assess --list-options       # valid values for the choice fields

Fields not asked for (region, documents, external scores...) are passed to
the model as missing, exactly as in the app.
"""

import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress minor scikit-learn version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import argparse
import sys

import numpy as np

from src.serve import (build_row, load_heldout, load_models, load_schema,
                       make_raw, predict, threshold, top_contributions)

VARIANTS = {"research": "With gender (statistics)",
            "compliant": "Without gender (compliance)"}

# (argument, type, default, help)
NUMERIC = [
    ("income", float, 150_000, "annual income"),
    ("credit", float, 500_000, "loan amount"),
    ("annuity", float, 25_000, "annual repayment"),
    ("goods", float, 450_000, "price of goods financed"),
    ("age", int, 35, "age in years"),
    ("years_employed", float, 5.0, "years in current job"),
    ("children", int, 0, "number of children"),
    ("family_members", int, 2, "family members"),
    ("car_age", float, 5.0, "car age in years (used if the applicant owns a car)"),
]
# (argument, dataset column, default); default None = leave missing
CATEGORICAL = [
    ("education", "NAME_EDUCATION_TYPE", "Secondary / secondary special"),
    ("income_type", "NAME_INCOME_TYPE", "Working"),
    ("family_status", "NAME_FAMILY_STATUS", "Married"),
    ("housing", "NAME_HOUSING_TYPE", "House / apartment"),
    ("occupation", "OCCUPATION_TYPE", None),
    ("contract", "NAME_CONTRACT_TYPE", "Cash loans"),
    ("own_car", "FLAG_OWN_CAR", "N"),
    ("own_realty", "FLAG_OWN_REALTY", "Y"),
    ("gender", "CODE_GENDER", None),
]


def flag(name):
    return "--" + name.replace("_", "-")


def build_parser():
    common = argparse.ArgumentParser(add_help=False)  # shared by both commands
    common.add_argument("--variant", choices=list(VARIANTS), default="research",
                        help="model variant (default: research = with gender)")
    common.add_argument("--cost-ratio", type=int, default=5,
                        help="a missed defaulter costs this many times a rejected "
                             "good borrower; reject if P(default) >= 1/(1+R)")
    common.add_argument("--top", type=int, default=8, help="drivers to show")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sa = sub.add_parser("assess", parents=[common],
                        help="score a new applicant (no credit history)")
    for name, typ, default, helptext in NUMERIC:
        sa.add_argument(flag(name), type=typ, default=default, help=helptext)
    for name, col, default in CATEGORICAL:
        sa.add_argument(flag(name), default=default, help=col)
    sa.add_argument("--unemployed", action="store_true",
                    help="not currently employed (ignores --years-employed)")
    sa.add_argument("-i", "--interactive", action="store_true",
                    help="prompt for each field (Enter keeps the default)")
    sa.add_argument("--list-options", action="store_true",
                    help="print valid values for the choice fields and exit")

    sr = sub.add_parser("replay", parents=[common],
                        help="replay a real held-out applicant")
    sr.add_argument("--kind", choices=["any", "defaulter", "repaid"], default="any")
    sr.add_argument("--seed", type=int, default=0)
    return ap


def ask(label, default, cast, options=None):
    shown = "skip" if default is None else default
    while True:
        text = input(f"{label} [{shown}]: ").strip()
        if not text:
            return default
        try:
            value = cast(text)
        except ValueError:
            print("  please enter a number")
            continue
        if options is not None and value not in options:
            print("  choose one of:", " | ".join(options))
            continue
        return value


def report(title, art, row, thr, k):
    p = predict(art, row)
    labels, values = top_contributions(art, row, k)
    print(f"\n== {title} ==")
    print(f"P(default) = {p:.1%}   reject threshold = {thr:.1%}   ->  "
          f"{'REJECT' if p >= thr else 'APPROVE'}")
    print("Top drivers (log-odds of default; + raises risk):")
    scale = max(float(np.abs(values).max()), 1e-9)
    for lab, v in zip(labels, values):
        bar = ("+" if v > 0 else "-") * max(1, round(abs(v) / scale * 20))
        print(f"  {v:+.3f} {bar:<20} {lab}")


def cmd_assess(a, models):
    schema = load_schema()
    options = {col: list(schema[col].cat.categories) for _, col, _ in CATEGORICAL}
    if a.list_options:
        for name, col, _ in CATEGORICAL:
            print(f"{flag(name)}  ({col}):\n    " + " | ".join(options[col]))
        return
    vals = {n: getattr(a, n) for n, *_ in NUMERIC + CATEGORICAL}
    employed = not a.unemployed
    if a.interactive:
        for name, typ, default, helptext in NUMERIC:
            vals[name] = ask(helptext, vals[name], typ)
        for name, col, _ in CATEGORICAL:
            if name == "gender" and a.variant != "research":
                continue
            vals[name] = ask(col, vals[name], str, options[col])
        employed = input("Currently employed? [Y/n]: ").strip().lower() != "n"
    for name, col, _ in CATEGORICAL:  # validate non-interactive values
        v = vals[name]
        if v is not None and v not in options[col]:
            sys.exit(f"{flag(name)}: '{v}' is not valid. Options: "
                     + " | ".join(options[col]))
    if vals["gender"] and a.variant != "research":
        print("note: --gender is ignored by the compliant variant")
    raw = make_raw(
        vals["income"], vals["credit"], vals["annuity"], vals["goods"],
        vals["age"], employed, vals["years_employed"], vals["children"],
        vals["family_members"], vals["education"], vals["income_type"],
        vals["family_status"], vals["housing"], vals["occupation"],
        vals["contract"], vals["own_car"], vals["car_age"], vals["own_realty"],
        gender=vals["gender"] if a.variant == "research" else None)
    row = build_row(schema, raw)
    report("Thin-file model (no credit-history features)", models["C"], row,
           threshold(a.cost_ratio), a.top)


def cmd_replay(a, models):
    print("Loading held-out applicants...", flush=True)
    X_te, y_te = load_heldout(models["B"]["sample"])
    y = y_te.to_numpy()
    mask = {"any": np.ones(len(y), bool), "defaulter": y == 1,
            "repaid": y == 0}[a.kind]
    pos = np.random.default_rng(a.seed).choice(np.flatnonzero(mask))
    row = X_te.iloc[[pos]]
    print(f"Actual outcome: {'DEFAULTED' if y[pos] == 1 else 'repaid'}")
    thr = threshold(a.cost_ratio)
    report("Thin-file model", models["C"], row, thr, a.top)
    report("Full model (with credit history)", models["B"], row, thr, a.top)


def main():
    a = build_parser().parse_args()
    try:
        all_models = load_models()
        if a.variant not in all_models:
            sys.exit(f"No '{a.variant}' model pair in models/. Available: "
                     f"{', '.join(all_models) or 'none'}. Run src.tune first.")
        models = all_models[a.variant]
        print(f"Variant: {VARIANTS[a.variant]} | cost ratio R={a.cost_ratio} "
              f"-> reject if P(default) >= {threshold(a.cost_ratio):.1%}")
        (cmd_assess if a.cmd == "assess" else cmd_replay)(a, models)
    except FileNotFoundError as e:
        sys.exit(f"Missing file: {e.filename}. Run the pipeline first (see README).")


if __name__ == "__main__":
    main()
