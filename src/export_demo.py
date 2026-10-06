"""Write the small files the demo needs, so it runs from a fresh clone.

Run from the repo root after training (src.tune) and before committing:
  python -m src.export_demo [--n 3000]

Creates, in demo_data/ (committed to git):
  schema.json            column order and category levels of the application form
  heldout_sample.parquet a stratified sample of applicants from the TEST split
                         (never seen in training or tuning), with their outcome

With these plus models/, `streamlit run app.py` and `python cli.py ...` need
neither the Kaggle data nor any training.
"""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.compare import SEED, load_split
from src.features import feature_columns, load_set_a

OUT = Path("demo_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="held-out applicants to keep")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    df = load_set_a()
    cols = feature_columns(df, "A")
    cats = {c: [str(x) for x in df[c].cat.categories]
            for c in cols if str(df[c].dtype) == "category"}
    (OUT / "schema.json").write_text(
        json.dumps({"columns": cols, "categories": cats}, indent=1), encoding="utf-8")

    sample = joblib.load("models/lgbm_B.joblib")["sample"]
    _, X_te, _, y_te = load_split("B", sample)  # the real, untouched test split
    if a.n < len(X_te):
        X_s, _, y_s, _ = train_test_split(X_te, y_te, train_size=a.n,
                                          stratify=y_te, random_state=SEED)
    else:
        X_s, y_s = X_te, y_te
    out = X_s.copy()
    out["TARGET"] = y_s.to_numpy()
    out.to_parquet(OUT / "heldout_sample.parquet", index=False, compression="zstd")

    for path in sorted(Path("models").glob("lgbm_*.joblib")):  # sanity check
        missing = set(joblib.load(path)["model"].feature_name_) - set(out.columns)
        if missing:
            print(f"WARNING: {path.name} needs columns absent from the sample, "
                  f"e.g. {sorted(missing)[:3]}")
    for p in sorted(OUT.iterdir()):
        print(f"{p}: {p.stat().st_size / 1e6:.1f} MB")
    print(f"{len(out):,} held-out applicants, {int(out['TARGET'].sum())} defaulters")


if __name__ == "__main__":
    main()
