"""Inference helpers shared by the Streamlit app (app.py) and the CLI (cli.py)."""
import json
import numbers
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import add_ratios, feature_columns, load_set_a


def load_models():
    """{'compliant': {'B': art, 'C': art}, 'research': {...}} for what exists."""
    out = {}
    for s in ("B", "C"):
        for suffix, key in (("_compliant", "compliant"), ("", "research")):
            path = Path(f"models/lgbm_{s}{suffix}.joblib")
            if path.exists():
                out.setdefault(key, {})[s] = joblib.load(path)
    return {k: v for k, v in out.items() if {"B", "C"} <= v.keys()}


DEMO = Path("demo_data")  # committed by src.export_demo


def load_schema():
    """Empty frame with the form's column order, dtypes and category levels.

    Uses the committed demo_data/schema.json when present (fresh clone, no
    dataset needed); otherwise derives it from the processed application table.
    """
    path = DEMO / "schema.json"
    if path.exists():
        spec = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame({
            c: (pd.Categorical([], categories=spec["categories"][c])
                if c in spec["categories"] else pd.Series([], dtype="float32"))
            for c in spec["columns"]})
    df = load_set_a()
    return df[feature_columns(df, "A")].head(1000).copy()


def load_heldout(sample=0):
    """(X, y) of held-out applicants: the committed sample if present, else the full test split."""
    path = DEMO / "heldout_sample.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return df.drop(columns=["TARGET"]), df["TARGET"]
    from src.compare import load_split
    _, X_te, _, y_te = load_split("B", sample)
    return X_te, y_te


def threshold(cost_ratio):
    """Cost-minimising reject threshold for calibrated probabilities."""
    return 1 / (1 + cost_ratio)


def make_raw(income, credit, annuity, goods, age, employed, years, children,
             family, education, income_type, family_status, housing, occupation,
             contract, own_car, car_age, own_realty, gender=None):
    """Form values -> raw application fields (everything else stays missing)."""
    raw = {
        "AMT_INCOME_TOTAL": float(income), "AMT_CREDIT": float(credit),
        "AMT_ANNUITY": float(annuity), "AMT_GOODS_PRICE": float(goods),
        "DAYS_BIRTH": float(-age * 365),
        "DAYS_EMPLOYED": float(-years * 365) if employed else np.nan,
        "DAYS_EMPLOYED_ANOM": 0 if employed else 1,
        "CNT_CHILDREN": children, "CNT_FAM_MEMBERS": family,
        "NAME_EDUCATION_TYPE": education, "NAME_INCOME_TYPE": income_type,
        "NAME_FAMILY_STATUS": family_status, "NAME_HOUSING_TYPE": housing,
        "OCCUPATION_TYPE": (np.nan if occupation in (None, "(not given)")
                            else occupation),
        "NAME_CONTRACT_TYPE": contract, "FLAG_OWN_CAR": own_car,
        "OWN_CAR_AGE": float(car_age) if own_car == "Y" else np.nan,
        "FLAG_OWN_REALTY": own_realty,
    }
    if gender not in (None, "(not given)"):
        raw["CODE_GENDER"] = gender
    return raw


def build_row(schema, raw):
    """One applicant -> a full-width feature row (unasked fields = missing)."""
    cols = list(schema.columns)
    row = pd.DataFrame(np.nan, index=[0], columns=cols)
    for k, v in raw.items():
        row[k] = v
    row = add_ratios(row)  # same code that built the training features
    for c in cols:
        if str(schema[c].dtype) == "category":
            row[c] = pd.Categorical(row[c], categories=schema[c].cat.categories)
    return row


def predict(art, row):
    model = art["model"]
    return float(model.predict_proba(row[model.feature_name_])[0, 1])


def top_contributions(art, row, k=8):
    """Exact TreeSHAP (log-odds) for the k largest drivers: (labels, values)."""
    model = art["model"]
    X = row[model.feature_name_]
    contrib = pd.Series(model.predict(X, pred_contrib=True)[0][:-1],
                        index=model.feature_name_)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:k]
    labels = []
    for f in top.index:
        v = X[f].iloc[0]
        txt = ("missing" if pd.isna(v)
               else f"{v:.3g}" if isinstance(v, numbers.Number) else str(v))
        labels.append(f"{f} = {txt}")
    return labels, top.to_numpy()
