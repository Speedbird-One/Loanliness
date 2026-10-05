"""Feature set B: application + aggregated auxiliary tables.

Run from the repo root:  python -m src.aggregate

Each auxiliary table is one-to-many per client, so it is aggregated down to
one row per SK_ID_CURR (mean/max/min/sum + category frequencies + row count)
and left-joined onto the application table. Every per-table aggregate is
cached to data/processed/agg_<name>.parquet, so the tables can be worked on
(and re-run) independently.
"""
import re

import numpy as np
import pandas as pd

from src.features import (ID, PROCESSED, RAW, clean_and_engineer, downcast,
                          load_application)


def _read(name: str) -> pd.DataFrame:
    return downcast(pd.read_csv(RAW / f"{name}.csv"))


def _one_hot(df: pd.DataFrame):
    """One-hot encode string columns; return (df, list of new columns)."""
    cats = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    before = set(df.columns) - set(cats)
    df = pd.get_dummies(df, columns=cats, dummy_na=True, dtype="int8")
    return df, [c for c in df.columns if c not in before]


def _agg(df, key, prefix, drop=(), num_aggs=("mean", "max", "min", "sum")):
    df, cat_cols = _one_hot(df)
    df = df.drop(columns=[c for c in drop if c in df.columns])
    num_cols = [c for c in df.columns if c not in cat_cols and c != key]
    aggs = {c: list(num_aggs) for c in num_cols}
    aggs.update({c: ["mean"] for c in cat_cols})
    g = df.groupby(key)
    out = g.agg(aggs)
    out.columns = [f"{prefix}_{col}_{fn}".upper() for col, fn in out.columns]
    count = g.size().rename(f"{prefix}_COUNT".upper())
    return pd.concat([out, count], axis=1).reset_index()


def _clean_inf(df, cols):
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------
# One function per table (split these between teammates)
# --------------------------------------------------------------------------
def agg_bureau():
    # bureau_balance is aggregated twice: per bureau record, then per client.
    bb = _agg(_read("bureau_balance"), "SK_ID_BUREAU", "BB",
              num_aggs=("min", "max"))
    bureau = _read("bureau").merge(bb, on="SK_ID_BUREAU", how="left")
    return _agg(bureau, ID, "BUREAU", drop=["SK_ID_BUREAU"])


def agg_previous():
    df = _read("previous_application")
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
              "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        df[c] = df[c].replace(365243, np.nan)  # same sentinel as DAYS_EMPLOYED
    df["APP_CREDIT_PERC"] = df["AMT_APPLICATION"] / df["AMT_CREDIT"]
    _clean_inf(df, ["APP_CREDIT_PERC"])
    return _agg(df, ID, "PREV", drop=["SK_ID_PREV"])


def agg_pos_cash():
    return _agg(_read("POS_CASH_balance"), ID, "POS", drop=["SK_ID_PREV"])


def agg_installments():
    df = _read("installments_payments")
    df["PAYMENT_PERC"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"]
    df["PAYMENT_DIFF"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]
    late = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["DPD"] = late.clip(lower=0)       # days past due
    df["DBD"] = (-late).clip(lower=0)    # days before due
    _clean_inf(df, ["PAYMENT_PERC"])
    return _agg(df, ID, "INST", drop=["SK_ID_PREV"])


def agg_credit_card():
    return _agg(_read("credit_card_balance"), ID, "CC", drop=["SK_ID_PREV"])


TABLES = {
    "bureau": agg_bureau,
    "previous": agg_previous,
    "pos_cash": agg_pos_cash,
    "installments": agg_installments,
    "credit_card": agg_credit_card,
}


def _cached(name, fn):
    path = PROCESSED / f"agg_{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = downcast(fn())
    out.to_parquet(path, index=False)
    return out


def build_set_b() -> pd.DataFrame:
    df = clean_and_engineer(load_application())
    for name, fn in TABLES.items():
        print(f"aggregating {name} ...", flush=True)
        agg = _cached(name, fn)
        df = df.merge(agg, on=ID, how="left")
        del agg

    # No history at all -> count 0 (the absence itself is a signal).
    count_cols = [c for c in df.columns if c.endswith("_COUNT")]
    df[count_cols] = df[count_cols].fillna(0)

    # LightGBM rejects special characters in feature names.
    df.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in df.columns]
    df = downcast(df)
    df.to_parquet(PROCESSED / "set_b.parquet", index=False)
    return df


def load_set_b() -> pd.DataFrame:
    path = PROCESSED / "set_b.parquet"
    return pd.read_parquet(path) if path.exists() else build_set_b()


if __name__ == "__main__":
    d = build_set_b()
    print(f"Saved set B: {d.shape[0]:,} rows x {d.shape[1]} cols")
