"""Feature engineering for Home Credit Default Risk.

Run from the repo root:  python -m src.features
Builds feature set A (application_train only) and caches it as Parquet.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
TARGET = "TARGET"
ID = "SK_ID_CURR"
# Dropped in the "compliant" variant. Extend this list if needed.
PROTECTED = ["CODE_GENDER"]


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric dtypes to save memory."""
    for c in df.select_dtypes("float64"):
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64"):
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def load_application() -> pd.DataFrame:
    return downcast(pd.read_csv(RAW / "application_train.csv"))


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Domain ratios (affordability is what lenders look at).

    Shared with app.py so the demo computes features exactly as training did.
    """
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    df["EMPLOYED_AGE_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].clip(lower=1)
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 365243 is a sentinel for "not employed / pensioner", not a real value.
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype("int8")
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Drop a handful of junk categories.
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)
    df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].replace("XNA", np.nan)

    df = add_ratios(df)

    # Strings -> pandas category (LightGBM uses these natively; sklearn
    # models one-hot them later inside a Pipeline).
    # is_string_dtype works on both pandas 2 (object) and pandas 3 (str).
    str_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    for c in str_cols:
        df[c] = df[c].astype("category")

    return downcast(df)


def variant_tag(feature_set: str, compliant: bool = False) -> str:
    """Suffix used in artifact/report names, e.g. "B" or "B_compliant"."""
    return feature_set + ("_compliant" if compliant else "")


def feature_columns(df: pd.DataFrame, feature_set: str = "A",
                    compliant: bool = False) -> list[str]:
    """Column lists per feature set.

    A: all application features
    C: 'thin-file' = no EXT_SOURCE_* and no credit-bureau enquiry columns
    (B = A + aggregated auxiliary tables, added later)
    """
    cols = [c for c in df.columns if c not in (ID, TARGET)]
    if feature_set == "C":
        cols = [c for c in cols
                if not c.startswith("EXT_SOURCE")
                and not c.startswith("AMT_REQ_CREDIT_BUREAU")]
    if compliant:
        cols = [c for c in cols if c not in PROTECTED]
    return cols


def build_set_a() -> pd.DataFrame:
    df = clean_and_engineer(load_application())
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "set_a.parquet", index=False)
    return df


def load_set_a() -> pd.DataFrame:
    path = PROCESSED / "set_a.parquet"
    return pd.read_parquet(path) if path.exists() else build_set_a()


if __name__ == "__main__":
    d = build_set_a()
    print(f"Saved {d.shape[0]:,} rows x {d.shape[1]} cols")
    print(f"Default rate: {d[TARGET].mean():.2%}")
    print(f"Feature set A: {len(feature_columns(d, 'A'))} features, "
          f"C (thin-file): {len(feature_columns(d, 'C'))} features")
