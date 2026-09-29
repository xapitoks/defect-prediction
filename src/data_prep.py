"""
Data loading & cleaning for the Software Defect Prediction project.
Dataset: NASA PROMISE repository (KC1 by default).
"""
import pandas as pd
import numpy as np

DATASETS = {
    "kc1": "data/kc1.csv",
    "jm1": "data/jm1.csv",
    "cm1": "data/cm1.csv",
    "kc2": "data/kc2.csv",
    "pc1": "data/pc1.csv",
}

TARGET_COL = "defects"


def load_dataset(name: str = "kc1") -> pd.DataFrame:
    path = DATASETS[name]
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # the label column name differs slightly across PROMISE files
    label_candidates = [c for c in df.columns if c.lower() in ("defects", "defect", "problems", "class")]
    label_col = label_candidates[0]
    df = df.rename(columns={label_col: "defects"})

    # normalize label to 0/1
    df["defects"] = df["defects"].astype(str).str.strip().str.lower()
    df["defects"] = df["defects"].map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
    df = df.dropna(subset=["defects"])
    df["defects"] = df["defects"].astype(int)

    # coerce all feature columns to numeric, drop rows that fail entirely
    feature_cols = [c for c in df.columns if c != "defects"]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop columns that are entirely NaN, then impute remaining NaNs with median
    df = df.dropna(axis=1, how="all")
    feature_cols = [c for c in df.columns if c != "defects"]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df.reset_index(drop=True)


if __name__ == "__main__":
    for name in DATASETS:
        df = load_dataset(name)
        n_defective = df["defects"].sum()
        print(f"{name.upper():5s} shape={df.shape}  defective={n_defective} "
              f"({100*n_defective/len(df):.1f}%)")
