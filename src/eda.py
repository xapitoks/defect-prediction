"""
Exploratory Data Analysis for the defect prediction dataset.
Produces: class balance plot, correlation heatmap, LOC-vs-defect boxplot.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_prep import load_dataset

sns.set_style("whitegrid")
FIG_DIR = "reports/figures"


def run_eda(name: str = "kc1"):
    df = load_dataset(name)
    import os
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Class balance
    plt.figure(figsize=(5, 4))
    counts = df["defects"].value_counts().sort_index()
    plt.bar(["Clean (0)", "Defective (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    plt.title(f"{name.upper()}: Class Balance")
    plt.ylabel("Number of modules")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 5, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}_class_balance.png", dpi=130)
    plt.close()

    # 2. Correlation heatmap (top correlated features with target)
    corr = df.corr(numeric_only=True)["defects"].drop("defects").sort_values(key=abs, ascending=False)
    plt.figure(figsize=(6, 6))
    sns.heatmap(df[corr.index[:10].tolist() + ["defects"]].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", square=True, cbar_kws={"shrink": 0.7})
    plt.title(f"{name.upper()}: Correlation (Top 10 features vs defects)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}_correlation_heatmap.png", dpi=130)
    plt.close()

    # 3. LOC vs defect boxplot (loc is present in all PROMISE sets)
    if "loc" in df.columns:
        plt.figure(figsize=(5, 4))
        sns.boxplot(x="defects", y="loc", data=df[df["loc"] < df["loc"].quantile(0.95)])
        plt.xticks([0, 1], ["Clean", "Defective"])
        plt.title(f"{name.upper()}: Lines of Code vs Defect Status")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/{name}_loc_vs_defect.png", dpi=130)
        plt.close()

    return corr


if __name__ == "__main__":
    corr = run_eda("kc1")
    print("Top features correlated with defects (KC1):")
    print(corr.head(10).to_string())
