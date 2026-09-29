"""
Research Experiments for the defect prediction study.

Experiment 1 (model comparison) lives in train_models.py.
Experiment 2: Feature ablation - drop complexity metrics, measure performance drop.
Experiment 3: Class balancing with SMOTE - measure recall/F1 improvement on the
              minority (defective) class.
"""
import json
import pandas as pd
from train_models import run_pipeline

RESULTS_PATH = "reports/experiment_results.json"


def experiment_feature_ablation(dataset_name="kc1"):
    """Drop McCabe complexity metrics (v(g), ev(g), iv(g)) and Halstead metrics
    (v, l, d, i, e, b, t) separately, see how much each family contributes."""
    baseline, _ = run_pipeline(dataset_name)

    complexity_cols = ["v(g)", "ev(g)", "iv(g)"]
    halstead_cols = ["v", "l", "d", "i", "e", "b", "t", "n"]

    no_complexity, _ = run_pipeline(dataset_name, drop_features=complexity_cols)
    no_halstead, _ = run_pipeline(dataset_name, drop_features=halstead_cols)

    summary = {
        "baseline_all_features": {m: baseline[m]["f1_score"] for m in baseline},
        "without_mccabe_complexity": {m: no_complexity[m]["f1_score"] for m in no_complexity},
        "without_halstead_metrics": {m: no_halstead[m]["f1_score"] for m in no_halstead},
    }
    return summary


def experiment_smote(dataset_name="kc1"):
    """Compare model performance with vs. without SMOTE oversampling of the
    minority (defective) class on the training set."""
    without_smote, _ = run_pipeline(dataset_name, use_smote=False)
    with_smote, _ = run_pipeline(dataset_name, use_smote=True)

    rows = []
    for model in without_smote:
        rows.append({
            "model": model,
            "recall_no_smote": without_smote[model]["recall"],
            "recall_with_smote": with_smote[model]["recall"],
            "f1_no_smote": without_smote[model]["f1_score"],
            "f1_with_smote": with_smote[model]["f1_score"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Experiment 2: Feature Ablation (F1-score by model) ===")
    ablation = experiment_feature_ablation("kc1")
    print(pd.DataFrame(ablation).to_string())

    print("\n=== Experiment 3: SMOTE Class Balancing (recall & F1) ===")
    smote_df = experiment_smote("kc1")
    print(smote_df.to_string(index=False))

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "feature_ablation": ablation,
            "smote_comparison": smote_df.to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nSaved -> {RESULTS_PATH}")
