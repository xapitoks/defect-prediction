"""
Train and compare ML models for software defect prediction.
Models: Logistic Regression, Random Forest, XGBoost, Neural Network (MLP)
Metrics: Accuracy, Precision, Recall, F1, ROC-AUC (5-fold CV + held-out test)
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)
from xgboost import XGBClassifier

from data_prep import load_dataset

FIG_DIR = "reports/figures"
MODEL_DIR = "models"
RESULTS_PATH = "reports/results.json"


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None,
                                                  class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                  eval_metric="logloss", random_state=42,
                                  scale_pos_weight=1),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                                         random_state=42),
    }


def run_pipeline(dataset_name="kc1", test_size=0.25, random_state=42, use_smote=False,
                  drop_features=None):
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_dataset(dataset_name)
    if drop_features:
        df = df.drop(columns=[c for c in drop_features if c in df.columns])

    X = df.drop(columns=["defects"])
    y = df["defects"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if use_smote:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)

    results = {}
    roc_data = {}
    fitted_models = {}

    for name, model in get_models().items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        probs = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else preds

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_acc = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy").mean()

        metrics = {
            "cv_accuracy_5fold": round(float(cv_acc), 4),
            "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        }
        results[name] = metrics
        fitted_models[name] = model
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_data[name] = (fpr, tpr, metrics["roc_auc"])

        joblib.dump(model, f"{MODEL_DIR}/{name.replace(' ', '_').lower()}_{dataset_name}.joblib")

    joblib.dump(scaler, f"{MODEL_DIR}/scaler_{dataset_name}.joblib")
    joblib.dump(list(X.columns), f"{MODEL_DIR}/features_{dataset_name}.joblib")

    # --- Plots ---
    # ROC curves, all models overlaid
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{dataset_name.upper()}: ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{dataset_name}_roc_curves.png", dpi=130)
    plt.close()

    # Confusion matrices, 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, (name, model) in zip(axes.flat, fitted_models.items()):
        preds = model.predict(X_test_s)
        cm = confusion_matrix(y_test, preds)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Clean", "Defective"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Clean", "Defective"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle(f"{dataset_name.upper()}: Confusion Matrices")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{dataset_name}_confusion_matrices.png", dpi=130)
    plt.close()

    # Feature importance (Random Forest)
    rf = fitted_models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(7, 5))
    importances.head(10).sort_values().plot(kind="barh", color="#55A868")
    plt.title(f"{dataset_name.upper()}: Top 10 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{dataset_name}_feature_importance.png", dpi=130)
    plt.close()

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results, importances


if __name__ == "__main__":
    results, importances = run_pipeline("kc1")
    print(pd.DataFrame(results).T.to_string())
    print("\nTop 5 features (Random Forest importance):")
    print(importances.head(5).to_string())
