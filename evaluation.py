import os
import json
import tarfile
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = "/opt/ml/processing"
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.tar.gz")
TEST_PATH = os.path.join(BASE_DIR, "test", "test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation")

PROTECTED_COLS = ["_protected_race", "_protected_sex", "_protected_age"]


def disparate_impact_and_spd(y_pred, group_series, privileged_value):
    privileged_mask = group_series == privileged_value
    unprivileged_mask = ~privileged_mask

    if privileged_mask.sum() == 0 or unprivileged_mask.sum() == 0:
        return None, None

    p_privileged = y_pred[privileged_mask].mean()
    p_unprivileged = y_pred[unprivileged_mask].mean()

    di = None if p_privileged == 0 else float(p_unprivileged / p_privileged)
    spd = float(p_unprivileged - p_privileged)
    return di, spd


def main():
    with tarfile.open(MODEL_PATH) as tar:
        tar.extractall(path=".")

    model = xgb.Booster()
    model.load_model("xgboost-model")

    df = pd.read_csv(TEST_PATH, header=None)

    n_protected = len(PROTECTED_COLS)
    protected_df = df.iloc[:, -n_protected:].copy()
    protected_df.columns = PROTECTED_COLS

    feature_df = df.iloc[:, :-n_protected]
    y_true = feature_df.iloc[:, 0].values
    X_test = feature_df.iloc[:, 1:]

    dtest = xgb.DMatrix(X_test.values)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_proba)

    sex_di, sex_spd = disparate_impact_and_spd(y_pred, protected_df["_protected_sex"], "Male")
    race_di, race_spd = disparate_impact_and_spd(y_pred, protected_df["_protected_race"], "White")

    age_median = protected_df["_protected_age"].median()
    age_group = np.where(protected_df["_protected_age"] >= age_median, "older", "younger")
    age_di, age_spd = disparate_impact_and_spd(y_pred, pd.Series(age_group), "older")

    report = {
        "regression_metrics": {
            "accuracy": {"value": float(accuracy), "standard_deviation": "NaN"},
        },
        "classification_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
        },
        "fairness_metrics": {
            "sex": {"disparate_impact": sex_di, "statistical_parity_difference": sex_spd, "privileged_group": "Male"},
            "race": {"disparate_impact": race_di, "statistical_parity_difference": race_spd, "privileged_group": "White"},
            "age": {"disparate_impact": age_di, "statistical_parity_difference": age_spd, "privileged_group": "older (>= median age)"},
        },
        "fairness_gate": {
            "disparate_impact_sex": sex_di,
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
