import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

BASE_DIR = "/opt/ml/processing"
INPUT_DIR = os.path.join(BASE_DIR, "input")
TRAIN_OUTPUT_DIR = os.path.join(BASE_DIR, "train")
VALIDATION_OUTPUT_DIR = os.path.join(BASE_DIR, "validation")
TEST_OUTPUT_DIR = os.path.join(BASE_DIR, "test")

NUMERIC_FEATURES = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_FEATURES = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
TARGET = "income_binary"
PROTECTED_ATTRS = ["race", "sex", "age"]


def load_split(filename):
    return pd.read_csv(os.path.join(INPUT_DIR, filename))


def main():
    train_df = load_split("train.csv")
    val_df = load_split("validation.csv")
    test_df = load_split("test.csv")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    X_train = preprocessor.fit_transform(train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    X_val = preprocessor.transform(val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    X_test = preprocessor.transform(test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])

    y_train = train_df[TARGET].values
    y_val = val_df[TARGET].values
    y_test = test_df[TARGET].values

    train_out = pd.DataFrame(np.column_stack([y_train, X_train]))
    val_out = pd.DataFrame(np.column_stack([y_val, X_val]))

    test_out = pd.DataFrame(np.column_stack([y_test, X_test]))
    for attr in PROTECTED_ATTRS:
        test_out[f"_protected_{attr}"] = test_df[attr].values

    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(VALIDATION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    train_out.to_csv(os.path.join(TRAIN_OUTPUT_DIR, "train.csv"), header=False, index=False)
    val_out.to_csv(os.path.join(VALIDATION_OUTPUT_DIR, "validation.csv"), header=False, index=False)
    test_out.to_csv(os.path.join(TEST_OUTPUT_DIR, "test.csv"), header=False, index=False)

    print("train shape:", train_out.shape)
    print("val shape:", val_out.shape)
    print("test shape:", test_out.shape)


if __name__ == "__main__":
    main()
