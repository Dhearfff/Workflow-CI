"""Modelling"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn


def train_model(preprocessed_path):
    """
    Train RandomForest untuk klasifikasi pass_math
    dan log artefak ke MLflow.
    """

    # Tracking harus ke folder aman di GitHub Actions
    mlflow.set_tracking_uri("file:/tmp/mlruns")
    mlflow.set_experiment("student-performance-pass-math")

    # AUTLOG DIMATIKAN karena menyebabkan error di GitHub Actions
    # mlflow.sklearn.autolog(log_models=True)

    df = pd.read_csv(preprocessed_path)

    X = df.drop("pass_math", axis=1)
    y = df["pass_math"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Start Run aman — TIDAK nested
    with mlflow.start_run(run_name="rf_baseline") as run:
        print("RUN ID:", run.info.run_id)

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log manual
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print("=== MODEL PERFORMANCE ===")
        print("Accuracy:", acc)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    train_model(args.data_path)
