# # -*- coding: utf-8 -*-
"""Modelling"""
# # modelling.py

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
    dan log semua artefak ke MLflow.
    """

    # 1. Set tracking ke local MLflow (mlruns)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("student-performance-pass-math")

    # 2. Autolog (wajib log_models=True sesuai saran reviewer)
    mlflow.sklearn.autolog(log_models=True)

    # 3. Load data preprocessing dari parameter MLflow
    df = pd.read_csv(preprocessed_path)

    X = df.drop("pass_math", axis=1)
    y = df["pass_math"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Start MLflow run
    with mlflow.start_run(run_name="rf_baseline"):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("=== MODEL PERFORMANCE ===")
        print("Accuracy:", acc)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Semua metric & model otomatis dilog oleh autolog()


if __name__ == "__main__":
    # Ambil parameter dari MLproject
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    # Jalankan training
    train_model(args.data_path)

