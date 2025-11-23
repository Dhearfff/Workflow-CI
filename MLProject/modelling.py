# -*- coding: utf-8 -*-
"""Modelling"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import mlflow
import mlflow.sklearn

def train_model(preprocessed_path="StudentsPerformance_preprocessing.csv"):

    # 1. Aktifkan autolog (WAJIB UNTUK BASIC)
    mlflow.sklearn.autolog()

    # 2. Mulai run MLflow
    with mlflow.start_run():

        # --- LOAD DATA ---
        df = pd.read_csv(preprocessed_path)

        # --- SPLIT FITUR & LABEL ---
        X = df.drop("pass_math", axis=1)
        y = df["pass_math"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # --- MODEL ---
        model = RandomForestClassifier(random_state=42)

        # --- TRAIN ---
        model.fit(X_train, y_train)

        # --- EVALUASI ---
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("=== MODEL PERFORMANCE ===")
        print(f"Accuracy: {acc}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    print("\nTracking MLflow selesai! Buka MLflow UI untuk melihat hasil run.")


if __name__ == "__main__":
    train_model()
