# -*- coding: utf-8 -*-
"""Modelling"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import mlflow
import mlflow.sklearn

def train_model(preprocessed_path="StudentsPerformance_preprocessing.csv"):

    # PASTIKAN AUTOLOG MATI TOTAL
    mlflow.autolog(disable=True)

    with mlflow.start_run():

        # LOAD DATA
        df = pd.read_csv(preprocessed_path)

        # SPLIT
        X = df.drop("pass_math", axis=1)
        y = df["pass_math"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # MODEL
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # EVALUASI
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("Accuracy:", acc)
        print(classification_report(y_test, y_pred))


        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", acc)

        # INI YANG PALING PENTING:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

    print("Model berhasil disimpan di artifacts/model")

if __name__ == "__main__":
    train_model()
