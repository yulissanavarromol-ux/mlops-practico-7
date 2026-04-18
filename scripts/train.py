from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os

def main():
    print("Cargando dataset Iris...")

    iris = load_iris(as_frame=True)
    df = iris.frame

    X = df.drop(columns=["target"])
    y = df["target"]

    print("Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Entrenando modelo...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("Evaluando modelo...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc}")

    print("Guardando modelo...")
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")

    print("Proceso completado ✅")

if __name__ == "__main__":
    main()