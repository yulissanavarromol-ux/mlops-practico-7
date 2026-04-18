from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
import boto3

load_dotenv()

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

    bucket = os.getenv("S3_BUCKET")
    print("Bucket:", bucket)

    if bucket:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )

        s3.upload_file("model/model.pkl", bucket, "models/model.pkl")
        print("Subido a S3: models/model.pkl")

        df.to_csv("iris.csv", index=False)
        s3.upload_file("iris.csv", bucket, "data/iris.csv")
        print("Subido a S3: data/iris.csv")

    print("Proceso completado ✅")

if __name__ == "__main__":
    main()