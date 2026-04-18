from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="ML API - Iris")

MODEL_PATH = os.path.join("model", "model.pkl")
model = joblib.load(MODEL_PATH)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "API funcionando correctamente"}

@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}