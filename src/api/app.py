from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/RandomForest.pkl")
features = joblib.load("models/features.pkl")


@app.get("/")
def home():
    return {"message": "IDS ML API is running"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    df = pd.get_dummies(df)

    df = df.reindex(columns=features, fill_value=0)

    prediction = model.predict(df)

    result = "Attack" if prediction[0] == 1 else "Normal"

    return {"prediction": result}