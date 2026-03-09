from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = FastAPI()

with open("scaler_weights.pkl", "rb") as f:
    scaler = pickle.load(f)

model = load_model("model_weights.keras")


class Features(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int
    month: int
    year: int


@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.post("/predict")
def predict(features: Features):
    input_data = np.array([[
        features.bedrooms,
        features.bathrooms,
        features.sqft_living,
        features.sqft_lot,
        features.floors,
        features.waterfront,
        features.view,
        features.condition,
        features.grade,
        features.sqft_above,
        features.sqft_basement,
        features.yr_built,
        features.yr_renovated,
        features.zipcode,
        features.lat,
        features.long,
        features.sqft_living15,
        features.sqft_lot15,
        features.month,
        features.year
    ]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    return {"predicted_price": round(float(prediction[0][0]), 2)}