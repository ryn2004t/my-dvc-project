from typing import List

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = mlflow.pyfunc.load_model("model")


class PredictRequest(BaseModel):
    features: List[float]  # 64 values (an 8x8 digit image, flattened)


@app.post("/predict")
def predict(req: PredictRequest):
    pred = model.predict(np.array([req.features]))
    return {"prediction": pred.tolist()}