from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from function import (
    train_model_function,
    generate_text_function,
    get_top_10,
    predict_appreciation_and_rentability
)

app = FastAPI()

class TrainFilePath(BaseModel):
    file_path: str

class GenerateData(BaseModel):
    prompt: str
    max_length: int = 50

class PredictData(BaseModel):
    text: str

class PredictTitle(BaseModel):
    title: str

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction de films et séries"}

@app.post("/training")
def train_model(train_file: TrainFilePath):
    try:
        return train_model_function(train_file.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate_text(generate_data: GenerateData):
    try:
        return generate_text_function(generate_data.prompt, generate_data.max_length)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_appreciation_rentability(predict_title: PredictTitle):
    try:
        return predict_appreciation_and_rentability(predict_title.title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/top10/appreciated")
def top_10(year: int):
    try:
        return get_top_10_appreciated(year)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


