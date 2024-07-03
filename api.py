from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from function import (
    train_model_function,
    generate_text_function,
    get_top_10,
    predict_appreciation_and_rentability
)

app = FastAPI(
    title="Netflix Movies & Series API",
    description="API pour prédire l'appréciation et la rentabilité des films et séries Netflix.",
    version="1.0.0"
)

class TrainFilePath(BaseModel):
    file_path: str

class GenerateData(BaseModel):
    prompt: str
    max_length: int = 50

class PredictData(BaseModel):
    text: str

class PredictTitle(BaseModel):
    title: str

@app.get("/", summary="Bienvenue", description="Message de bienvenue pour l'API")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction de films et séries"}

@app.post("/training", summary="Entraîner le modèle", description="Entraîner le modèle avec le chemin du fichier CSV.")
def train_model(train_file: TrainFilePath):
    try:
        return train_model_function(train_file.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", summary="Générer du texte", description="Générer du texte basé sur un prompt donné.")
def generate_text(generate_data: GenerateData):
    try:
        return generate_text_function(generate_data.prompt, generate_data.max_length)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", summary="Prédire appréciation et rentabilité", description="Prédire l'appréciation et la rentabilité d'un film ou d'une série donné.")
def predict_appreciation_rentability(predict_title: PredictTitle):
    try:
        return predict_appreciation_and_rentability(predict_title.title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/top10/appreciated", summary="Top 10 ", description="Obtenir le top 10 des films et séries les plus appréciés pour une année donnée.")
def top_10(year: int):
    try:
        return get_top_10(year)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
