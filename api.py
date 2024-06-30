from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from function import train_model_function, generate_text_function, get_columns_function

app = FastAPI()

class TrainData(BaseModel):
    text: list
    label: list

class GenerateData(BaseModel):
    prompt: str
    max_length: int = 50

@app.post("/training")
def train_model(train_data: TrainData):
    return train_model_function(train_data.text, train_data.label)

@app.post("/generate")
def generate_text(generate_data: GenerateData):
    return generate_text_function(generate_data.prompt, generate_data.max_length)

@app.get("/model")
def get_model():
    return {"message": "Modèle : openai-community/gpt2, Tâche : Génération de texte"}


