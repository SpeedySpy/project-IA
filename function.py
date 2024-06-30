import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import joblib
from transformers import pipeline

generator = pipeline("text-generation", model="openai-community/gpt2")


def train_model_function(texts, labels):
    if not texts or not labels:
        raise HTTPException(status_code=400, detail="Les données d'entraînement sont invalides : 'text' et 'label' ne peuvent pas être vides.")
    
    if len(texts) != len(labels):
        raise HTTPException(status_code=400, detail="Les données d'entraînement sont invalides : 'text' et 'label' doivent avoir la même longueur.")
    
    try:
        df = pd.DataFrame({"text": texts, "label": labels})
        
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
        
        model = make_pipeline(TfidfVectorizer(), LogisticRegression())
        model.fit(X_train, y_train)
        joblib.dump(model, "model/model.pkl")
        
        return {"message": "Modèle entraîné avec succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'entraînement du modèle : {str(e)}")
    
def generate_text_function(prompt, max_length):
    try:
        result = generator(prompt, max_length=max_length)
        return {"generated_text": result[0]['generated_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_columns_function():
    try:
        file_path = 'Netflix film.csv'
        df = pd.read_csv(file_path)
        
        # Renvoyer les noms des colonnes
        columns = df.columns.tolist()
        return {"columns": columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des colonnes : {str(e)}")