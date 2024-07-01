import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import joblib
from transformers import pipeline
from fastapi import HTTPException

generator = pipeline("text-generation", model="openai-community/gpt2")


def train_model_function(file_path):
    try:

        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
        
      
        if 'title' not in df.columns or 'type' not in df.columns or 'rating' not in df.columns:
            raise HTTPException(status_code=400, detail="Le fichier CSV doit contenir les colonnes 'title', 'type' et 'rating'.")
        
        
        df = df.dropna(subset=['title', 'type', 'rating'])
        
       
        df['Appreciation'] = (df['rating'].str.contains('TV-MA|R', case=False)).astype(int)  
        X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(df['title'], df['Appreciation'], test_size=0.2, random_state=42)
        model_appreciation = make_pipeline(TfidfVectorizer(), LogisticRegression())
        model_appreciation.fit(X_train_app, y_train_app)
        joblib.dump(model_appreciation, "model/model_appreciation.pkl")


        df['Rentability'] = (df['listed_in'].str.contains('Drama|Action', case=False)).astype(int)  
        X_train_rent, X_test_rent, y_train_rent, y_test_rent = train_test_split(df['title'], df['Rentability'], test_size=0.2, random_state=42)
        model_rentability = make_pipeline(TfidfVectorizer(), LogisticRegression())
        model_rentability.fit(X_train_rent, y_train_rent)
        joblib.dump(model_rentability, "model/model_rentability.pkl")

        return {
            "message": "Modèles de classification entraînés avec succès"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'entraînement des modèles de classification : {str(e)}")
    

def generate_text_function(prompt, max_length):
    if not prompt:
        raise HTTPException(status_code=400, detail="Le prompt pour la génération de texte ne peut pas être vide.")
    
    try:
        result = generator(prompt, max_length=max_length)
        return {"generated_text": result[0]['generated_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération du texte : {str(e)}")


def get_top_10_appreciated(year):
    try:
        file_path = 'netflix titles.csv'
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
        
        df = df[df['release_year'] == year]

        model_appreciation = joblib.load("model/model_appreciation.pkl")

        df['appreciation_score'] = df['title'].apply(lambda x: model_appreciation.predict_proba([x])[0][1])

        top_10_appreciated = df.nlargest(10, 'appreciation_score')

        top_10_appreciated['appreciation_score'] = top_10_appreciated['appreciation_score'].apply(lambda x: f"{x * 100:.2f}%")
        
        return top_10_appreciated[['title', 'listed_in', 'release_year', 'appreciation_score']].to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des films et séries les plus appréciés : {str(e)}")
    
def predict_appreciation_and_rentability(title):
    try:
        model_appreciation = joblib.load("model/model_appreciation.pkl")
        model_rentability = joblib.load("model/model_rentability.pkl")
        
        appreciation_prob = model_appreciation.predict_proba([title])[0][1]
        
        rentability_prob = model_rentability.predict_proba([title])[0][1]
        
        return {
            "title": title,
            "appreciation_probability": f"{appreciation_prob * 100:.2f}%",
            "rentability_probability": f"{rentability_prob * 100:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

