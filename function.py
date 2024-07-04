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
        # Chargement des données
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
        
        # Vérification des colonnes nécessaires
        required_columns = ['title', 'type', 'rating', 'description', 'duration']
        if not all(column in df.columns for column in required_columns):
            raise HTTPException(status_code=400, detail=f"Le fichier CSV doit contenir les colonnes {required_columns}.")

        # Nettoyage des données
        df = df.dropna(subset=required_columns)
        
        # Création de la colonne 'Appreciation'
        df['Appreciation'] = df['rating'].str.contains('TV-MA|R', case=False).astype(int)
        df['description_length'] = df['description'].apply(len)  # Longueur de la description
        df['num_seasons'] = df['duration'].str.extract(r'(\d+)').fillna(1).astype(int)  # Extraction du nombre de saisons

        # Préparation des données pour l'entraînement
        features = df[['title', 'description', 'num_seasons']]
        X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(features, df['Appreciation'], test_size=0.2, random_state=42)
        
        # Pipeline pour l'entraînement du modèle
        model_appreciation = make_pipeline(
            TfidfVectorizer(stop_words='english'),
            LogisticRegression()
        )
        model_appreciation.fit(X_train_app['title'] + ' ' + X_train_app['description'], y_train_app)  # Utilisation des titres et descriptions comme entrées textuelles
        joblib.dump(model_appreciation, "model/model_appreciation.pkl")

        return {"message": "Modèles de classification entraînés avec succès"}
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


def get_top_10(year):
    try:
        file_path = 'netflix titles.csv'
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
        
        df = df[df['release_year'] == year]

        model_appreciation = joblib.load("model/model_appreciation.pkl")
        model_rentability = joblib.load("model/model_rentability.pkl")

        df['appreciation_score'] = df['title'].apply(lambda x: model_appreciation.predict_proba([x])[0][1])
        df['rentability_score'] = df['title'].apply(lambda x: model_rentability.predict_proba([x])[0][1])

        # Trier les scores avant de les formater en pourcentage
        top_10_appreciated = df.nlargest(10, 'appreciation_score')[['title', 'listed_in', 'release_year', 'appreciation_score']]
        top_10_profitable = df.nlargest(10, 'rentability_score')[['title', 'listed_in', 'release_year', 'rentability_score']]

        # Formater les scores en pourcentage après tri
        top_10_appreciated['appreciation_score'] = top_10_appreciated['appreciation_score'].apply(lambda x: f"{x * 100:.2f}%")
        top_10_profitable['rentability_score'] = top_10_profitable['rentability_score'].apply(lambda x: f"{x * 100:.2f}%")

        return {
            "top_10_appreciated": top_10_appreciated.to_dict(orient='records'),
            "top_10_profitable": top_10_profitable.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des films et séries les plus appréciés et rentables : {str(e)}")

    
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

