Projet M2 DEV - Groupe 1
- Omma Habiba BIPLOB
- Biraveen SIVAHARAN
- Alex CORCEIRO 


# Machine Learning API Project

Projet développé par le groupe 1 dans le cadre d'un cours d'intelligence artificielle, dont le but est qu'on utilise FastAPI pour créer une API web rapide et performante. Les principales bibliothèques utilisées sont FastAPI, Uvicorn, pandas, joblib, requests et scikit-learn.


## Project Structure

- `api.py`: Le fichier contenant l'application FastAPI.
- `requirements.txt`: Le fichier listant les dépendances nécessaires pour le projet.
- `function.py`: Le fichier contenant les principales fonctions pour le prétraitement des données, l'entraînement du modèle et la réalisation des prédictions.
- `app.py`: L'application Streamlit pour interagir avec l'API.
- `model`: Le répertoire où le modèle entraîné est sauvegardé.
- `netflix titles.csv`: Le jeu de données utilisé pour entraîner le modèle.
- `Notebook.ipynb`: Le notebook contenant votre code préliminaire.


## Installation

### Prérequis

-   Python 3.12 ou version supérieure
-   pip (gestionnaire de paquets Python)

### Étapes d'installation

1.  **Clonez le dépôt du projet :**

   ```
   git clone https://github.com/alexcorceiro/ia-Deep-Learning
   cd ia-Deep-Learning
   ```

2.  **Installez les dépendances :**

   ```
   pip install -r requirements.txt
   ```

### Lancer le serveur

1. **Démarrer FastAPI server :**

   ```
   python -m uvicorn api:app --reload
   ```

   Lien du server démarré : `http://127.0.0.1:8000`.

2. **Accéder à la documenttaion de l'API :**

   Vous pouvez accéder à la documentation interactive de l'API une fois le serveur lancé via : `http://127.0.0.1:8000/docs` 


## Démarrer l'application Streamlit App

1. **Start the Streamlit application:**

   ```
   streamlit run app.py
   ```

### Entraînement du modèle

1. **Téléchargez un fichier CSV** avec votre jeu de données via l'interface Streamlit.
2. **Cliquez sur le bouton "Entraîner le modèle"** pour envoyer les données à l'API et entraîner le modèle.

### Faire des prédictions

1. **Entrez les données** pour la prédiction dans la zone de texte fournie dans l'interface Streamlit.
2. **Cliquez sur le bouton "Prédire"** pour obtenir la prédiction à partir du modèle entraîné.


## Étapes pour contribuer :

1.  Fork le dépôt
2.  Créez votre branche de fonctionnalité (`git checkout -b ia-Deep-Learning`)
3.  Committez vos changements (`git commit -m 'project ia'`)
4.  Poussez vers la branche (`git push main`)
5.  Ouvrez une Pull Request


## Dépendances

-   FastAPI
-   [Uvicorn](https://www.uvicorn.org/)
-   pandas
-   requests
-   scikit-learn
-   [joblib](https://joblib.readthedocs.io/)
