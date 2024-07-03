import streamlit as st
import requests
import pandas as pd
import joblib
import os

API_URL = "http://127.0.0.1:8000"

st.title("Application de Prédiction de Films et Séries de la plateforme Netflix")

st.header("Prédiction de l'Appréciation et de la Rentabilité")
title = st.text_input("Entrez le titre du film ou de la série :")

if st.button("Prédire"):
    if title:
        response = requests.post(f"{API_URL}/predict", json={"title": title})
        if response.status_code == 200:
            prediction = response.json()
            st.write(f"Titre : {prediction['title']}")
            st.write(f"Probabilité d'appréciation : {prediction['appreciation_probability']}")
            st.write(f"Probabilité de rentabilité : {prediction['rentability_probability']}")
        elif response.status_code == 404:
            st.error("Ce titre n'est pas présent sur Netflix.")
        else:
            st.error(f"Erreur lors de la prédiction : {response.text}")

st.header("Top 10 Films ou Séries Appréciés et Rentables")
year = st.number_input("Entrez l'année :", min_value=1900, max_value=2100, step=1, value=2020)

if st.button("Afficher Top 10 Appréciés"):
    response = requests.get(f"{API_URL}/top10/appreciated?year={year}")
    if response.status_code == 200:
        data = response.json()
        top_10_appreciated = pd.DataFrame(data["top_10_appreciated"])
        st.table(top_10_appreciated)
    else:
        st.error(f"Erreur lors de la récupération des top 10 appréciés : {response.text}")

if st.button("Afficher Top 10 Rentables"):
    response = requests.get(f"{API_URL}/top10/appreciated?year={year}")
    if response.status_code == 200:
        data = response.json()
        top_10_profitable = pd.DataFrame(data["top_10_profitable"])
        st.table(top_10_profitable)
    else:
        st.error(f"Erreur lors de la récupération des top 10 rentables : {response.text}")

st.header("Télécharger le Modèle Entraîné")

model_choice = st.selectbox("Choisissez le modèle à télécharger :", ["model_appreciation.pkl", "model_rentability.pkl"])

if st.button("Télécharger le Modèle"):
    model_path = f"model/{model_choice}"
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            st.download_button(label="Télécharger", data=file, file_name=model_choice)
    else:
        st.error("Le modèle sélectionné n'existe pas.")

st.sidebar.header("À propos")
st.sidebar.info("Cette application permet d'interagir avec une API pour prédire l'appréciation et la rentabilité des films et séries, afficher les top 10 films ou séries appréciés et rentables, et télécharger les modèles entraînés.")
