import streamlit as st
import requests


text_input = st.text_input("Entrez un texte pour générer la suite")
if st.button("Générer"):
    response = requests.post("http://127.0.0.1:8000/generate", json={"prompt": text_input})
    if response.status_code == 200:
        result = response.json()
        st.write(f"Texte généré : {result['generated_text']}")
    else:
        st.write("Erreur lors de la génération")
