import streamlit as st
import pandas as pd
import numpy as np
import scipy
from sentence_transformers import SentenceTransformer
from PIL import Image

title = "Extraction de propos"
sidebar_name = "Extraction de propos"
titre1 = "1. Détéction de commentaires similaire"

def run():
    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        Ce module vous permet d'analyser les commentaires que vous recevez.
        
        2 possibilités : 
         - Identifier des commentaires similaires au cas de figure que vous rencontrez.
         - Rechercher un modèle de réponse à apporter au commentaire que vous avez reçu
        
        """
    )
    option = st.selectbox(
        'Quelle analyse voulez-vous effecter?',
        ('Sélectionnez la requête souhaitée','Recherche de commentaires similaires', 'Recherche de réponses types associées à un commentaire',))
    if option == 'Sélectionnez la requête souhaitée':
        st.write('')
        image = Image.open('C:/Users/youne/Desktop/Application-streamlit/images/eCall.jpg')
        st.image(image, caption='You have a new comment')
    elif option == 'Recherche de commentaires similaires':
        st.write('Votre requête:', option)
        df = pd.read_pickle("C:\\Users\\youne\\Desktop\\Application-streamlit\\dataset and models\\df_transformer.pkl")

        def user_input():
            Commentaire = st.text_input('Insérez votre texte')
            return Commentaire

        commentaire = user_input()

        action = st.button('Executer')
        if action:
            st.session_state.df = df
            st.session_state.df = st.session_state.df.append({'Commentaire': commentaire}, ignore_index=True)
            st.dataframe(st.session_state.df.tail(1))

            def comments_similarity(n):
                model = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')
                df2 = df.copy()
                n_vector = model.encode(n)
                s = df2['embeddings'].apply(lambda x: round((1 - scipy.spatial.distance.cosine(x, n_vector)) * 100, 2))
                df2 = df2.assign(similarity=s)
                return (
                    df2[['Commentaire', 'star', 'similarity']].sort_values('similarity', ascending=False).head(10))

            results = comments_similarity(st.session_state.df.Commentaire.iloc[18801])
            st.table(results)
    elif option == 'Recherche de réponses types associées à un commentaire':
        st.write('Votre requête:', option)
        df = pd.read_pickle("C:\\Users\\youne\\Desktop\\Application-streamlit\\dataset and models\\df_reponse.pkl")

        def user_input():
            Commentaire = st.text_input('Insérez votre texte')
            return Commentaire

        commentaire = user_input()

        action = st.button('Executer')
        if action:
            st.session_state.df = df
            st.session_state.df = st.session_state.df.append({'Commentaire': commentaire}, ignore_index=True)
            st.dataframe(st.session_state.df.tail(1))

            def comments_similarity(n):
                model = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')
                df2 = df.copy()
                n_vector = model.encode(n)
                s = df2['embeddings'].apply(lambda x: round((1 - scipy.spatial.distance.cosine(x, n_vector)) * 100, 2))
                df2 = df2.assign(similarity=s)
                return (
                    df2[['Commentaire', 'reponse', 'similarity']].sort_values('similarity', ascending=False).head(10))

            results = comments_similarity(st.session_state.df.Commentaire.iloc[8037])
            st.table(results)




