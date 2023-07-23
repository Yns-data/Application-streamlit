import streamlit as st
import pandas as pd
from PIL import Image
title = "Projet satisfaction clients"
sidebar_name = "Introduction"

def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.markdown("---")

    st.header("Introduction")
    st.markdown(
        """
        Le projet "satisfaction clients" se base sur l’analyse de commentaires d’internautes issus des sites Trustedshop et  Trustpilot. 

        Cette application streamlite propose plusieurs fonctionnalités :
        - la prédiction de la satisfaction d’un client en se basant sur des [modèles de classification] (https://github.com/Yns-data/Notebook-satisfaction-client/blob/main/4.%20Pr%C3%A9diction%20et%20interpr%C3%A9tabilit%C3%A9.ipynb).
        - [l’identification des entités importantes d’un message] (https://github.com/Yns-data/Notebook-satisfaction-client/blob/main/5.NER.ipynb) pour catégoriser les informations relatives à chaque entité (entreprise, localisation).
        - [l'extraction de propos] (https://github.com/Yns-data/Notebook-satisfaction-client/blob/main/6.%20Analyse%20des%20commentaires.ipynb) pour extraire des commentaires similaires ou identifier des réponses à une problématique rencontrée.
        """
    )
    image = Image.open('C:\\Users\\youne\\Desktop\\Application-streamlit\\images\\mots.png')
    st.image(image, caption='Customers words')
    st.header("Presentation du dataframe")
    df=pd.read_csv("C:\\Users\\youne\\Desktop\\Application-streamlit\\dataset and models\\dataset.csv",sep="\t")
    st.markdown("Le fichier contient 19863 enregistrement et 11 variables")
    st.write(df)



