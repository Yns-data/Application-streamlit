# Satisfaction Client


**README**

Le fichier README.md est un élément central de tout repository git. Il permet de présenter votre projet, ses objectifs, ainsi que d'expliquer comment installer et lancer le projet, ou même y contribuer.


**Application Streamlit**

Un template d'application [Streamlit](https://streamlit.io/) est disponible dans le dossier [`streamlit_app`](streamlit_app). Vous pouvez partir de ce template pour mettre en avant votre projet.

## Presentation


This work is part of the general framework NLP (Natural Language Processing) based on the analysis of comments from Internet users.
From a technical point of view, our project called on many skills acquired during the training:
- Regression models
- Classification models
- Semantic analysis with approaches such as bag of words
- Contextual embeddings or named entity recognition (NER).

This repository contains the code for our project SATISFACTION CLIENTS, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is  to extract information from comments. The different areas of work chosen were:

- Prediction of customer satisfaction: classification problem (predict the number of stars) in an exploratory framework.
- Identification of important entities of a message to categorize the information relating to each entity (company, location).
- Extraction and analysis of comments from comments to look for commonalities between the comments of different Internet users, but also to identify the vocabulary    and subjects associated with each note.

This project was developed by the following team :

- Guy BADO  [LinkedIn](https://www.linkedin.com/in/guy-armand-bado-5588698b/)
- Younes ES-SOUALHI [LinkedIn](https://www.linkedin.com/in/younes-essoualhi-3a5a88135/)

You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :

## Installation
```
pip install -r requirements.txt
```

## Streamlit App

**Add explanations on how to use the app.**

To run the app :

```shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).