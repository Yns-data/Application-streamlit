import re

import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from transformers import pipeline
import json




title = "Identification des entités importantes"
sidebar_name = "Identification des entités"



def tag_sentence(text:str):
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    test_entity = list()
    test_word = list()
    test_score = list()
    for ne in nlp(text):
        test_entity.append(ne['entity'])
        test_word.append(ne['word'])
        test_score.append(ne['score'])

    word_tags_new =[(test_word[i], test_entity[i], test_score[i])for i in range(len(test_word))]

    df=pd.DataFrame(word_tags_new, columns=['word', 'tag', 'probability'])
    return df


def convert_df(df: pd.DataFrame):
    return df.to_csv(index=False).encode('utf-8')


def convert_json(df: pd.DataFrame):
    result = df.to_json(orient="index")
    parsed = json.loads(result)
    json_string = json.dumps(parsed)
    return json_string


def run():
    """ NLP Based App with Streamlit """

    # Title
    st.title("Identification des entités importantes")
    st.markdown("""
        	#### Description
        Identification des entités avec un modèle NER
        	""")
    with st.form(key='my_form'):

        x1 = st.text_input(label='Enter a sentence:', max_chars=10000)
        submit_button = st.form_submit_button(label='🏷️ Create tags')

    if submit_button:
        if re.sub('\s+', '', x1) == '':
            st.error('Please enter a non-empty sentence.')

        elif re.match(r'\A\s*\w+\s*\Z', x1):
            st.error("Please enter a sentence with at least one word")

        else:
            st.markdown("### Tagged Sentence")
            st.header("")

            results = tag_sentence(x1)

            cs, c1, c2, c3, cLast = st.columns([0.75, 1.5, 1.5, 1.5, 0.75])

            with c1:
                # csvbutton = download_button(results, "results.csv", "📥 Download .csv")
                csvbutton = st.download_button(label="📥 Download .csv", data=convert_df(results),
                                               file_name="results.csv", mime='text/csv', key='csv')
            with c2:
                # textbutton = download_button(results, "results.txt", "📥 Download .txt")
                textbutton = st.download_button(label="📥 Download .txt", data=convert_df(results),
                                                file_name="results.text", mime='text/plain', key='text')
            with c3:
                # jsonbutton = download_button(results, "results.json", "📥 Download .json")
                jsonbutton = st.download_button(label="📥 Download .json", data=convert_json(results),file_name="results.json", mime='application/json', key='json')

            st.header("")

            c1, c2, c3 = st.columns([1, 3, 1])

            st.table(results)


















