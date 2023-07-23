from collections import Counter
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from nltk.tokenize.regexp import RegexpTokenizer
from joblib import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# Importation de la base
df=pd.read_csv('C:\\Users\\youne\\Desktop\\Application-streamlit\\dataset and models\\dataset.csv', sep='\t')
# Initialiser un objet vectorizer
vectorizer = CountVectorizer(max_features = 1000)
#importer la liste des mots inutiles:
from nltk.corpus import stopwords
stop_words = set(stopwords.words('french'))
# Initialiser la variable des mots vides

#importer la liste des mots inutiles:
from nltk.corpus import stopwords
stop_words = set(stopwords.words('french'))

# ajout de mots à stop_words
stop_words.update(["a","à","â","abord","afin","ah","ai","aie","ainsi","allaient","allo","allô","allons","après","assez","attendu","au","aucun","aucune","aujourd","aujourd'hui","auquel","aura","auront","aussi","autre","autres","aux","auxquelles","auxquels","avaient","avais","avait","avant","avec","avoir","ayant","b","bah","beaucoup","bien","bigre","boum","bravo","brrr","c","ça","car","ce","ceci","cela","celle","celle-ci","celle-là","celles","celles-ci","celles-là","celui","celui-ci","celui-là","cent","cependant","certain","certaine","certaines","certains","certes","ces","cet","cette","ceux","ceux-ci","ceux-là","chacun","chaque","cher","chère","chères","chers","chez","chiche","chut","ci","cinq","cinquantaine","cinquante","cinquantième","cinquième","clac","clic","combien","comme","comment","compris","concernant","contre","couic","crac","d","da","dans","de","debout","dedans","dehors","delà","depuis","derrière","des","dès","désormais","desquelles","desquels","dessous","dessus","deux","deuxième","deuxièmement","devant","devers","devra","différent","différente","différentes","différents","dire","divers","diverse","diverses","dix","dix-huit","dixième","dix-neuf","dix-sept","doit","doivent","donc","dont","douze","douzième","dring","du","duquel","durant","e","effet","eh","elle","elle-même","elles","elles-mêmes","en","encore","entre","envers","environ","es","ès","est","et","etant","étaient","étais","était","étant","etc","été","etre","être","eu","euh","eux","eux-mêmes","excepté","f","façon","fais","faisaient","faisant","fait","feront","fi","flac","floc","font","g","gens","h","ha","hé","hein","hélas","hem","hep","hi","ho","holà","hop","hormis","hors","hou","houp","hue","hui","huit","huitième","hum","hurrah","i","il","ils","importe","j","je","jusqu","jusque","k","l","la","là","laquelle","las","le","lequel","les","lès","lesquelles","lesquels","leur","leurs","longtemps","lorsque","lui","lui-même","m","ma","maint","mais","malgré","me","même","mêmes","merci","mes","mien","mienne","miennes","miens","mille","mince","moi","moi-même","moins","mon","moyennant","n","na","ne","néanmoins","neuf","neuvième","ni","nombreuses","nombreux","non","nos","notre","nôtre","nôtres","nous","nous-mêmes","nul","o","o|","ô","oh","ohé","olé","ollé","on","ont","onze","onzième","ore","ou","où","ouf","ouias","oust","ouste","outre","p","paf","pan","par","parmi","partant","particulier","particulière","particulièrement","pas","passé","pendant","personne","peu","peut","peuvent","peux","pff","pfft","pfut","pif","plein","plouf","plus","plusieurs","plutôt","pouah","pour","pourquoi","premier","première","premièrement","près","proche","psitt","puisque","q","qu","quand","quant","quanta","quant-à-soi","quarante","quatorze","quatre","quatre-vingt","quatrième","quatrièmement","que","quel","quelconque","quelle","quelles","quelque","quelques","quelqu'un","quels","qui","quiconque","quinze","quoi","quoique","r","revoici","revoilà","rien","s","sa","sacrebleu","sans","sapristi","sauf","se","seize","selon","sept","septième","sera","seront","ses","si","sien","sienne","siennes","siens","sinon","six","sixième","soi","soi-même","soit","soixante","son","sont","sous","stop","suis","suivant","sur","surtout","t","ta","tac","tant","te","té","tel","telle","tellement","telles","tels","tenant","tes","tic","tien","tienne","tiennes","tiens","toc","toi","toi-même","ton","touchant","toujours","tous","tout","toute","toutes","treize","trente","très","trois","troisième","troisièmement","trop","tsoin","tsouin","tu","u","un","une","unes","uns","v","va","vais","vas","vé","vers","via","vif","vifs","vingt","vivat","vive","vives","vlan","voici","voilà","vont","vos","votre","vôtre","vôtres","vous","vous-mêmes","vu","w","x","y","z","zut","alors","aucuns","bon","devrait","dos","droite","début","essai","faites","fois","force","haut","ici","juste","maintenant","mine","mot","nommés","nouveaux","parce","parole","personnes","pièce","plupart","seulement","soyez","sujet","tandis","valeur","voie","voient","état","étions"])
stop_words.update(["donc", "alors","fait","toujour","c'est","j'ai","dit","n'est","qu'il","quand","chez","avoir","vraiment","car","cela","j a i","je","je ","Jai","jai","J","cest","2","3","faire",'1ère2016','2017','2018','2019','2020','2021','2ème','1ère','2016','2018','2019','2020',])

import string
punc_list = string.punctuation     #returns a list of punctuation
stop_words.update(punc_list)
def clean_docs(texts, remove_stopwords=True,longtoken=3):
    """" 
    cette fonction permet de supprimer non seulement les mots de chaque commentaire mais 
    aussi ceux qui ont une longueur inferieur à longtoken et change la casse en minuscule
    """
    
    kk="[a-zA-Zéè+0-9]{"+"{}".format(longtoken)+","+"}"
    tokenizer = RegexpTokenizer(kk)
    
    docs=texts
    stopwords=stop_words
    
    docs_cleaned = []
       
    for elt in docs:
        docS="{}".format(elt).lower()
        ListeMot=docS.split()
        tokenS=[tok for tok in ListeMot]
#         print(tokenS)
        if remove_stopwords :
            tokenS = [tok for tok in tokenS if tok not in stopwords]
            text= ' '.join(tokenS)
            tokenS=tokenizer.tokenize(text)
#             print(tokenS)      
        else:
            text= ' '.join(tokenS)
            tokenS=tokenizer.tokenize(text)
#             print(tokenS)
        doc_clean = ' '.join(tokenS)
        docs_cleaned.append(doc_clean)
           
    return docs_cleaned


title = "Modèles de classification"
sidebar_name = "Modèles de classification"

def run():

    st.title(title)

    st.markdown(
        """
        ## Nettoyage du commentaire
        L'application suivante a pour objectif d'afficher les prédictions de la note d'un commentaire
        en utilisant les modèles que nous avions entrainés durant le projet.
        """
    )
    def user_input():
        Commentaire = st.text_input('Insérez votre texte')
        return Commentaire
    commentaire = user_input()
    st.markdown(
        """
        ## Commentaire nettoyé
        Après le nettoyage, les mots qui auront l'inluence sur la note sont:
        """
    )
    
    df['Commentaire'][0]=commentaire 
    # Application de la fonction pour le netoyage
    feature=pd.DataFrame(clean_docs(df.Commentaire,longtoken=4), columns=["clean_comment"])
    st.write(feature.clean_comment[0])
     
    st.markdown(
        """
        ## Regression Logistique
        Cette partie concerne la prédiction de la note du commentaire que vous avez inserez en utilisant l'algorithme de la régression logistique.
        """
    )    
   
    # Séparer la variable explicative de la variable à prédire
    X = feature.clean_comment
    # Initialiser un objet vectorizer
    vectorizer = CountVectorizer(max_features = 1000)
    # Mettre à jour la valeur de X
    X = vectorizer.fit_transform(X)
    
    #from joblib import load
    
    model_lr = load('C:\\Users\\youne\\Desktop\\Application-streamlit\\dataset and models\\Logistic_linear.joblib')
    
    y_pred_lr = model_lr.predict(X)
    
    st.markdown(
        """
        La note attribuée par le modèle est:
        """
    )
    st.write(y_pred_lr[0])
    st.markdown(
        """
        ## Gradient Boosting Classifier
        Cette partie concerne la prédiction de la note du commentaire que vous avez inserez en utilisant l'algorithme de Gradient Boosting Classifier.
        """
    )

    # from joblib import load

    model_gbc = load(
        'C:\\Users\\youne\\Desktop\\Application-streamlit\\dataset and models\\GradinentBoosting_Classifier.joblib')

    y_pred_gbc = model_gbc.predict(X)

    st.markdown(
        """
        La note attribuée par le modèle est:
        """
    )
    st.write(y_pred_gbc[0])
    
    st.markdown(
        """
        ## Modèle BERT pré-entrainé
        Cette partie concerne la prédiction de la note du commentaire que vous avez inserez en utilisant un modèle pré-entrainé du Deep Learning. 
        """
    )
    #Instantiation du modèle
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model_BERT = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    #Definition de la fonction sentiment_score pour prédire les notes
    def sentiment_score(review):
        tokens = tokenizer.encode(review, return_tensors='pt')
        result = model_BERT(tokens)
        return int(torch.argmax(result.logits))+1
    st.markdown(
        """
        La note attribuée par le modèle est:
        """
    )
    commentaire_Bert =  str(df['Commentaire'][0])
    Note = sentiment_score(commentaire_Bert)
    st.write(Note)
    


    
    






