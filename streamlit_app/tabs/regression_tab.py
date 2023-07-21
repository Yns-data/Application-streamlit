import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import emoji
# import gensim

title = "Modèles de regression"
sidebar_name = "Modèles de regression"


# @st.cache(persist=True)
def run():
    st.title(title)
    st.write("------------")
    st.markdown(
        """
            Dans cette partie nous avons trois modèles de regression candidats.
            Il s'agit de:
            >* `RandomForestRegressor`
            >* `GradientBoostingRegressor`
            >* `RidgeCV`
            Il s'agit de predire la note du client à partir de son commentaire.
        """)

     # Importation de la base
    df = pd.read_csv('C:\\Users\younes.essoualhi\\Documents\\satisfaction-client\\streamlit_app\\tabs\\dataset.csv', sep='\t')

    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('french'))

    # ajout de mots à stop_words
    stop_words.update(
        ["a", "à", "â", "abord", "afin", "ah","ait","ais", "ai", "aie", "ainsi", "allaient", "allo", "allô", "allons", "après",
         "assez", "attendu", "au", "aucun", "aucune", "aujourd", "aujourd'hui", "auquel", "aura", "auront", "aussi",
         "autre", "autres", "aux", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avoir",
         "ayant", "b", "bah", "beaucoup", "bien", "bigre", "boum", "bravo", "brrr", "c", "ça", "car", "ce", "ceci",
         "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là",
         "cent", "cependant", "certain", "certaine", "certaines", "certains", "certes", "ces", "cet", "cette", "ceux",
         "ceux-ci", "ceux-là", "chacun", "chaque", "cher", "chère", "chères", "chers", "chez", "chiche", "chut", "ci",
         "cinq", "cinquantaine", "cinquante", "cinquantième", "cinquième", "clac", "clic", "combien", "comme",
         "comment", "compris", "concernant", "contre", "couic", "crac", "d", "da", "dans", "de", "debout", "dedans",
         "dehors", "delà", "depuis", "derrière", "des", "dès", "désormais", "desquelles", "desquels", "dessous",
         "dessus", "deux", "deuxième", "deuxièmement", "devant", "devers", "devra", "différent", "différente",
         "différentes", "différents", "dire", "divers", "diverse", "diverses", "dix", "dix-huit", "dixième", "dix-neuf",
         "dix-sept", "doit", "doivent", "donc", "dont", "douze", "douzième", "dring", "du", "duquel", "durant", "e",
         "effet", "eh", "elle", "elle-même", "elles", "elles-mêmes", "en", "encore", "entre", "envers", "environ", "es",
         "ès", "est", "et", "etant", "étaient", "étais", "était", "étant", "etc", "été", "etre", "être", "eu", "euh",
         "eux", "eux-mêmes", "excepté", "f", "façon", "fais", "faisaient", "faisant", "fait", "feront", "fi", "flac",
         "floc", "font", "g", "gens", "h", "ha", "hé", "hein", "hélas", "hem", "hep", "hi", "ho", "holà", "hop",
         "hormis", "hors", "hou", "houp", "hue", "hui", "huit", "huitième", "hum", "hurrah", "i", "il", "ils",
         "importe", "j", "je", "jusqu", "jusque", "k", "l", "la", "là", "laquelle", "las", "le", "lequel", "les", "lès",
         "lesquelles", "lesquels", "leur", "leurs", "longtemps", "lorsque", "lui", "lui-même", "m", "ma", "maint",
         "mais", "malgré", "me", "même", "mêmes", "merci", "mes", "mien", "mienne", "miennes", "miens", "mille",
         "mince", "moi", "moi-même", "moins", "mon", "moyennant", "n", "na", "ne", "néanmoins", "neuf", "neuvième",
         "ni", "nombreuses", "nombreux", "non", "nos", "notre", "nôtre", "nôtres", "nous", "nous-mêmes", "nul", "o",
         "o|", "ô", "oh", "ohé", "olé", "ollé", "on", "ont", "onze", "onzième", "ore", "ou", "où", "ouf", "ouias",
         "oust", "ouste", "outre", "p", "paf", "pan", "par", "parmi", "partant", "particulier", "particulière",
         "particulièrement", "pas", "passé", "pendant", "personne", "peu", "peut", "peuvent", "peux", "pff", "pfft",
         "pfut", "pif", "plein", "plouf", "plus", "plusieurs", "plutôt", "pouah", "pour", "pourquoi", "premier",
         "première", "premièrement", "près", "proche", "psitt", "puisque", "q", "qu", "quand", "quant", "quanta",
         "quant-à-soi", "quarante", "quatorze", "quatre", "quatre-vingt", "quatrième", "quatrièmement", "que", "quel",
         "quelconque", "quelle", "quelles", "quelque", "quelques", "quelqu'un", "quels", "qui", "quiconque", "quinze",
         "quoi", "quoique", "r", "revoici", "revoilà", "rien", "s", "sa", "sacrebleu", "sans", "sapristi", "sauf", "se",
         "seize", "selon", "sept", "septième", "sera", "seront", "ses", "si", "sien", "sienne", "siennes", "siens",
         "sinon", "six", "sixième", "soi", "soi-même", "soit", "soixante", "son", "sont", "sous", "stop", "suis",
         "suivant", "sur", "surtout", "t", "ta", "tac", "tant", "te", "té", "tel", "telle", "tellement", "telles",
         "tels", "tenant", "tes", "tic", "tien", "tienne", "tiennes", "tiens", "toc", "toi", "toi-même", "ton",
         "touchant", "toujours", "tous", "tout", "toute", "toutes", "treize", "trente", "très", "trois", "troisième",
         "troisièmement", "trop", "tsoin", "tsouin", "tu", "u", "un", "une", "unes", "uns", "v", "va", "vais", "vas",
         "vé", "vers", "via", "vif", "vifs", "vingt", "vivat", "vive", "vives", "vlan", "voici", "voilà", "vont", "vos",
         "votre", "vôtre", "vôtres", "vous", "vous-mêmes", "vu", "w", "x", "y", "z", "zut", "alors", "aucuns", "bon",
         "devrait", "dos", "droite", "début", "essai", "faites", "fois", "force", "haut", "ici", "juste", "maintenant",
         "mine", "mot", "nommés", "nouveaux", "parce", "parole", "personnes", "pièce", "plupart", "seulement", "soyez",
         "sujet", "tandis", "valeur", "voie", "voient", "état", "étions"])
    stop_words.update(
        ["donc","1er","1ere","1ère","1moi","1mois","2eme","2ème","3ème", "alors", "fait", "toujour", "c'est", "j'ai", "dit", "n'est", "qu'il", "quand", "chez", "avoir",
         "vraiment", "car", "cela", "j a i", "je", "je ", "Jai", "jai", "J", "cest", "2", "3", "faire"])

    import string
    punc_list = string.punctuation  # returns a list of punctuation
    stop_words.update(punc_list)

    from nltk.tokenize.regexp import RegexpTokenizer

    #######################################################
    ######### Defintion de fonction netoyage##########
    #######################################################

    def clean_docs(texts, remove_stopwords=True, longtoken=3):
        """"
        cette fonction permet de supprimer non seulement les mots de chaque commentaire mais
        aussi ceux qui ont une longueur inferieur à longtoken et change la casse en minuscule
        """

        kk = "[a-zA-Zéè+0-9]{" + "{}".format(longtoken) + "," + "}"
        tokenizer = RegexpTokenizer(kk)

        docs = texts
        stopwords = stop_words

        docs_cleaned = []

        for elt in docs:
            docS = "{}".format(elt).lower()
            ListeMot = docS.split()
            tokenS = [tok for tok in ListeMot]
            #         print(tokenS)
            if remove_stopwords:
                tokenS = [tok for tok in tokenS if tok not in stopwords]
                text = ' '.join(tokenS)
                tokenS = tokenizer.tokenize(text)
            #             print(tokenS)
            else:
                text = ' '.join(tokenS)
                tokenS = tokenizer.tokenize(text)
            #             print(tokenS)
            doc_clean = ' '.join(tokenS)
            docs_cleaned.append(doc_clean)

        return docs_cleaned

    ## Definition de feutures

    feature = pd.DataFrame(clean_docs(df.Commentaire),
                           columns=["clean_comment"])

    import wordcloud
    from wordcloud import WordCloud

    text = ""
    for comment in feature.clean_comment:
        text += comment

    wc = WordCloud(background_color='black',
                   max_words=2000,
                   stopwords=stop_words,
                   max_font_size=50,
                   random_state=42)

    wc.generate(text)  # "Calcul" du wordcloud

    #########################################
    ####### Vectorisation ###################
    #########################################

    MesMots = wc.words_.keys()  # reperage des 500 top mots

    vectoR = CountVectorizer(ngram_range=(1, 1), max_features=5000)  # les groupes de deux 2 mots
    vectoR.fit(MesMots)  # entrainement sur les 500 top mots, pour le reconnaissance
    lst = sorted(vectoR.vocabulary_.items(), key=lambda t: t[1])  ## le dictionnaire ordonné en focntion des valeurs (sinon t[0] clé)
    Mots_Colonne = [x[0] for x in lst]  ## recupperation des clés uniquement
    features = vectoR.transform(feature.clean_comment)
    commentaire_vectorize = pd.DataFrame(features.toarray(), columns=Mots_Colonne)


    ###########################################
    ######## Separation des données  ##########
    ###########################################
    X_train, X_test, y_train, y_test = train_test_split(commentaire_vectorize, df['star'], test_size=0.25,
                                                        random_state=123)

    #########################################
    ######### Regression ####################
    #########################################
    ##---------- Entrainement
    Mdel=GradientBoostingRegressor(max_depth= 9, max_features="sqrt", n_estimators=50)
    Mdel.fit(X_train, y_train)

    # y_pred_reg_test = Mdel.predict(X_test)  # ligne test de prediction



    ###--------- Prediction des notes selon le commentaire
    txt=st.text_input("Entrer le commentaire", max_chars=2500)
    c1,c2,c3,c4,c5,c6,c7=st.columns(7)
    with c7:
        st.button("Valider")

    txt=vectoR.transform(["{}".format(txt)])

    Pred=Mdel.predict(txt)
    Pred=int(np.round(Pred))

    co1,co2,co3=st.columns(3)
    with co2:
        st.write("-------------------")
        st.write("**Nombre d'étoiles estimé**")

        if (Pred<2):
            st.write(emoji.emojize(":star:"))
        elif (Pred==2):
            st.write(emoji.emojize(":star:"),
                     emoji.emojize(":star:"))
        elif (Pred==3):
            st.write(emoji.emojize(":star:"),
                     emoji.emojize(":star:"),
                     emoji.emojize(":star:"))
        elif (Pred==4):
            st.write(emoji.emojize(":star:"),
                     emoji.emojize(":star:"),
                     emoji.emojize(":star:"),
                     emoji.emojize(":star:"))
        else :
            st.write(emoji.emojize(":star:"),
                     emoji.emojize(":star:"),
                     emoji.emojize(":star:"),
                     emoji.emojize(":star:"),
                     emoji.emojize(":star:"))

        st.write('--------------------------')
