import streamlit as st
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import time

from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

# NLP pkgs
import spacy
from spacy import displacy
import gensim
from gensim.models import KeyedVectors

# from textblob import TextBlob

def read_pdf(file):
    output_string = StringIO()
    parser = PDFParser(file)
    try: 
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            try: 
                interpreter.process_page(page)
            except:
                print("not able to read the pdf file")
                pass
    except:
        print("not able to read pdf file")
        pass
    return_string = preprocess_pdf(output_string.getvalue())
    return return_string   

def preprocess_pdf(string):
    string_final = ""
    # print(string+"\n")
    bla = string.split("\n")
    for line in bla:
        string_final += line
    return string_final

def text_analyzer(my_text):
    nlp = spacy.load('de_core_news_sm')
    docs =nlp(my_text)
    tokens = [token.text for token in docs]
    # allData = [(f'"Tokens": {token.text},\n"Lemma":{token.lemma_}') for token in docs]
    allData = [(token.text,token.lemma_) for token in docs]
    return allData


def word_to_vector(word):
    nlp = spacy.load('de_core_news_sm')
    doc = nlp(word)
    if len(doc) > 1:
        return doc.vector
    else: 
        return doc[0].vector


def entity_recognizer(my_text):
    nlp = spacy.load("models/ml_rule_model")
    docs = nlp(my_text)
    entities = [(entity.text, entity.label_) for entity in docs.ents]
    return docs, entities

# Entity types in uppercase
colors = {"KULTUR": "#33ff3c", 
          "ERREGER": " #e0ff33 ",
          "MITTEL":"  #98d7f9  ", 
          "ORT":" #a233ff ",
          "ZEIT": " #19c5e8 ",
          "AUFTRETEN": "  #f998d1  ",
          "BBCH_STADIUM":" #d0f998 ",
          "WITTERUNG": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
options = {"ents": ["KULTUR", "ERREGER","MITTEL", "ORT","ZEIT","AUFTRETEN","BBCH_STADIUM","WITTERUNG"], "colors": colors}

def gen_similarity(word):
    model = KeyedVectors.load_word2vec_format("word_vectors/german.model", binary=True)
    results = model.most_similar(positive=[word])
    return results


def search(query):
    df = pd.read_csv("BB_Feldbau_2007_to_2021.tsv", delimiter= "\t", encoding="utf-8")
    nlp = spacy.load("de_core_news_sm")
    text_list = df["text"].values
    tok_text = []
    #Tokenising using SpaCy:
    for doc in nlp.pipe(text_list, disable=["tagger", "parser","ner"]):
        tok = [t.text for t in doc if t.is_alpha]
        tok_text.append(tok)
    # build a BM25 index
    bm25 = BM25Okapi(tok_text)
    tokenized_query = query.split(" ")
    t0 = time.time()

    scores = bm25.get_scores(tokenized_query)
    top_n_scores = np.sort(scores)[::-1][:3]
    # top_n_scores = np.argsort(scores)[::-1][:3]
    results = bm25.get_top_n(tokenized_query, df["file_name"].values, n=3)
    text_results = bm25.get_top_n(tokenized_query, df["text"].values, n=3)

    t1 = time.time()
    print(f'Searched {len(text_list)} records in {round(t1-t0,3) } seconds \n')
    return results,text_results, top_n_scores,len(text_list),round(t1-t0,3)
    

def main():
    """ NLP App with Streamlit"""
    st.title("HortiSem (Horticulture Semantic)")
    st.subheader("Natural Language Processing Demo")

    # st.sidebar.text('')
    # Tokenization
    if st.checkbox("Show Tokens and Lemma"):
        st.subheader("Tokenize Your Text")
        message = st.text_area("Enter Your Text", "")
        if st.button("Analyze"):
            nlp_result = text_analyzer(message)
            df = pd.DataFrame(nlp_result, columns = ["Token", "Lemma"])
            st.dataframe(df)
            # st.json(nlp_result)
    
    # Token2Vec 
    if st.checkbox("Show Word Vector"):
        st.subheader("Get the vector for text")
        text = st.text_area("Enter Your Word or Text", "")
        if st.button("Process"):
            vector = word_to_vector(text)
            df = pd.DataFrame(vector,columns=[text])
            st.dataframe(df.T)

     # Word similarity
    if st.checkbox("Show Word Similarity"):
        st.subheader("Find similar words")
        word = st.text_area("Enter Your Word", "")
        if st.button("Search"):
            nlp_result = gen_similarity(word)
            df = pd.DataFrame(nlp_result,columns =["similar_word", "probability"])
            st.dataframe(df)

    # Named Entity Recognition
    if st.checkbox("Show Named Entities"):
        st.subheader("Extract Entities From Your Text")
        uploaded_file = st.file_uploader('You can upload a pdf file', type=['pdf'])
        # or enter the text
        message = st.text_area("Or enter your text directly here", "")

        if st.button("Extract"):
            if uploaded_file is not None:
                file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
                st.write(file_details)
                text = read_pdf(uploaded_file)
                doc, entities = entity_recognizer(text)
                html = displacy.render(doc, style="ent",options=options)
                st.markdown(html, unsafe_allow_html=True)
            else:
                doc, entities = entity_recognizer(message)
                html = displacy.render(doc, style="ent",options=options)
                st.markdown(html, unsafe_allow_html=True)   
    
    # Search engine
    if st.checkbox("Suche Machine"):
        st.subheader("Search the related document from document stores")
        query = st.text_area("Enter keyword or texts")
        if st.button("Search"):
            results, text_results, scores, n_docs, search_time = search(query)
            df = pd.DataFrame({"Doc Name": results, "Text":text_results, "Scores": scores})
            st.dataframe(df)
            st.markdown(f'Searched {n_docs} records in {search_time } seconds \n')


if __name__ == '__main__':
    main()