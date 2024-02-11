import streamlit as st
import PyPDF2
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import requests

def extract_text_from_pdf(uploaded_file):
    text = ''
    reader = PyPDF2.PdfReader(uploaded_file)
    num_pages = len(reader.pages)
    for page_num in range(num_pages):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text



def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation using regular expression
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Join the tokens back into a single string
    cleaned_text = ' '.join(filtered_tokens)

    return cleaned_text

def main():
    st.title("PDF Text Extraction and Cleaning")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        raw_text = extract_text_from_pdf(uploaded_file)
        # print(raw_text)
        cleaned_text = clean_text(raw_text)
        HUGGING_FACE_API_KEY = st.secrets["huggingface"]["api_key"]
        API_URL = "https://api-inference.huggingface.co/models/d4data/biomedical-ner-all"
        headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
        # Your API call and output retrieval
        output = query(cleaned_text)

        st.header("Raw Text:")
        st.text_area("Raw Text", raw_text, height=200)

        st.header("Cleaned Text:")
        st.text_area("Cleaned Text", cleaned_text, height=200)

        st.header("Ner Text:")
        # ner_entities = [{"entity_group": entity["entity_group"], "word": entity["word"]} for entity in output]
        # st.text_area("NER Text", ner_entities, height=200)
        desired_entity_groups = ["Sign_symptom"]
        filtered_entities = [entity for entity in output if entity['entity_group'] in desired_entity_groups]
        sign_entities = [{"word": entity["word"]}for entity in filtered_entities]
        st.text_area("NER Text", sign_entities, height=200) 
        st.header("Words in New Lines:")
        s = ''
        for entity in sign_entities:
            s += (entity["word"] + ' ')
            st.text(entity["word"])
        
        # new_output = query(s)

        # st.header("NER Text from Sign Entities:")
        # st.text_area("NER Text", new_output, height=200)

        # filtered_entities2 = [entity for entity in new_output if entity['entity_group'] in desired_entity_groups]
        # sign_entities2 = [{"word": entity["word"]}for entity in filtered_entities2]

        # st.header("Words in New Lines from Sign Entities2:")
        # for entity in sign_entities2:
        #     st.text(entity["word"])

if __name__ == "__main__":
    main()
