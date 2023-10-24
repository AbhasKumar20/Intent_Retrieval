from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import re
import string
import nltk
import faiss
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np

app = FastAPI()

DB_SIZE_LIMIT = 1000  

intents_db = {}
intent_mapping = []

st_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
dimension = st_model.get_sentence_embedding_dimension()

faiss_index = faiss.IndexFlatL2(dimension)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words and words with length <= 2 characters
    tokens = [token for token in tokens if token not in stopwords.words('english') and len(token) > 2]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove non-ASCII characters
    tokens = [token for token in tokens if all(ord(character) < 128 for character in token)]

    # Removing brackets but keeping content
    text = ' '.join(tokens).replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("(", "").replace(")", "")
    
    return text

def build_corpus_and_mapping():
    corpus = []
    intent_mapping.clear()  # Clear existing mapping
    for intent_name, examples in intents_db.items():
        for example in examples:
            corpus.append(example)
            intent_mapping.append(intent_name)
    return corpus, intent_mapping


@app.post("/add_intent")
async def add_intent(data: Dict[str, Any]):

    global faiss_index
    # Validate the input data
    name = data.get("name")
    examples = data.get("examples")

    if not name or not isinstance(name, str):
        return {"state": "failure", "detail": "Invalid or missing 'name' field."}

    if not examples or not isinstance(examples, List) or not all(isinstance(item, str) and item for item in examples):
        return {"state": "failure", "detail": "Invalid or missing 'examples' field."}

    if name in intents_db:
        return {"state": "failure", "detail": "Intent already exists"}

    # Check database size limit
    if len(intents_db) >= DB_SIZE_LIMIT:
        return {"state": "failure", "detail": "Database size limit exceeded"}

    # Update the in-memory database
    intents_db[name] = examples

    new_embeddings = st_model.encode([preprocess_text(ex) for ex in examples])
    faiss_index.add(np.array(new_embeddings))  # Adding new embeddings to FAISS index
    
    build_corpus_and_mapping()  # Update the intent_mapping list
    
    return {"state": "success"}

@app.post("/delete_intent")
async def delete_intent(data: Dict[str, Any]):

    global faiss_index
    machine_name = data.get("machine_name")

    if not machine_name or not isinstance(machine_name, str):
        return {"state": "failure", "detail": "Invalid or missing 'machine_name' field."}

    if machine_name not in intents_db:
        return {"state": "failure", "detail": "Intent not found."}

    del intents_db[machine_name]

    if not intents_db:
        # Resetting the FAISS index if intents_db is empty after deletion
        faiss_index = faiss.IndexFlatL2(dimension)
        _, _ = build_corpus_and_mapping()
        return {"state": "success"}

    # Rebuild the entire FAISS index without the embeddings of the deleted intent
    corpus, _ = build_corpus_and_mapping()
    new_corpus_embeddings = st_model.encode([preprocess_text(ex) for ex in corpus])
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(new_corpus_embeddings))
    
    return {"state": "success"}

@app.post("/match_intent")
async def match_intent(data: Dict[str, Any]):

    global faiss_index
    utterance = data.get("utterance")

    if not utterance or not isinstance(utterance, str):
        return {"state": "failure", "detail": "Invalid or missing 'utterance' field."}

    # Check if there are intents in the database and FAISS index
    if not intents_db or faiss_index.ntotal == 0:
        return {"state": "failure", "detail": "No intents in the database"}

    # Preprocess the utterance and compute its embedding
    utterance_embedding = st_model.encode(preprocess_text(utterance))

    # Search for the most similar embedding in the FAISS index
    _, matched_indices = faiss_index.search(utterance_embedding.reshape(1, dimension), 1)
    matched_index = matched_indices[0][0]

    return {"intent": intent_mapping[matched_index]}


