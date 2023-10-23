from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = FastAPI()

DB_SIZE_LIMIT = 1000  # Example limit

intents_db = {}

st_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
st_corpus_embeddings = None  # This will store our intent embeddings

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
    intent_mapping = []
    for intent_name, examples in intents_db.items():
        for example in examples:
            corpus.append(preprocess_text(example))
            intent_mapping.append(intent_name)
    return corpus, intent_mapping

def update_embeddings():
    global st_corpus_embeddings
    corpus, _ = build_corpus_and_mapping()
    st_corpus_embeddings = st_model.encode(corpus)


@app.post("/add_intent")
async def add_intent(data: Dict[str, Any]):
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

    update_embeddings()
    
    return {"state": "success"}

@app.post("/delete_intent")
async def delete_intent(data: Dict[str, Any]):
    machine_name = data.get("machine_name")

    if not machine_name or not isinstance(machine_name, str):
        return {"state": "failure", "detail": "Invalid or missing 'machine_name' field."}

    if machine_name not in intents_db:
        return {"state": "failure", "detail": "Intent not found."}

    del intents_db[machine_name]

    update_embeddings()
    
    return {"state": "success"}


@app.post("/match_intent")
async def get_intent(data: Dict[str, Any]):
    utterance = data.get("utterance")
    
    if not utterance or not isinstance(utterance, str):
        raise HTTPException(status_code=400, detail="Invalid or missing 'utterance' field.")
    
    if not intents_db or st_corpus_embeddings is None:
        raise HTTPException(status_code=400, detail="No intents in the database")
        
    utterance_embedding = st_model.encode(preprocess_text(utterance))
    cosine_scores = util.pytorch_cos_sim(utterance_embedding, st_corpus_embeddings).flatten()
    matched_index = cosine_scores.argmax().item()
    _, intent_mapping = build_corpus_and_mapping()

    return {"intent": intent_mapping[matched_index]}
