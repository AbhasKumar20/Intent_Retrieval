## Table of Contents
1. [Project Overview](##project-overview)
2. [Setup and Installation](#setup-and-installation)
3. [API Endpoints](#api-endpoints)
4. [Repository Structure and files](#Repository-Structure-and-files)
5. [Testing the Application](#testing-the-application)

---

## **Project Overview**

### Problem Statement

In the realm of customer support, understanding user intentions is vital. Customers communicate their needs, concerns, or feedback through a variety of intents. For instance, an intent like "Book a Ticket" signifies a customer's wish to purchase a flight or train ticket, while "Shipping Inquiry" indicates a request for package status.

However, the nature of these intents can vary based on the business context. While a train booking scenario might revolve around intents such as "Book a Ticket", "PNR Inquiry", "Seat Preference", and "Train Info", an insurance domain might focus on "Claim Inquiry" or "Insurance Info".

Given the vast array of business-specific intents, predefining a comprehensive training dataset for every intent becomes challenging. Instead, our approach allows users to define or create an intent by inputting example utterances. The core objective of this project is to develop a solution that, given a new customer utterance, references these user-defined intent examples to accurately classify the utterance into one of the existing intents.

### Proposed Solution

This solution adopts a multi-faceted approach to address the intent classification challenge:

#### 1. Data Curation using LLaMa-2 13b LLM

An essential aspect of our intent recognition system's success lies in the quality and diversity of the data it's trained and evaluated on. While traditional data gathering approaches might involve manual annotation or the accumulation of real-world examples over time, we leveraged the capabilities of **LLaMa-2 13b LLM (Language Model)** for this purpose.

#### a. Data Augmentation:

- **Generating Unique Intents**: To capture a broad spectrum of potential customer inquiries, we prompted LLaMa-2 13b LLM with a few example intents like 'book-a-ticket' and 'flight-cancellation'. The model, leveraging its vast knowledge, was able to generate a list of 100 unique intents, thus ensuring a wide coverage of potential user queries.

```python
few_shot_prompt =  f"""Generate 35 new, different and diverse intents. Examples of various intents are given in the below list\n " + {str(['flight-cancellation', 'check-reservation', 'book-a-ticket','Shipping-Inquiry',...])} + "\n All intents must be unique.\n Each intent should consists of minimum 2 and maximum 3 words only.\nDont generate same topics which are mentioned in the examples.\nProvide list of new intents in pyhton list format """

```
```python
    New_generated intents = [Travel-Agent-Services','Tourism-Packages','Motorcycle-Rental','Golf-Course-Reservations','Ski-Resort-Bookings', 'Luxury-Car-Rentals',...]
```

- **Curating Examples for Intents**: For each of the 100 intents, we again utilized LLaMa-2 13b LLM's capabilities. By providing a one-shot prompting technique that involved giving the model one example of an intent, it was able to produce a list of relevant examples for that intent. These examples serve as reference points when determining the intent of a new user utterance.


```python
    one_shot_prompt = f"""
    See below dictiona data point which has two keys 'name' and 'examples'. 'name' key represents the intent.
    'examples' key contains examples related to this intent.
    Below is a sample data point for intent {example["name"]} and corresponding examples.
    
    {example}
    
    Create a data point following the exact same format for intent {intent} and always use "" for strings.
    Do not use placeholder for entities.
    """
```
Generated examples sample
```json
    {"name": "Shipping-Inquiry", "examples": ["What is the estimated delivery time for my package that was shipped to [address]?", "I'm waiting for a package that was shipped on [date] but it hasn't arrived yet. Can you help me with tracking information?", "I'm having trouble finding my delivery address for package [package ID]. Can you assist me with this?", "I've received a package that was supposed to be delivered on [date] but it was delivered [days later] instead. Can you help me with this?", "I'm having trouble locating the carrier's website for tracking [package ID]. Can you provide the URL for me?"]}
    
```

#### b. Evaluation:

- **Generating Utterances for Testing**: Post the training phase, it's crucial to evaluate the system's performance on unseen data. For this, we needed utterances that the model hadn't encountered during training. Using a one-shot prompting approach, where LLaMa-2 13b LLM was given a pair of an intent and its example, the model generated utterances relevant to the given intent. These utterances were then stored in `utterances.json`.

```python
one_shot_prompt = f"""
    For below example data point 
    {example}
    corresponding utterance is  = "Inquiring about my reservation status.".
    Provide only 1 generic utterance for the below data point.
    {intent}
    Do not use any placeholder for entities. Always assign proper nouns. 
    Utterance should contain maximum 8 words not more than that. 
    Always Provide answer in a single line in this format only ["utterance" = "your_answer"]                               
    """
```
Generated Utterance sample

```json
{"name": "Ski-Resort-Bookings", "utterance": " \"I'd like to book a ski resort for my family.\""}
```

This approach not only provided us with diverse and high-quality data for training and evaluation but also showcased how advanced language models like LLaMa-2 13b LLM can be instrumental in the data curation process, bypassing traditional, labor-intensive methods.

#### 2. Data Preprocessing Steps

Before embeddings can be generated and used, it's imperative to process raw data to ensure quality embeddings. Our preprocessing pipeline includes:

- **Lowercasing**: To maintain consistency and avoid duplications based on case differences.
  
- **Punctuation Removal**: To strip unnecessary characters and focus on the semantic content of utterances.

- **Tokenization**: Breaking down sentences into individual words or tokens.
  
- **Stopwords Removal**: Removing common words like 'and', 'the', etc., that don't contribute to the semantic meaning.
  
- **Lemmatization**: Converting words to their base or root form (e.g., 'running' to 'run').

This preprocessing ensures that the embeddings generated truly capture the semantic essence of each utterance, optimizing the intent recognition process.

#### 3. Sentence Transformer Integration

**SentenceTransformer** is a library for sentence, paragraph, and image embeddings. We opted for this over other embedding techniques due to its pre-trained models that allow for semantic sentence representations. 

In the context of our application, the `paraphrase-distilroberta-base-v1` model from SentenceTransformer is utilized. This model has been trained to recognize the semantic similarity of sentences, making it ideal for our intent-matching use case. 

#### 4. Fast Nearest Neighbors Search with FAISS

**FAISS (Facebook AI Similarity Search)** is a library for efficient similarity search and clustering of dense vectors. To ensure our application's response times remain low, even as the number of intents grow, we use FAISS. The embeddings from SentenceTransformer are stored in a FAISS index. When an utterance comes in, its embedding is compared against those in the FAISS index to quickly determine the closest matching intent.

#### 5. Incremental Updates:
Recognizing that intents and their examples might evolve over time, I've implemented functionality that allows users to:

- **Add Intents**: Users can define new intents by providing example utterances. The system then computes embeddings for these examples and integrates them into the FAISS index for future searches.
- **Delete Intents**: If an intent becomes redundant, users can remove it. Post-deletion, our system rebuilds the FAISS index to ensure accurate intent detection.

#### 6. Intent Recognition Mechanism

The overarching goal is to match a given utterance to one of the known intents based on semantic similarity. When a user submits an utterance:

1. The utterance undergoes preprocessing (lowercasing, punctuation removal, tokenization, stopwords removal, and lemmatization).
2. The processed utterance is then embedded using the SentenceTransformer.
3. This embedding is matched against the embeddings of the known intents in the FAISS index.
4. The intent with the most semantically similar embedding to the utterance is returned as the match.


#### 7. FastAPI Framework:
Our solution is encapsulated as a RESTful API, developed using the FastAPI framework. This provides flexibility, allowing users to interact with our system via HTTP requests, making it easily integratable with other systems or platforms.


#### **Key Technical Highlights**:

- **LLaMa-2 13B**: Used for knowledge distillation, generative data augmentation
- **SentenceTransformers**: Used for converting text data into meaningful embeddings.
- **FAISS**: Enables efficient similarity search for intent detection.
- **FastAPI**: Powers the backend, ensuring a responsive and scalable solution.


---

## **Setup and Installation**

Follow these steps to get the intent classification system up and running on your local machine or external IP:

### Prerequisites:
- Python 3.7 or higher
- pip (Python package installer)

### 1. **Clone the Repository**:
Clone the project repository to your local machine.
```bash
git clone https://github.com/AbhasKumar20/Intent_Retrieval.git
cd Intent_Retrieval
```

### 2. **Setup a Virtual Environment** (Recommended):
It's advisable to set up a virtual environment to keep dependencies required by different projects separate and to manage them efficiently.
```bash
pip install virtualenv
virtualenv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. **Install Dependencies**:
Install all necessary packages listed in the `requirements.txt`.
```bash
pip install -r requirements.txt
```

### 4. **Start the Application**:

You have two options:

**Option A:** To run the application locally:
```bash
uvicorn app:app --reload
```
This will start the server on `http://127.0.0.1:8000/`. The `--reload` flag enables the server to restart upon changes, which is useful during development.

**Option B:** To run the application on a specific port and make it accessible via an external IP:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8888
```
This starts the server on `http://<external_ip>:8888/`, making it accessible externally.

### 5. **Access the API Documentation**:
With the server running, you can access the API documentation by navigating to:
```
http://127.0.0.1:8000/docs or http://<external_ip>:8888/docs
```
This interactive interface, courtesy of FastAPI, allows you to test different endpoints and understand their usage.

Once set up, users can proceed to use the provided endpoints to add, delete, and match intents.

---

## **API Endpoints**:

This section provides a brief overview of each endpoint, its usage, the expected request format, and the corresponding response format.


#### 1. **Add Intent (`/add_intent`)**

This endpoint is designed to allow users to add a new intent to the system. Each intent consists of a unique name (acting as an identifier) and a list of examples that represent the intent. These examples are textual sentences or phrases that users might utter, and they act as the training data for the system. Once an intent is added, the system computes the embeddings of these examples and adds them to a FAISS index for efficient similarity searching. By utilizing this endpoint, users can dynamically expand the intent recognition capabilities of the system.

- **HTTP Method:** POST

**CURL Command:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/add_intent' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "check-reservation",
  "examples": [
    "What'\''s the reservation status for booking ID ABC123?",
    "Can you provide me with the current boarding status for my flight?",
    "I'\''d like to know if my reservation for flight XYZ456 is confirmed.",
    "What'\''s the seat number allocated for my reservation with confirmation number DEF789?",
    "Has there been any seat upgrade for my booking on flight LMN567?",
    "Can you check the reservation status for the email address john.doe@email.com?",
    "I want to inquire about the boarding status of my flight with booking reference GHI234.",
    "Please tell me the current status of my reservation for flight JKL890.",
    "Has there been any change in the seat assignment for my booking on flight MNO123?",
    "What'\''s the reservation status for my ticket on the flight with departure code PQR456?"
  ]
}'
```

**Response:**
```json
{
  "state": "success"
}
```

#### 2. **Delete Intent (`/delete_intent`)**

The purpose of the delete intent endpoint is to remove a previously added intent from the system. This operation ensures that the intent's associated examples are no longer used for intent recognition. The user provides the machine name (the unique identifier) of the intent they wish to delete. Post deletion, the FAISS index is rebuilt to exclude the embeddings of the deleted intent, ensuring efficient and accurate intent recognition for the remaining intents.

- **HTTP Method:** POST

**CURL Command:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/delete_intent' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "machine_name": "check-reservation"
}'
```

**Response:**
```json
{
  "state": "success"
}
```

#### 3. **Match Intent (`/match_intent`)**

This endpoint's primary function is to identify the most closely related intent for a given utterance. When a user provides an utterance, the system computes its embedding and searches the FAISS index to identify the most similar example from the intents. The recognized intent is then returned in the response. This endpoint is the core of the system, allowing users to determine the intent of any new utterance based on the examples provided during intent creation.

- **HTTP Method:** POST

**CURL Command:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/match_intent' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d "{\"utterance\": \"I'd like to know if I have confirmed seat reservation in flight PNR123\"}"
```

**Response:**
```json
{
  "intent": "check-reservation"
}
```

---

## **Repository Structure and files**

### 1. **`data_generation.ipynb`**:
A Jupyter notebook responsible for the synthetic data generation process using the LLaMA-2 13B LLM. This notebook can be broken down into three distinct sections:

- **Intent Generation**: Uses few-shot prompting to create a list of 100 unique intents. A few examples of intents are initially provided (like book-a-ticket, flight-cancellation), and the model is tasked with generating a diverse set of 100 unique intents.
  
- **Examples Generation**: For each of the 100 unique intents, examples are generated using a one-shot prompting technique. One example of an intent and its corresponding sample utterances is provided to the model, which in turn generates a list of examples for the intent.

- **Utterances Generation**: Given a specific intent, the model generates potential utterances that a user might express. This is achieved via one-shot prompting where one (intent, utterance) pair is supplied to guide the model.

### 2. **`data/`**: 
The generated (intent, example) pairs are saved in `intents.json` and the (intent, utterance) pairs in `utterances.json` are stored inside the "data/" folder

### 3. **`data_injector.py`**:
A Python script designed to populate the backend database with intent examples. It reads the (intent, examples) pairs from the `intents.json` file and sends them sequentially to the `add_intent` endpoint of the backend, effectively initializing the intent database with examples.

### 4. **`app.py`**:
This is the main FastAPI application file, encapsulating the backend logic for intent recognition. It provides endpoints to add intents, delete intents, and match utterances to their closest intent. The application uses Sentence Transformers for embedding generation and FAISS for efficient similarity-based intent retrieval.

### 5. **`semantic_search_modeling.ipynb`**:
This notebook showcases the initial approach to intent recognition using both TF-IDF and Sentence Transformers. It contains an evaluation of both these methods on a dataset to compare their performance in recognizing intents based on semantic meaning.

### 6. **`post_evaluation.py`**:
A Python script dedicated to evaluating the final performance of the intent recognition system (build with Sentence Transformer + FAISS) post-deployment. It sends each utterance from the `utterances.json` file to the `match_intent` endpoint, captures the returned intent (predicted intent), and evaluates the system's accuracy by comparing the predicted intents to the true intents.

* **Note** requirement.txt contains all the requirements for running the application (**app.py**) using standard uvicorn command and python scripts(`data_injector.py` and `post_evaluation.py` and notebook and `semantic_search_modeling.ipynb`) which is enough for the assignment. To reproduce the notebooks which perform data generation using LLaMa-2(`data_generation.ipynb`) one have to run cell-by-cell to install dependencies on the go.  
---

## **Testing the Application**

The performance of our Intent Retrieval application can be gauged both before deployment (pre-deployment) and after deployment (post-deployment). Here's a step-by-step guide on how to test the model's performance during these two phases:

#### **Pre-Deployment Testing:**

1. **Semantic Search Notebook (`semantic_search_modeling.ipynb`):** 
   
   Before deploying your model, it's essential to ensure its accuracy and reliability. The `semantic_search_modeling.ipynb` notebook is specifically designed to help you evaluate the potential performance of your intent recognition system.

   - **TF-IDF and Sentence Transformers:** This notebook contains the implementation and evaluation of two primary semantic search methods: TF-IDF and Sentence Transformers. 
   
   - **Steps:** To test the model's performance:
     - a. Navigate to the location of `Semantic_search_modeling.ipynb`.
     - b. Open the notebook and run all the cells sequentially.
     - c. Observe the evaluation metrics displayed for both methods to understand how each might perform in a real-world scenario. These metrics will provide an initial insight into the model's accuracy, precision, recall, etc.

#### **Post-Deployment Testing:**

1. **Injecting Intents and Examples:** 

   Once the application is deployed, the first step is to populate the intent database with relevant examples. Use the `Data_injector.py` script to do this.

   - **Steps:** 
     - a. Ensure that the `intents.json` (containing intent and example pairs) is present in the '/data' folder.
     - b. Run the ```bash $python3 data_injecter.py ``` script. This script sends the intent-example pairs from `intents.json` to the `add_intent` endpoint of the deployed application, thereby populating the intent database.

2. **Evaluating Post-Deployment Performance:** 

   After populating the database, it's crucial to test the application's real-world performance using the `Post_Evaluation.py` script.

   - **Steps:** 
     - a. Run the ```bash $python3 data_injecter.py ``` script. This script:
        - i. Sends utterances (from `utterances.json`) to the `match_intent` endpoint.
        - ii. Collects the predicted intent for each utterance.
     - b. The script then evaluates the performance of the deployed model by comparing the predicted intents with the true intents. 

Following the above steps for both pre-deployment and post-deployment testing ensures a comprehensive evaluation of the application, guaranteeing robust and reliable performance in live scenarios.

The below table shows the intent retrieval accuracies of TF-IDF, sentence-transformer, and sentence-transformer + FIASS models

| Method                       | Correct Predictions | Incorrect Predictions | Accuracy  |
|------------------------------|---------------------|-----------------------|-----------|
| Pre-deployment (TF-IDF)      | 87                  | 7                     | 92.55%    |
| Pre-deployment (Sent-Trans)  | 90                  | 4                     | 95.74%    |
| Post-deployment (Sent-Trans + FAISS) | 90         | 4                     | 95.74%    |
