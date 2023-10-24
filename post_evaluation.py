import time
import json
import requests
# Load your data from the JSON file
with open('./data/utterances.json', 'r') as file:
    data_points = json.load(file)

# Base URL for the FastAPI application
base_url = "http://127.0.0.1:8000/match_intent"

# Loop over each data point and send a POST request
counter = 0

true_intents = []
predicted_intents = []

for data in data_points:

    true_intents.append(data["name"])
    response = requests.post(base_url, json={"utterance":data["utterance"]})
    intent = response.json()['intent']
    predicted_intents.append(intent)

    counter += 1
    if counter % 10 == 0:
        print(f'{counter} data points processed')

    time.sleep(1)

correct_predictions = sum(1 for true, pred in zip(true_intents, predicted_intents) if true == pred)
incorrect_predictions = len(true_intents) - correct_predictions
accuracy = correct_predictions / len(true_intents) * 100

print(f"Post Deployement(Sent-Trans + FAISS):")
print(f"Number of Correct Predictions: {correct_predictions}")
print(f"Number of Incorrect Predictions: {incorrect_predictions}")
print(f"Accuracy: {accuracy:.2f}%")