import json
import requests

# Load your data from the JSON file
with open('./data/intents.json', 'r') as file:
    data_points = json.load(file)

# Base URL for the FastAPI application
base_url = "http://127.0.0.1:8000/add_intent"

# Loop over each data point and send a POST request
counter = 0
for data in data_points:
    response = requests.post(base_url, json=data)
    
    if response.status_code == 200 and response.json()["state"] == "success":
        print(f"Successfully added intent: {data['name']}")
    else:
        print(f"Failed to add intent: {data['name']}. Reason: {response.json()['detail']}")

    counter += 1
    if counter % 10 == 0:
        print(f'{counter} data points processed')
