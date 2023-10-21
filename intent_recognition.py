from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any

app = FastAPI()

DB_SIZE_LIMIT = 1000  # Example limit

intents_db = {}

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
    return {"state": "success"}

@app.post("/delete_intent")
async def delete_intent(data: Dict[str, Any]):
    machine_name = data.get("machine_name")

    if not machine_name or not isinstance(machine_name, str):
        return {"state": "failure", "detail": "Invalid or missing 'machine_name' field."}

    if machine_name not in intents_db:
        return {"state": "failure", "detail": "Intent not found."}

    del intents_db[machine_name]
    return {"state": "success"}
