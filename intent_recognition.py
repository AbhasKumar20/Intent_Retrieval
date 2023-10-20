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
