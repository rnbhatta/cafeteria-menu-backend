from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import anthropic
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database import db

router = APIRouter(prefix="/api/query", tags=["query"])

SCHEMA = """
Collections in the cafeteria_menu database:

1. menu_items: {
   id: int, name: str, description: str, price: float,
   category: str  (values: breakfast | lunch | snacks | beverages | desserts),
   tags: array of str  (values: vegetarian | vegan | healthy | gluten-free),
   calories: int, rating: float (0-5), popular: bool, emoji: str
}

2. orders: {
   order_id: str,
   items: [{ id: int, name: str, price: float, quantity: int, emoji: str }],
   details: { name: str, email: str, seat: str, notes: str },
   payment: { method: str  (values: card | upi | cash) },
   subtotal: float, tax: float, total: float,
   placed_at: ISODate, estimated_ready: ISODate,
   status: str  (values: pending | ready | completed)
}
"""


class QueryRequest(BaseModel):
    question: str


@router.post("/")
async def natural_language_query(req: QueryRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set in backend/.env")

    try:
        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""You are a MongoDB query generator for a cafeteria menu app.
Given a plain-English question, return ONLY a valid JSON object representing a MongoDB read query.
No markdown fences, no explanation — raw JSON only.

Schema:
{SCHEMA}

Question: {req.question}

Return a JSON object in exactly one of these shapes:

For find:
{{"collection":"menu_items","operation":"find","filter":{{}},"projection":{{"_id":0}},"sort":{{}},"limit":50}}

For count:
{{"collection":"menu_items","operation":"count","filter":{{}}}}

For aggregate:
{{"collection":"menu_items","operation":"aggregate","pipeline":[{{"$project":{{"_id":0}}}}]}}

Rules:
- Only use read operations (find, count, aggregate).
- Always exclude _id.
- Choose the collection that makes sense for the question.
"""

        message = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        print("AI response:", message.content[0].text.strip())
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid Anthropic API key. Update ANTHROPIC_API_KEY in backend/.env")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=503, detail="Could not reach Anthropic API. Check your internet connection.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {str(e)}")

    raw = message.content[0].text.strip()

    # Strip markdown code fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        query_json = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {raw}")

    collection_name = query_json.get("collection")
    if collection_name not in ("menu_items", "orders"):
        raise HTTPException(status_code=400, detail=f"Unknown collection: {collection_name}")

    collection = db[collection_name]
    operation = query_json.get("operation")

    if operation == "find":
        filter_ = query_json.get("filter", {})
        projection = query_json.get("projection", {"_id": 0})
        sort = query_json.get("sort") or {}
        limit = int(query_json.get("limit") or 50)
        cursor = collection.find(filter_, projection)
        if sort:
            cursor = cursor.sort(list(sort.items()))
        cursor = cursor.limit(limit)
        results = [doc async for doc in cursor]

    elif operation == "count":
        filter_ = query_json.get("filter", {})
        count = await collection.count_documents(filter_)
        results = [{"count": count}]

    elif operation == "aggregate":
        pipeline = query_json.get("pipeline", [])
        results = []
        async for doc in collection.aggregate(pipeline):
            doc.pop("_id", None)
            results.append(doc)

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported operation: {operation}")

    return {
        "question": req.question,
        "query": query_json,
        "results": results,
        "count": len(results),
    }
