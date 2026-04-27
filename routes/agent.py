from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import anthropic
import json
import os
import sys
from typing import List
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database import db

router = APIRouter(prefix="/api/agent", tags=["agent"])

TOOLS = [
    {
        "name": "search_menu",
        "description": "Search menu items with optional filters. Use this to find food that matches the user's request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category",
                    "enum": ["breakfast", "lunch", "snacks", "beverages", "desserts"],
                },
                "tag": {
                    "type": "string",
                    "description": "Filter by dietary tag",
                    "enum": ["vegetarian", "vegan", "healthy", "gluten-free"],
                },
                "max_price": {"type": "number", "description": "Maximum price"},
                "min_rating": {"type": "number", "description": "Minimum rating (0-5)"},
                "search": {"type": "string", "description": "Text search in item name"},
            },
        },
    },
    {
        "name": "get_popular_items",
        "description": "Get the most popular and highest-rated menu items.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of items (default 5)"},
            },
        },
    },
    {
        "name": "get_item_details",
        "description": "Get full details about a specific menu item by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the menu item"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_order_status",
        "description": "Look up the status and details of a placed order by order ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID to look up"},
            },
            "required": ["order_id"],
        },
    },
]

SYSTEM_PROMPT = """You are a friendly cafeteria assistant for an office cafeteria. You help employees:
- Browse the menu and find items they'll enjoy
- Answer questions about dietary options, calories, and prices
- Make personalized recommendations based on preferences and budget
- Check the status of placed orders

Keep replies concise and friendly. Format prices as $X.XX. When listing items, include the emoji, name, price, and a brief note.
If someone wants to place an order, let them know they can add items to cart using the menu on the left and check out from there."""


class Message(BaseModel):
    role: str
    content: str


class AgentRequest(BaseModel):
    messages: List[Message]


async def run_tool(tool_name: str, tool_input: dict) -> str:
    try:
        if tool_name == "search_menu":
            query = {}
            if tool_input.get("category"):
                query["category"] = tool_input["category"]
            if tool_input.get("tag"):
                query["tags"] = {"$in": [tool_input["tag"]]}
            if tool_input.get("max_price") is not None:
                query["price"] = {"$lte": float(tool_input["max_price"])}
            if tool_input.get("min_rating") is not None:
                query["rating"] = {"$gte": float(tool_input["min_rating"])}
            if tool_input.get("search"):
                query["name"] = {"$regex": tool_input["search"], "$options": "i"}

            items = []
            async for item in db.menu_items.find(query, {"_id": 0}).limit(20):
                items.append(item)

            if not items:
                return "No menu items found matching those criteria."
            return json.dumps(items)

        elif tool_name == "get_popular_items":
            limit = int(tool_input.get("limit", 5))
            items = []
            async for item in db.menu_items.find({"popular": True}, {"_id": 0}).sort("rating", -1).limit(limit):
                items.append(item)
            if not items:
                async for item in db.menu_items.find({}, {"_id": 0}).sort("rating", -1).limit(limit):
                    items.append(item)
            return json.dumps(items)

        elif tool_name == "get_item_details":
            name = tool_input.get("name", "")
            item = await db.menu_items.find_one(
                {"name": {"$regex": name, "$options": "i"}}, {"_id": 0}
            )
            if not item:
                return f"No item found matching '{name}'."
            return json.dumps(item)

        elif tool_name == "get_order_status":
            order_id = tool_input.get("order_id", "")
            order = await db.orders.find_one({"order_id": order_id}, {"_id": 0})
            if not order:
                return f"No order found with ID '{order_id}'."
            return json.dumps(order, default=str)

        return f"Unknown tool: {tool_name}"

    except Exception as e:
        return f"Tool error: {str(e)}"


@router.post("/")
async def agent_chat(req: AgentRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set in backend/.env")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        for _ in range(10):  # max tool loop iterations
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                text = next(
                    (block.text for block in response.content if hasattr(block, "text")), ""
                )
                return {"reply": text}

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = await run_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        return {"reply": "I'm sorry, I couldn't complete that request."}

    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid Anthropic API key.")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=503, detail="Could not reach Anthropic API.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {str(e)}")
