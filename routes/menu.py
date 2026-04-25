from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database import db

router = APIRouter(prefix="/api/menu", tags=["menu"])


@router.get("/")
async def get_menu(
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
):
    query = {}
    if category and category != "all":
        query["category"] = category
    if search:
        query["name"] = {"$regex": search, "$options": "i"}
    if tag:
        query["tags"] = {"$in": [tag]}

    items = []
    async for item in db.menu_items.find(query, {"_id": 0}):
        items.append(item)
    return items


@router.get("/{item_id}")
async def get_menu_item(item_id: int):
    item = await db.menu_items.find_one({"id": item_id}, {"_id": 0})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item
