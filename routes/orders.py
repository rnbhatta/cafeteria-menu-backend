from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database import db
from models import OrderCreate, OrderResponse

router = APIRouter(prefix="/api/orders", tags=["orders"])


@router.post("/", response_model=OrderResponse)
async def create_order(order: OrderCreate):
    order_id = "CAF-" + str(random.randint(10000, 99999))
    now = datetime.utcnow()
    estimated_ready = now + timedelta(minutes=18)

    doc = {
        "order_id": order_id,
        "items": [i.model_dump() for i in order.items],
        "details": order.details.model_dump(),
        "payment": order.payment.model_dump(),
        "subtotal": order.subtotal,
        "tax": order.tax,
        "total": order.total,
        "placed_at": now,
        "estimated_ready": estimated_ready,
        "status": "pending",
    }

    await db.orders.insert_one(doc)

    return OrderResponse(**{k: v for k, v in doc.items() if k != "_id"})


@router.get("/")
async def get_orders():
    orders = []
    async for order in db.orders.find({}, {"_id": 0}).sort("placed_at", -1):
        orders.append(order)
    return orders


@router.get("/{order_id}")
async def get_order(order_id: str):
    order = await db.orders.find_one({"order_id": order_id}, {"_id": 0})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order
