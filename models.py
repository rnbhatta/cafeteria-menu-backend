from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class MenuItem(BaseModel):
    id: int
    name: str
    description: str
    price: float
    category: str
    tags: List[str] = []
    calories: int
    rating: float
    popular: bool
    emoji: str


class OrderItem(BaseModel):
    id: int
    name: str
    price: float
    quantity: int
    emoji: str


class OrderDetails(BaseModel):
    name: str
    email: str
    seat: str
    notes: str = ""


class PaymentInfo(BaseModel):
    method: str  # 'card', 'upi', 'cash'


class OrderCreate(BaseModel):
    items: List[OrderItem]
    details: OrderDetails
    payment: PaymentInfo
    subtotal: float
    tax: float
    total: float


class OrderResponse(BaseModel):
    order_id: str
    items: List[OrderItem]
    details: OrderDetails
    payment: PaymentInfo
    subtotal: float
    tax: float
    total: float
    placed_at: datetime
    estimated_ready: datetime
    status: str
