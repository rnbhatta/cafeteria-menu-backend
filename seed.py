"""Run this once to populate MongoDB with menu data: python seed.py"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "cafeteria_menu")

MENU_ITEMS = [
    # Breakfast
    {"id": 1, "name": "Avocado Toast", "description": "Sourdough toast with smashed avocado, cherry tomatoes, and poached egg", "price": 8.50, "category": "breakfast", "tags": ["vegetarian"], "calories": 420, "rating": 4.7, "popular": True, "emoji": "🥑"},
    {"id": 2, "name": "Greek Yogurt Parfait", "description": "Creamy Greek yogurt layered with granola, honey, and fresh berries", "price": 6.00, "category": "breakfast", "tags": ["vegetarian", "healthy"], "calories": 280, "rating": 4.5, "popular": False, "emoji": "🫙"},
    {"id": 3, "name": "Egg & Cheese Sandwich", "description": "Fluffy scrambled eggs, cheddar cheese, and turkey bacon on a brioche bun", "price": 7.25, "category": "breakfast", "tags": [], "calories": 510, "rating": 4.6, "popular": True, "emoji": "🥚"},
    {"id": 4, "name": "Overnight Oats", "description": "Rolled oats soaked in almond milk with chia seeds, banana, and maple syrup", "price": 5.50, "category": "breakfast", "tags": ["vegan", "healthy"], "calories": 340, "rating": 4.3, "popular": False, "emoji": "🌾"},
    # Lunch
    {"id": 5, "name": "Grilled Chicken Bowl", "description": "Herb-marinated grilled chicken over brown rice with roasted vegetables and tahini", "price": 12.50, "category": "lunch", "tags": ["healthy", "gluten-free"], "calories": 580, "rating": 4.8, "popular": True, "emoji": "🍗"},
    {"id": 6, "name": "Caesar Salad", "description": "Crisp romaine lettuce, parmesan, house-made croutons, and classic Caesar dressing", "price": 9.00, "category": "lunch", "tags": ["vegetarian"], "calories": 380, "rating": 4.4, "popular": False, "emoji": "🥗"},
    {"id": 7, "name": "Turkey Club Wrap", "description": "Sliced turkey, bacon, lettuce, tomato, and avocado in a whole-wheat tortilla", "price": 10.75, "category": "lunch", "tags": ["healthy"], "calories": 490, "rating": 4.6, "popular": True, "emoji": "🌯"},
    {"id": 8, "name": "Margherita Pizza Slice", "description": "Classic Neapolitan-style with San Marzano tomato, fresh mozzarella, and basil", "price": 4.50, "category": "lunch", "tags": ["vegetarian"], "calories": 290, "rating": 4.5, "popular": False, "emoji": "🍕"},
    {"id": 9, "name": "Lentil Soup", "description": "Hearty red lentil soup with cumin, turmeric, lemon, and crusty bread", "price": 7.00, "category": "lunch", "tags": ["vegan", "healthy"], "calories": 310, "rating": 4.3, "popular": False, "emoji": "🍲"},
    {"id": 10, "name": "Beef Burger", "description": "Smash burger with aged cheddar, caramelised onions, pickles, and special sauce", "price": 13.00, "category": "lunch", "tags": [], "calories": 720, "rating": 4.9, "popular": True, "emoji": "🍔"},
    # Snacks
    {"id": 11, "name": "Mixed Fruit Bowl", "description": "Seasonal fresh fruit — watermelon, grapes, pineapple, and strawberries", "price": 4.00, "category": "snacks", "tags": ["vegan", "healthy", "gluten-free"], "calories": 120, "rating": 4.4, "popular": False, "emoji": "🍓"},
    {"id": 12, "name": "Hummus & Veggies", "description": "Creamy roasted red pepper hummus with cucumber, carrots, and bell peppers", "price": 5.50, "category": "snacks", "tags": ["vegan", "healthy"], "calories": 180, "rating": 4.5, "popular": False, "emoji": "🥕"},
    {"id": 13, "name": "Protein Energy Balls", "description": "Oat, peanut butter, chocolate chip, and honey bites — 3 per serving", "price": 3.50, "category": "snacks", "tags": ["vegetarian", "healthy"], "calories": 210, "rating": 4.6, "popular": True, "emoji": "🟤"},
    {"id": 14, "name": "Cheese & Crackers", "description": "Selection of artisan cheeses with whole-grain crackers and fig jam", "price": 6.50, "category": "snacks", "tags": ["vegetarian"], "calories": 340, "rating": 4.4, "popular": False, "emoji": "🧀"},
    # Beverages
    {"id": 15, "name": "Cold Brew Coffee", "description": "Smooth 12-hour cold-brewed coffee served over ice — customize with oat milk", "price": 4.50, "category": "beverages", "tags": ["vegan"], "calories": 15, "rating": 4.8, "popular": True, "emoji": "🧊"},
    {"id": 16, "name": "Flat White", "description": "Double ristretto espresso with velvety steamed whole milk", "price": 4.00, "category": "beverages", "tags": ["vegetarian"], "calories": 120, "rating": 4.7, "popular": True, "emoji": "☕"},
    {"id": 17, "name": "Fresh Green Juice", "description": "Spinach, cucumber, apple, ginger, and lemon — cold pressed daily", "price": 6.00, "category": "beverages", "tags": ["vegan", "healthy", "gluten-free"], "calories": 90, "rating": 4.5, "popular": False, "emoji": "🥤"},
    {"id": 18, "name": "Sparkling Water", "description": "San Pellegrino natural sparkling mineral water 500ml", "price": 2.50, "category": "beverages", "tags": ["vegan", "gluten-free"], "calories": 0, "rating": 4.2, "popular": False, "emoji": "💧"},
    # Desserts
    {"id": 19, "name": "Chocolate Lava Cake", "description": "Warm dark chocolate cake with molten center, served with vanilla ice cream", "price": 7.50, "category": "desserts", "tags": ["vegetarian"], "calories": 520, "rating": 4.9, "popular": True, "emoji": "🍫"},
    {"id": 20, "name": "NY Cheesecake", "description": "Classic baked New York cheesecake with strawberry compote", "price": 6.50, "category": "desserts", "tags": ["vegetarian"], "calories": 450, "rating": 4.7, "popular": False, "emoji": "🍰"},
    {"id": 21, "name": "Mango Sorbet", "description": "Dairy-free alphonso mango sorbet — two scoops", "price": 4.50, "category": "desserts", "tags": ["vegan", "gluten-free"], "calories": 180, "rating": 4.5, "popular": False, "emoji": "🍧"},
]


async def seed():
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DATABASE_NAME]

    existing = await db.menu_items.count_documents({})
    if existing > 0:
        print(f"Database already has {existing} menu items. Skipping seed.")
        client.close()
        return

    await db.menu_items.insert_many(MENU_ITEMS)
    await db.menu_items.create_index("id", unique=True)
    await db.menu_items.create_index("category")
    await db.orders.create_index("order_id", unique=True)
    await db.orders.create_index("placed_at")

    print(f"Seeded {len(MENU_ITEMS)} menu items into MongoDB.")
    client.close()


if __name__ == "__main__":
    asyncio.run(seed())
