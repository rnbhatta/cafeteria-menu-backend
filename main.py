import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routes.menu import router as menu_router
from routes.orders import router as orders_router
from routes.query import router as query_router
from routes.agent import router as agent_router
from routes.analytics import router as analytics_router

load_dotenv()

app = FastAPI(title="Cafeteria Menu API", version="1.0.0")

default_origins = "http://localhost:5173,http://localhost:4173"
origins = os.getenv("ALLOWED_ORIGINS", default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(menu_router)
app.include_router(orders_router)
app.include_router(query_router)
app.include_router(agent_router)
app.include_router(analytics_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
