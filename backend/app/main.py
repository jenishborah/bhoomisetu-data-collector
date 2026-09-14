from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers.dashboard import (
    router as dashboard_router,
)

from backend.app.routers.projects import (
    router as projects_router,
)


app = FastAPI(
    title="BhoomiSetu API",
    description=(
        "Land Acquisition Intelligence "
        "& Decision Support API"
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://bhoomisetu-data-collector.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================
# Routers
# ============================================================

app.include_router(
    projects_router,
    prefix="/api",
)

app.include_router(
    dashboard_router,
    prefix="/api",
)


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {
        "name": "BhoomiSetu",

        "description": (
            "Land Acquisition Intelligence "
            "& Decision Support Platform"
        ),

        "version": "1.0.0",

        "status": "running",
    }


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",

        "service": (
            "bhoomisetu-api"
        ),
    }