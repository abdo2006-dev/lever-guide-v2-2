"""
LeverGuide — single-service deployment.
FastAPI handles the ML/causal API and also serves the Next.js static frontend.
"""
from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers import analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Static files are built by the Dockerfile into /app/web/out
STATIC_DIR = os.environ.get(
    "STATIC_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "out"),
)
HAS_FRONTEND = os.path.isdir(STATIC_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"LeverGuide API starting — frontend present: {HAS_FRONTEND}")
    yield


app = FastAPI(
    title="LeverGuide",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── API routes (must be registered BEFORE the static file catch-all) ─────────
app.include_router(analysis.router, prefix="/api")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": "2.0.0"})


# ── Frontend routes ───────────────────────────────────────────────────────────
if HAS_FRONTEND:
    # Serve _next assets (JS/CSS bundles)
    next_assets = os.path.join(STATIC_DIR, "_next")
    if os.path.isdir(next_assets):
        app.mount("/_next", StaticFiles(directory=next_assets), name="next-assets")

    # Serve demo CSV
    demo_dir = os.path.join(STATIC_DIR, "demo")
    if os.path.isdir(demo_dir):
        app.mount("/demo", StaticFiles(directory=demo_dir), name="demo")

    def _html(page: str) -> FileResponse:
        path = os.path.join(STATIC_DIR, page, "index.html")
        if os.path.exists(path):
            return FileResponse(path)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/setup")
    @app.get("/setup/")
    async def serve_setup() -> FileResponse:
        return _html("setup")

    @app.get("/analyze")
    @app.get("/analyze/")
    async def serve_analyze() -> FileResponse:
        return _html("analyze")

    @app.get("/")
    async def serve_home() -> FileResponse:
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
