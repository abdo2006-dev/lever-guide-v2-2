"""
LeverGuide — single Render service.
FastAPI handles the ML/causal API and serves the Next.js static frontend.
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

# When start command is "cd apps/api && uvicorn ...", cwd is apps/api
# so ../web/out resolves to apps/web/out (the Next.js static export)
STATIC_DIR = os.environ.get("STATIC_DIR", "../web/out")
HAS_FRONTEND = os.path.isdir(STATIC_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting — static frontend at '{STATIC_DIR}': {HAS_FRONTEND}")
    yield


app = FastAPI(
    title="LeverGuide",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(analysis.router, prefix="/api")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": "2.0.0"})


# ── Frontend static files (registered after API routes) ──────────────────────
if HAS_FRONTEND:
    _next = os.path.join(STATIC_DIR, "_next")
    if os.path.isdir(_next):
        app.mount("/_next", StaticFiles(directory=_next), name="next-assets")

    _demo = os.path.join(STATIC_DIR, "demo")
    if os.path.isdir(_demo):
        app.mount("/demo", StaticFiles(directory=_demo), name="demo")

    def _page(name: str) -> FileResponse:
        p = os.path.join(STATIC_DIR, name, "index.html")
        return FileResponse(p if os.path.exists(p) else os.path.join(STATIC_DIR, "index.html"))

    @app.get("/setup")
    @app.get("/setup/")
    async def serve_setup() -> FileResponse:
        return _page("setup")

    @app.get("/analyze")
    @app.get("/analyze/")
    async def serve_analyze() -> FileResponse:
        return _page("analyze")

    @app.get("/")
    async def serve_home() -> FileResponse:
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
