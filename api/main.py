"""FastAPI application entrypoint. Run with:
    .venv\\Scripts\\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8420
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes_rest import router as rest_router
from api.routes_ws import router as ws_router
from config.settings import settings
from database.db import init_db, start_writer_thread, stop_writer_thread
from utils.logger import get_logger

log = get_logger("main")

# Derived from this file's own location (not settings.db_path, which tests
# override to a temp directory) so it's always the real project root.
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting Omni Mind — atm_id=%s", settings.atm_id)
    init_db()
    start_writer_thread()
    yield
    stop_writer_thread()
    log.info("Omni Mind shut down cleanly")


app = FastAPI(title="Omni Mind — AI ATM Security System", lifespan=lifespan)

app.include_router(rest_router)
app.include_router(ws_router)

app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
app.mount("/audio", StaticFiles(directory=str(FRONTEND_DIR / "audio")), name="frontend-audio")
app.mount("/models", StaticFiles(directory=str(FRONTEND_DIR / "models")), name="frontend-models")
app.mount("/evidence", StaticFiles(directory=str(settings.evidence_dir)), name="evidence")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/favicon.ico")
def favicon():
    from fastapi.responses import Response

    return Response(status_code=204)
