"""Vercel entrypoint: exposes the FastAPI app (backend/app) as `app`."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
from app.main import app  # noqa: E402,F401
