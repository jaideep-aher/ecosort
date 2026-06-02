"""EcoSort entry point.

Launches the interactive inference web app. In production (Railway) the app is
served by gunicorn via the module-level ``app`` object (``main:app``); locally
you can run ``python main.py`` for a development server.
"""

from __future__ import annotations

import os

from app.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
