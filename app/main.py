from __future__ import annotations

from app.factory import create_app

app = create_app()


if __name__ == "__main__":
    settings = app.config["settings"]
    app.run(host=settings.host, port=settings.port)
