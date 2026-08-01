cd cis6035-ai-service

Copy-Item .env.example .env

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000