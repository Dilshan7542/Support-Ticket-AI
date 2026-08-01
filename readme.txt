cd cis6035-ai-service

Copy-Item .env.example .env

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python -m app.main
