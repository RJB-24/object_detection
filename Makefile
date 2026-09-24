.PHONY: install models run docker test lint clean

install:
	pip install -r requirements.txt

models:
	python scripts/download_models.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker:
	docker compose up --build

test:
	pytest -q

lint:
	python -m compileall app scripts

clean:
	rm -rf __pycache__ app/__pycache__ outputs/* .pytest_cache
