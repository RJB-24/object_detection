.PHONY: install models run docker test compile clean web-install web-build web-dev

install:
	pip install -r requirements-dev.txt

models:
	python scripts/download_models.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

web-install:
	cd frontend && npm install

web-build:
	cd frontend && npm run build

web-dev:
	cd frontend && npm run dev

docker:
	docker compose up --build

test:
	pytest -q

compile:
	python -m compileall app scripts training

clean:
	rm -rf __pycache__ app/__pycache__ outputs/* .pytest_cache
