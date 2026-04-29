.PHONY: help venv install run test docker-build docker-run docker-stop fmt

help:
	@echo "make install | run | test | docker-build | docker-run"

install:
	pip install -r requirements.txt

run:
	python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

test:
	pytest -q

docker-build:
	docker buildx build --load -t claims-triage-api:0.4 .

docker-run:
	docker run --rm -p 8001:8000 --env-file .env claims-triage-api:0.4

