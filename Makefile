.PHONY: install test lint typecheck quality api

install:
	python -m pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check .

typecheck:
	mypy packages/core/src apps/api/src

quality: lint typecheck test

api:
	uvicorn quant_os_api.main:app --app-dir apps/api/src --reload
