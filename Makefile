.PHONY: install run-ui eval eval-html test lint docker-build

install:
	uv sync

run-ui:
	uv run streamlit run ui/app.py

eval:
	uv run python scripts/eval.py

eval-html:
	uv run python scripts/eval.py --html report.html

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check . && uv run ruff format --check .

docker-build:
	docker build -t job-fit-estimator .
