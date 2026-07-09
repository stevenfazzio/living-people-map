.PHONY: install lint format test clean

install:
	uv sync --extra dev

lint:
	uv run ruff check pipeline tests
	uv run ruff format --check pipeline tests

format:
	uv run ruff format pipeline tests

test:
	uv run pytest

clean:
	rm -rf .ruff_cache .pytest_cache
