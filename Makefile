.PHONY: install test run clean

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest

run:
	python try_audioscript.py

demo: install run

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf output/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete