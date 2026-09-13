PY ?= python3
RUN ?= runs/val.npz
PORT ?= 8000

.PHONY: install dev selftest test lint check campaign renders serve clean

install:
	$(PY) -m pip install -r requirements.txt

dev:
	$(PY) -m pip install -r requirements-dev.txt

selftest:
	$(PY) -m astraeus selftest

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check --select E,F,W --line-length 130 astraeus isaac tests scripts

check: lint test selftest

campaign:
	$(PY) -m astraeus run --n 4000 --seed 20260912 --out $(RUN)
	$(PY) -m astraeus cem $(RUN) --prior P2
	$(PY) -m astraeus cem $(RUN) --prior P5
	$(PY) -m astraeus report $(RUN)

renders: $(RUN)
	$(PY) scripts/make_renders.py $(RUN) --prior P1 --quality 3

serve:
	$(PY) -m uvicorn astraeus.app:app --host 0.0.0.0 --port $(PORT)

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
