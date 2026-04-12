PY=venv/bin/python

start:
	$(PY) -m video_tranquitor --watch

process:
	$(PY) -m video_tranquitor --file "$(FILE)"

test:
	$(PY) -m pytest tests/ -v

lint:
	$(PY) -m ruff check src/ tests/

format:
	$(PY) -m ruff format src/ tests/
