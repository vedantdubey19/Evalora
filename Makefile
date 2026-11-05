PYTHON ?= python

.PHONY: help install verify api dashboard
help:
	@echo 'make install    Install dependencies in the active environment'
	@echo 'make verify     Check dependencies and run offline tests'
	@echo 'make api        Start the API on localhost:8000'
	@echo 'make dashboard  Start the dashboard on localhost:8501'

install:
	$(PYTHON) -m pip install -r requirements.txt

verify:
	$(PYTHON) -m pip check
	$(PYTHON) -m pytest -q

api:
	$(PYTHON) -m uvicorn app.main:app --host 127.0.0.1 --port 8000

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py --server.address 127.0.0.1
