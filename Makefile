.PHONY: setup install-dev fetch train app test lint clean

setup:
	python -m venv .venv
	@echo "Activate with: source .venv/bin/activate (Windows: .venv\\Scripts\\activate)"

install-dev:
	python -m pip install -e ".[app,dev]"

fetch:
	protfunc fetch --output data/uniprot_ec.csv --per-class 500

train:
	protfunc train --data data/uniprot_ec.csv --model models/protfunc.joblib --reports-dir reports

app:
	streamlit run app.py

test:
	python -m unittest discover -s tests -v

lint:
	ruff check .

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('.').rglob('*.pyc')]"

