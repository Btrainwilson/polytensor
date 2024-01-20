PACKAGE = polytensor
PY = python
VENV = .env
TENV = .tenv
BIN = $(VENV)/bin
TIN = $(TENV)/bin

all: doc .tenv

$(TENV): testrequirements.txt
	$(PY) -m venv $(TENV)
	$(TIN)/pip install --upgrade -r testrequirements.txt
	touch $(TENV)

.PHONY: doc
doc: $(TENV)
	@cd docs && make clean
	@$(TIN)/sphinx-build -M html docs/source docs/build

clean:
	rm -rf $(VENV)
	find . -type f -name *.pyc -delete
	find . -type d -name __pycache__ -delete

.PHONY: serve
serve: doc
	cd docs/build/html && $(PY) -m http.server 8018
