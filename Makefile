PY = python
VENV = .env
BIN = $(VENV)/bin

all: test .env

$(VENV): requirements.txt setup.py
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade -r requirements.txt
	$(BIN)/pip install -e .
	touch $(VENV)

.PHONY: doc
doc: $(VENV)
	@cd docs && make clean
	@$(BIN)/sphinx-apidoc -o ./docs/source/ ./polytensor
	@cd docs && make html

.PHONY: test
test: $(VENV)
	$(BIN)/pytest -s ./test/testPackage.py 
	$(BIN)/pytest -s ./test/testGrad.py 

clean:
	rm -rf $(VENV)
	find . -type f -name *.pyc -delete
	find . -type d -name __pycache__ -delete

.PHONY: serve
serve: doc
	cd docs/build/html && $(PY) -m http.server 8018
