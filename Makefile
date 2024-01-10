PACKAGE = polytensor
PY = python
VENV = .env
TENV = .tenv
BIN = $(VENV)/bin
TIN = $(TENV)/bin

all: doc test .testenv

$(VENV): requirements.txt
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade -r requirements.txt
	touch $(VENV)

$(TENV): testrequirements.txt setup.py
	$(PY) -m venv $(TENV)
	$(TIN)/pip install --upgrade -r testrequirements.txt
	$(TIN)/pip install -e .
	touch $(TENV)

.PHONY: pypi
pypi: 
	python setup.py sdist
	twine upload dist/*

.PHONY: initdoc
initdoc: $(TENV)
	@$(TIN)/sphinx-quickstart docs

.PHONY: doc
doc: $(TENV)
	@cd docs && make clean
	@$(TIN)/sphinx-apidoc -o ./docs/source/ ./$(PACKAGE)
	@cd docs && make html

.PHONY: test
test: $(TENV)
	#$(BIN)/pytest -s ./test/testCoefficients.py 
	$(TIN)/pytest -s ./test/testPackage.py 
	#$(BIN)/pytest -s ./test/testGrad.py 

clean:
	rm -rf $(VENV)
	find . -type f -name *.pyc -delete
	find . -type d -name __pycache__ -delete

.PHONY: serve
serve: doc
	cd docs/build/html && $(PY) -m http.server 8018
