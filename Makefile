PACKAGE = polytensor
PY = python3
VENV = .env
TENV = .tenv
BIN = $(VENV)/bin
TIN = $(TENV)/bin

all: doc test .tenv

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
	@$(TIN)/sphinx-build -M html docs/source docs/build

.PHONY: test
test: $(TENV)
	#$(BIN)/pytest -s ./test/testCoefficients.py 
	#$(TIN)/pytest -s ./test/testPackage.py 
	#$(BIN)/pytest -s ./test/testGrad.py 
	$(TIN)/pytest -s ./test/testPotts.py 
	$(TIN)/pytest -s ./test/testClock.py 

clean:
	rm -rf $(VENV)
	find . -type f -name *.pyc -delete
	find . -type d -name __pycache__ -delete

.PHONY: serve
serve: doc
	cd docs/build/html && $(PY) -m http.server 8018
