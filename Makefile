UV = uv
PYTHON = $(UV) run python
VENV_DIR = .venv
GOINFRE_VENV = /goinfre/$(USER)/call_me_maybe_venv
INSTALLED_FLAG = $(VENV_DIR)/.installed
ARGS ?= --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json

export XDG_CACHE_HOME := /goinfre/$(USER)/.cache
export UV_CACHE_DIR := /goinfre/$(USER)/.cache/uv

all: $(INSTALLED_FLAG)

$(INSTALLED_FLAG): pyproject.toml
	@mkdir -p /goinfre/$(USER)/.cache
	@mkdir -p $(GOINFRE_VENV)
	@if [ ! -d $(GOINFRE_VENV)/bin ]; then \
		python3 -m venv $(GOINFRE_VENV); \
	fi
	@if [ ! -L $(VENV_DIR) ]; then \
		rm -rf $(VENV_DIR); \
		ln -s $(GOINFRE_VENV) $(VENV_DIR); \
	fi
	@UV_CACHE_DIR=/goinfre/$(USER)/.cache/uv uv sync
	@touch $(INSTALLED_FLAG)

run: $(INSTALLED_FLAG)
	@$(PYTHON) -m src $(ARGS)

debug: $(INSTALLED_FLAG)
	@$(PYTHON) -m pdb -m src

lint: $(INSTALLED_FLAG)
	@$(UV) run flake8 src
	@$(UV) run mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
clean:
	@rm -rf .mypy_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} +
fclean: clean
	@rm -rf $(VENV_DIR) $(INSTALLED_FLAG)
