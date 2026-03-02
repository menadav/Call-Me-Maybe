UV = uv
PYTHON = $(UV) run python
VENV_DIR = .venv
INSTALLED_FLAG = $(VENV_DIR)/.installed
GOINFRE_VENV = /goinfre/$(USER)/call_me_maybe_venv
ARGS ?= --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json

export UV_CACHE_DIR = /goinfre/$(USER)/uv_cache
export HF_HOME = /goinfre/$(USER)/huggingface_cache

all: install

install: $(INSTALLED_FLAG)

$(INSTALLED_FLAG): pyproject.toml
	@mkdir -p $(GOINFRE_VENV) $(UV_CACHE_DIR) $(HF_HOME)
	@rm -rf ~/.cache/huggingface
	@if [ ! -L $(VENV_DIR) ]; then \
		rm -rf $(VENV_DIR); \
		ln -s $(GOINFRE_VENV) $(VENV_DIR); \
	fi
	$(UV) sync
	@touch $(INSTALLED_FLAG)

run: install
	$(PYTHON) -m src $(ARGS)

debug: install
	$(PYTHON) -m pdb -m src

lint: install
	$(UV) run flake8 src
	$(UV) run mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

clean:
	@rm -rf .mypy_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} +

fclean: clean
	@rm -rf $(VENV_DIR)

.PHONY: all install run debug lint clean fclean
