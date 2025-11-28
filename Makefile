# Get the directory of the Makefile itself
PROJECT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
VENV_DIR := $(PROJECT_DIR).venv
ACTIVATE := $(VENV_DIR)/bin/activate

# Detect OS
UNAME_S := $(shell uname -s)

install:
	@echo "Checking for Python venv support..."
	@python3 -m venv --help > /dev/null 2>&1 || \
		(echo "❌ python3-venv is not available."; \
		if [ "$(UNAME_S)" = "Linux" ]; then \
			echo "👉 Try: sudo apt install python3-venv"; \
		elif [ "$(UNAME_S)" = "Darwin" ]; then \
			echo "👉 Try: brew install python@3"; \
		else \
			echo "👉 Please install Python venv support for your OS."; \
		fi; \
		exit 1)

	@echo "Creating virtual environment at $(VENV_DIR)..."
	@python3 -m venv $(VENV_DIR)

	@echo "Installing package in editable mode..."
	@. $(ACTIVATE) && pip install -e $(PROJECT_DIR)

	@echo "Running rdtomo setup..."
	@. $(ACTIVATE) && rdtomo dev update-install && rdtomo setup

	@echo "All done."

update:
	@. $(ACTIVATE) && pip install -e $(PROJECT_DIR)
	@. $(ACTIVATE) && rdtomo dev update-install
	@echo "Installation updated."