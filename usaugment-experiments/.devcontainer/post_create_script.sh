#!/bin/bash

# Switch to project root
cd ..

# Setup virtual environment
virtualenv .venv -p python3.11
source .venv/bin/activate
pip install --upgrade pip

# Install dependencies
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt

# Install the project as a package
pip install -e .

# Setup pre-commit hooks
pre-commit install
