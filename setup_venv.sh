#!/bin/bash

# Define the name of your virtual environment directory
VENV_DIR=".venv"

# Remove the existing virtual environment directory if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment: $VENV_DIR"
    rm -rf $VENV_DIR
else
    echo "No existing virtual environment found. Proceeding."
fi

# Create a new virtual environment
echo "Creating a new virtual environment..."
python3 -m venv $VENV_DIR

# Activate the virtual environment
echo "Activating the new virtual environment..."
source $VENV_DIR/bin/activate

# Install your package in editable mode
echo "Installing the package in editable mode..."
pip install -e .

echo "Setup completed successfully."
