#!/bin/bash

# Define source and destination directories
SOURCE_DIR="/Users/nikhilrazab-sekh/Desktop/defenseMats/simpful/simpful/gp_fuzzy_system/tests/best_model_dir"
DEST_DIR="/Users/nikhilrazab-sekh/Desktop/defenseMats/simpful_btc/implementation/saved_models/sept_models"

# Get today's date in YYYYMMDD format
TODAY=$(date +%Y%m%d)

# Create a new folder with today's date in the destination directory
NEW_DEST_DIR="$DEST_DIR/$TODAY"
mkdir -p "$NEW_DEST_DIR"

# Initialize the model counter
COUNTER=1

# Iterate through each subdirectory in the source directory
for MODEL_SUBDIR in "$SOURCE_DIR"/*; do
    if [ -d "$MODEL_SUBDIR" ]; then
        # Check if the best_model.pkl file exists in the subdirectory
        MODEL_FILE="$MODEL_SUBDIR/best_model.pkl"
        if [ -f "$MODEL_FILE" ]; then
            # Define the new file name with increment
            NEW_FILE_NAME="best_model${COUNTER}.pkl"
            
            # Copy the model file to the new directory with the new name
            cp "$MODEL_FILE" "$NEW_DEST_DIR/$NEW_FILE_NAME"
            
            echo "Copied $MODEL_FILE to $NEW_DEST_DIR/$NEW_FILE_NAME"
            
            # Increment the counter
            COUNTER=$((COUNTER + 1))
        fi
    fi
done