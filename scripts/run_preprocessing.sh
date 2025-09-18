#!/bin/bash
# Preprocessing pipeline execution script
# Run this script from the PROJECT root directory with: bash scripts/run_preprocessing.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Move to project root (parent of scripts directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Set paths relative to project root
AMOS_DIR="./data/AMOS"  # Update folder name for your name
CHAOS_DIR="./data/CHAOS"  # Update folder name for your name
OUTPUT_DIR="./preprocessed_data" 

# Print current working directory for clarity
echo "Working from: $(pwd)"
echo "Looking for datasets in:"
echo "  AMOS: $AMOS_DIR"
echo "  CHAOS: $CHAOS_DIR"
echo "Output will be saved to: $OUTPUT_DIR"
echo ""

# Check if data directory exists
if [ ! -d "./data" ]; then
    echo "ERROR: ./data directory not found!"
    echo "Please ensure you're running this script from the project root."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run preprocessing
echo "Starting preprocessing pipeline..."
echo "================================"

# Process both datasets in one command if both exist
if [ -d "$AMOS_DIR" ] && [ -d "$CHAOS_DIR" ]; then
    echo "Found both AMOS and CHAOS datasets. Processing together..."
    python src/run_preprocessing.py \
        --amos_dir "$AMOS_DIR" \
        --chaos_dir "$CHAOS_DIR" \
        --output_dir "$OUTPUT_DIR"
        
# Process only AMOS if available
elif [ -d "$AMOS_DIR" ]; then
    echo "Found AMOS dataset. Processing..."
    python src/run_preprocessing.py \
        --amos_dir "$AMOS_DIR" \
        --output_dir "$OUTPUT_DIR"
        
# Process only CHAOS if available
elif [ -d "$CHAOS_DIR" ]; then
    echo "Found CHAOS dataset. Processing..."
    python src/run_preprocessing.py \
        --chaos_dir "$CHAOS_DIR" \
        --output_dir "$OUTPUT_DIR"
else
    echo "ERROR: No datasets found!"
    echo "Please check that your data directory contains AMOS and/or CHAOS folders."
    exit 1
fi

# Check if preprocessing was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "Preprocessing complete!"
    echo "Output saved to: $OUTPUT_DIR"
    
    # Print summary of created files
    echo ""
    echo "Created directory structure:"
    tree -d -L 4 "$OUTPUT_DIR" 2>/dev/null || find "$OUTPUT_DIR" -type d | head -20
    
    # Count files created
    echo ""
    echo "File counts:"
    for dir in "$OUTPUT_DIR"/*/*/images "$OUTPUT_DIR"/*/*/labels "$OUTPUT_DIR"/*/*/labels_orig; do
        if [ -d "$dir" ]; then
            count=$(find "$dir" -name "*.nii.gz" 2>/dev/null | wc -l)
            echo "  $dir: $count files"
        fi
    done
else
    echo ""
    echo "ERROR: Preprocessing failed. Check the error messages above."
    exit 1
fi