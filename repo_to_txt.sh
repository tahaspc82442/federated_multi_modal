#!/bin/bash

# Check if a path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-git-repo>"
    exit 1
fi

REPO_PATH="$1"
OUTPUT_DIR="output"
OUTPUT_FILE="$OUTPUT_DIR/repo_text_output.txt"

# Check if the directory exists
if [ ! -d "$REPO_PATH" ]; then
    echo "Error: Directory '$REPO_PATH' does not exist."
    exit 1
fi

# Navigate to the repo
cd "$REPO_PATH" || exit

# Ensure it's a Git repository
if [ ! -d ".git" ]; then
    echo "Error: '$REPO_PATH' is not a valid Git repository."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Clear the output file
> "$OUTPUT_FILE"

# Find all text-based files, excluding .txt files and anything in the output directory
find . -type f ! -path "./.git/*" ! -name "*.txt" ! -path "./$OUTPUT_DIR/*" | while read -r file; do
    # Check if the file is a text file using `file` command
    if file --mime-type "$file" | grep -q 'text/'; then
        echo "Processing: $file"
        echo "========== FILE: $file ==========" >> "$OUTPUT_FILE"
        cat "$file" >> "$OUTPUT_FILE"
        echo -e "\n\n" >> "$OUTPUT_FILE"
    fi
done

echo "All text content has been saved to: $OUTPUT_FILE"