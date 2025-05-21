#!/bin/bash

# This script renames the configuration directories from MaPLe to DualPrompt
# and from MaPLeFederated to DualPromptFL, preserving the original directories
# in case you need them for reference.

# Create the destination directories if they don't exist
mkdir -p configs/trainers/DualPrompt configs/trainers/DualPromptFL

# Copy files from MaPLe to DualPrompt
cp -r configs/trainers/MaPLe/* configs/trainers/DualPrompt/

# Copy files from MaPLeFederated to DualPromptFL
cp -r configs/trainers/MaPLeFederated/* configs/trainers/DualPromptFL/

# Update the yaml files in DualPrompt directory
for file in configs/trainers/DualPrompt/*.yaml; do
  sed -i.bak 's/MAPLE:/DUALPROMPT:/' "$file"
  rm -f "${file}.bak"
done

# Update the yaml files in DualPromptFL directory
for file in configs/trainers/DualPromptFL/*.yaml; do
  sed -i.bak 's/MAPLE:/DUALPROMPT:/' "$file"
  rm -f "${file}.bak"
done

echo "Configuration directories have been renamed and updated:"
echo "- MaPLe -> DualPrompt"
echo "- MaPLeFederated -> DualPromptFL"
echo "Original directories are preserved for reference."

# Now rename the trainers
cp trainers/maple.py trainers/dualprompt.py
cp trainers/maple_fed.py trainers/dualprompt_fl.py

echo "Trainer files have been copied:"
echo "- trainers/maple.py -> trainers/dualprompt.py"
echo "- trainers/maple_fed.py -> trainers/dualprompt_fl.py"
echo "Original files are preserved for reference." 