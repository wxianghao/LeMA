#!/bin/bash

# Define the target directory
TARGET_DIR=".data/MNIST/raw"

# List of MNIST dataset files
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

# Using Google's stable mirror (Original source: http://yann.lecun.com/exdb/mnist/)
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Starting download of the MNIST dataset to $TARGET_DIR/..."

# Iterate through the list and download each file
for FILE in "${FILES[@]}"; do
    TARGET_PATH="$TARGET_DIR/$FILE"
    
    # Check if the file already exists to avoid redundant downloads
    if [ -f "$TARGET_PATH" ]; then
        echo "✅ $FILE already exists, skipping download."
    else
        echo "⬇️ Downloading $FILE ..."
        # Use curl to download: -L follows redirects, -# shows progress bar, -o specifies output path
        curl -L -# "${BASE_URL}/${FILE}" -o "$TARGET_PATH"
        
        # Alternatively, if you prefer wget, comment out the curl line above and uncomment this:
        # wget -q --show-progress "${BASE_URL}/${FILE}" -O "$TARGET_PATH"
    fi
done

echo "🎉 MNIST dataset download complete!"

# === Unzip the files ===
echo "Extracting files..."
for FILE in "${FILES[@]}"; do
    gzip -d -k "$TARGET_DIR/$FILE"
done
echo "Extraction complete!"