#!/bin/bash
# Save images to output folder with timestamped subfolder
# Usage: ./save_images.sh

SRC_DIR="$(dirname "$0")/temp_image"
DEST_BASE="$HOME/workspace/rfharvest/image_out"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DEST_DIR="$DEST_BASE/$TIMESTAMP"

# Create destination directory
mkdir -p "$DEST_DIR"

# Copy all PNG files
cp "$SRC_DIR"/*.png "$DEST_DIR/" 2>/dev/null

# Check if copy was successful
if [ $? -eq 0 ]; then
    echo "Images saved to: $DEST_DIR"
    ls -la "$DEST_DIR"
else
    echo "No images found in $SRC_DIR"
    exit 1
fi
