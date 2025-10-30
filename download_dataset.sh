#!/bin/bash
# ---
# This 'strict mode' makes the script safer:
# -e: Exit immediately if a command fails.
# -u: Treat unset variables as an error.
# -o pipefail: Make a pipe fail if any command in it fails.
# ---
set -euo pipefail

CONFIG_JSON="structure.json"

# --- Dependency Checks ---

# Ensure jq is available
if ! command -v jq &> /dev/null; then
  echo "jq not found. Installing jq..."
  sudo apt-get update && sudo apt-get install -y jq
fi

# Ensure unzip is available
if ! command -v unzip &> /dev/null; then
  echo "unzip not found. Installing unzip..."
  sudo apt-get update && sudo apt-get install -y unzip
fi

# Ensure AWS CLI is available
if ! command -v aws &> /dev/null
then
    echo "--------------------------------------------------"
    echo "AWS CLI (command 'aws') is not found."
    echo "Installing AWS CLI right now"
    # Download and install AWS CLI
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    
    # Clean up installation files
    rm -rf awscliv2.zip
    rm -rf ./aws
    echo "AWS CLI installed."
fi

# --- Main Script Logic ---

OUTPUT_DIR=$(jq -r '.output_dir' "$CONFIG_JSON")
echo "Ensuring output directory exists at: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

RELEASE_COUNT=$(jq '.releases | length' "$CONFIG_JSON")
echo "Found $RELEASE_COUNT releases to process."

for ((i=0; i<RELEASE_COUNT; i++)); do
  RELEASE_NAME=$(jq -r ".releases[$i].release_name" "$CONFIG_JSON")
  # This is now assumed to be an s3:// URI
  ZIP_URL=$(jq -r ".releases[$i].zip_url" "$CONFIG_JSON")
  TARGET_DIR=$(jq -r ".releases[$i].target_dir" "$CONFIG_JSON")

  LOCAL_ZIP="${OUTPUT_DIR}/${RELEASE_NAME}.zip"
  EXTRACT_DIR="${OUTPUT_DIR}/${RELEASE_NAME}"
  FINAL_TARGET="$OUTPUT_DIR/$TARGET_DIR"

  echo "---------------------------------------------"
  echo "Processing $RELEASE_NAME..."
  echo "  S3 Source: $ZIP_URL"
  echo "  Local Target: $FINAL_TARGET"

  # Download zip using AWS S3 CLI
  echo "Downloading $RELEASE_NAME from $ZIP_URL..."
  # Using --no-sign-request based on your other script
  aws s3 cp "${ZIP_URL}" "${LOCAL_ZIP}" --no-sign-request --only-show-errors

  echo "Download complete. Extracting..."
  # Extract zip
  unzip -o "$LOCAL_ZIP" -d "$OUTPUT_DIR"

  # If the extracted folder is not named as expected, try to find it
  # This handles cases where zips have a slightly different top-level folder name
  if [ ! -d "$EXTRACT_DIR" ]; then
    echo "Extracted directory '$EXTRACT_DIR' not found."
    # Try to find the first directory in OUTPUT_DIR that matches the release name pattern
    FOUND_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "${RELEASE_NAME}*" | head -n 1)
    
    if [ -n "$FOUND_DIR" ]; then
      echo "Found matching directory at '$FOUND_DIR'. Using it."
      EXTRACT_DIR="$FOUND_DIR"
    else
      echo "Error: Could not find extracted directory for $RELEASE_NAME. Skipping."
      # Clean up the zip and continue to the next item in the loop
      rm -f "$LOCAL_ZIP"
      continue
    fi
  fi

  # Move to target dir if needed
  if [ "$EXTRACT_DIR" != "$FINAL_TARGET" ]; then
    echo "Moving '$EXTRACT_DIR' to '$FINAL_TARGET'..."
    # Remove the final target if it already exists to prevent 'mv' errors
    rm -rf "$FINAL_TARGET"
    mv "$EXTRACT_DIR" "$FINAL_TARGET"
  else
    echo "Extracted directory is already at the final target location."
  fi

  # Clean up zip
  echo "Cleaning up local zip file: $LOCAL_ZIP"
  rm -f "$LOCAL_ZIP"

  echo "Done with $RELEASE_NAME. Final contents of $FINAL_TARGET:"
  ls -la "$FINAL_TARGET"
done

echo "---------------------------------------------"
echo "All releases processed."
