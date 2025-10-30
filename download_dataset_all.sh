#!/bin/bash
# ---
# This 'strict mode' makes the script safer:
# -e: Exit immediately if a command fails.
# -u: Treat unset variables as an error.
# -o pipefail: Make a pipe fail if any command in it fails.
# ---
set -euo pipefail

echo "--- Downloading HBN Data ---"

# Define the storage path.
# <<< CHANGE THIS if your mount point is different
STORAGE_PATH="/data/storage"

# Use quotes to handle paths with spaces or special characters
DATA_OUTPUT_DIR="${STORAGE_PATH}/HBN_DATA_FULL"

echo "Creating data directory at: ${DATA_OUTPUT_DIR}"
mkdir -p "${DATA_OUTPUT_DIR}"

# Navigate to storage. 'set -e' will make the script exit if this cd fails.
cd "${STORAGE_PATH}"
echo "Changed directory to ${STORAGE_PATH}."

# Download only the required FULL BDF releases (R1-R11)
for i in {1..11}; do
  release_name="R${i}_L100_bdf"
  s3_path="s3://nmdatasets/NeurIPS25/${release_name}"
  local_path="${DATA_OUTPUT_DIR}/${release_name}"

  echo "---"
  echo "Downloading ${release_name} to ${local_path}..."

  # Use quotes on paths
  # 'set -e' will automatically catch an error here and stop the script
  aws s3 cp --recursive "${s3_path}" "${local_path}" --no-sign-request --only-show-errors
done

# This message will only be reached if all commands above succeeded
echo "---"
echo "--- HBN Data Download Complete ---"
echo "Verifying contents of ${DATA_OUTPUT_DIR}:"

# Use quotes on path
ls -l "${DATA_OUTPUT_DIR}"

# Go back to home directory
cd ~
echo "Returned to home directory."