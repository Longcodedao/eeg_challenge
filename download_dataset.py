#!/usr/bin/env python3

"""
Downloads and extracts EEG datasets based on structure.json
to match the directory structure:
MyNeurIPSData/MyNeurIPSData/HBN_DATA_FULL/<release_name>/<temp_dir>

This version includes a tqdm progress bar for downloads.
"""

import json
import os
import pathlib
import zipfile
import sys
import tempfile

# --- Dependency Check ---
try:
    import requests
    from tqdm import tqdm # Import tqdm
except ImportError:
    print("--------------------------------------------------", file=sys.stderr)
    print("Error: 'requests' or 'tqdm' module not found.", file=sys.stderr)
    print("Please install it by running:", file=sys.stderr)
    print("  pip install requests tqdm", file=sys.stderr)
    print("--------------------------------------------------")
    sys.exit(1)
# --- End Dependency Check ---


# --- Configuration ---
# This is the base path from your desired structure
BASE_STORAGE_PATH = "LOL_DATASET"

# This is the main data folder from your structure
DATA_FOLDER_NAME = "HBN_DATA_FULL"

# The name of your JSON configuration file
JSON_FILE = "structure.json"
# --- End Configuration ---


def download_and_extract(release_info, base_dir):
    """
    Downloads and extracts a single release into the target structure.
    """
    progress_bar = None # Initialize for error handling
    temp_zip_path = None # Initialize for error handling
    
    try:
        release_name = release_info['release_name']
        zip_url = release_info['zip_url']
        # This is the directory name expected *inside* the zip (e.g., ds005505-bdf)
        expected_subdir = release_info['temp_dir'] 

        print("---")
        print(f"Processing: {release_name}")

        # 1. Define Paths
        # Path for the release, e.g., .../HBN_DATA_FULL/R1_L100_bdf
        release_path = base_dir / release_name
        # Final path for the data, e.g., .../R1_L100_bdf/ds005505-bdf
        final_data_path = release_path / expected_subdir

        # 2. Check if already exists
        # If the final data directory exists, we can skip
        if final_data_path.exists() and final_data_path.is_dir():
            print(f"Target directory {final_data_path} already exists. Skipping.")
            return True

        # 3. Create parent directory for extraction
        final_data_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory exists: {final_data_path}")

        # 4. Download to a temporary file (more memory-efficient for large files)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
            temp_zip_path = temp_zip.name # Store path for cleanup
            print(temp_zip_path)
            try:
                print(f"Starting download: {zip_url} ...")
                with requests.get(zip_url, stream=True) as r:
                    r.raise_for_status() # Check for download errors
                    
                    # Get total file size for tqdm
                    total_size_in_bytes = int(r.headers.get('content-length', 0))
                    block_size = 8192 # Set chunk size
                    
                    # Initialize tqdm progress bar
                    progress_bar = tqdm(
                        total=total_size_in_bytes, 
                        unit='iB', 
                        unit_scale=True, 
                        desc=f"Downloading {release_name}",
                        leave=True # Leave the bar after completion
                    )
                    
                    # Write data to temp file in chunks
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk: # filter out keep-alive new chunks
                            progress_bar.update(len(chunk)) # Update progress
                            temp_zip.write(chunk)
                    
                    progress_bar.close() # Close the progress bar
                    
                    # Check if download was successful but had no content-length
                    if total_size_in_bytes == 0 and progress_bar.n == 0:
                         print(f"Warning: Download complete for {release_name}, but file size was reported as 0. Continuing extraction.", file=sys.stderr)
                    elif total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                        print(f"Warning: Download for {release_name} may be incomplete. Expected {total_size_in_bytes} bytes, got {progress_bar.n} bytes.", file=sys.stderr)
                    else:
                        # Success message is handled by tqdm bar closing
                        pass

            except Exception:
                # Clean up temp file on download error
                if progress_bar:
                    progress_bar.close()
                if os.path.exists(temp_zip_path):
                    os.unlink(temp_zip_path)
                raise # Re-raise the exception

        # 5. Extract
        print(f"Extracting {temp_zip_path} to: {final_data_path} ...")
        # Note: z.extractall() doesn't have a simple progress hook.
        # For large zips, this will just take time.
        with zipfile.ZipFile(temp_zip_path, 'r') as z:
            z.extractall(final_data_path)

        # --- START: New logic to fix nested directory ---
        print("Checking for nested directory...")
        
        # Get a list of all items just extracted
        # We use list() to "freeze" the iterator before we move things
        extracted_items = list(final_data_path.iterdir())

        # Check if the zip extracted into a single wrapper directory
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            unwanted_dir = extracted_items[0]
            print(f"Detected nested directory: {unwanted_dir.name}. Moving contents up.")
            
            # Move all children from unwanted_dir up to final_data_path
            for item_to_move in unwanted_dir.iterdir():
                # The new destination path
                destination = final_data_path / item_to_move.name
                # Use pathlib's .rename() to move the file/dir
                item_to_move.rename(destination)
                
            # Remove the now-empty unwanted directory
            unwanted_dir.rmdir()
            print(f"Successfully moved contents and removed {unwanted_dir.name}.")
        else:
            print("No single nested directory found. Data is as extracted.")
        # --- END: New logic ---

        print("Extraction complete.")

        # 6. Clean up temporary zip file
        os.unlink(temp_zip_path)
        print(f"Removed temporary file: {temp_zip_path}")
        temp_zip_path = None # Clear path after successful removal

        # 7. Verify
        if final_data_path.exists() and final_data_path.is_dir():
            print(f"Successfully created: {final_data_path}")
            return True
        else:
            print(f"Warning: Extraction finished, but expected path {final_data_path} was not found.", file=sys.stderr)
            print(f"Please check the contents of {release_path}. The zip file might have an unexpected structure.", file=sys.stderr)
            return False

    except requests.RequestException as e:
        print(f"Error downloading {release_name}: {e}", file=sys.stderr)
    except zipfile.BadZipFile:
        print(f"Error: Downloaded file for {release_name} is not a valid zip file.", file=sys.stderr)
    except KeyError as e:
        print(f"Error: Missing key {e} in structure.json for release '{release_info.get('release_name', 'UNKNOWN')}'", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred for {release_name}: {e}", file=sys.stderr)
    
    # Final cleanup in case of error
    if progress_bar:
        progress_bar.close()
    if temp_zip_path and os.path.exists(temp_zip_path):
        print(f"Cleaning up temp file after error: {temp_zip_path}", file=sys.stderr)
        os.unlink(temp_zip_path)
        
    return False


def main():
    """
    Main function to read JSON and process all releases.
    """
    # 1. Find and load JSON
    json_path = pathlib.Path(JSON_FILE)
    if not json_path.exists():
        print(f"Error: {JSON_FILE} not found in {os.getcwd()}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading configuration from {json_path.resolve()}")
    try:
        with open(json_path, 'r') as f:
            structure = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding {JSON_FILE}: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Set up main data directory
    # This creates .../MyNeurIPSData/MyNeurIPSData/HBN_DATA_FULL
    main_data_dir = pathlib.Path(BASE_STORAGE_PATH) / DATA_FOLDER_NAME
    main_data_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- Starting Dataset Download and Extraction ---")
    print(f"All data will be stored in: {main_data_dir.resolve()}")

    # 3. Process all releases
    releases = structure.get('releases', [])
    if not releases:
        print("No 'releases' found in structure.json. Nothing to do.", file=sys.stderr)
        sys.exit(1)

    success_count = 0
    fail_count = 0
    for release in releases:
        if download_and_extract(release, main_data_dir):
            success_count += 1
        else:
            fail_count += 1

    # 4. Final report
    print("---")
    print("--- Processing Complete ---")
    print(f"Successfully processed/skipped: {success_count}")
    print(f"Failed: {fail_count}")

    if fail_count == 0:
        print("All datasets are ready.")
    else:
        print("Some datasets failed to process. Please check the errors above.")


if __name__ == "__main__":
    main()
