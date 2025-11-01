#!/usr/bin/env python3

"""
Downloads and extracts EEG datasets based on structure.json
to match the directory structure:
BASE_STORAGE_PATH/DATA_FOLDER_NAME/<release_name>/<temp_dir>

This version includes argparse for configuration and tqdm progress bar for downloads.
"""

import json
import os
import pathlib
import zipfile
import sys
import tempfile
import argparse

# --- Dependency Check ---
try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("--------------------------------------------------", file=sys.stderr)
    print("Error: 'requests' or 'tqdm' module not found.", file=sys.stderr)
    print("Please install it by running:", file=sys.stderr)
    print("  pip install requests tqdm", file=sys.stderr)
    print("--------------------------------------------------")
    sys.exit(1)
# --- End Dependency Check ---


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and extract EEG datasets based on structure JSON configuration"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--mode", 
        choices=["full", "mini"], 
        default="mini",
        help="Download mode: 'full' for complete dataseft, 'mini' for subset (default: mini)"
    )
    
    parser.add_argument(
        "--base-path", 
        type=str, 
        default="LOL_DATASET",
        help="Base storage path for downloaded datasets (default: LOL_DATASET)"
    )
    
    parser.add_argument(
        "--data-folder", 
        type=str, 
        default="HBN_DATA_FULL",
        help="Data folder name within base path (default: HBN_DATA_FULL)"
    )
    
    parser.add_argument(
        "--json-file", 
        type=str, 
        help="Custom JSON configuration file path (overrides --mode selection)"
    )
    
    # Download options
    parser.add_argument(
        "--skip-existing", 
        action="store_true", 
        default=True,
        help="Skip download if target directory already exists (default: True)"
    )
    
    parser.add_argument(
        "--force-redownload", 
        action="store_true",
        help="Force redownload even if target directory exists"
    )
    
    # Progress and verbosity
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress progress bars and verbose output"
    )
    
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=8192,
        help="Download chunk size in bytes (default: 8192)"
    )
    
    return parser.parse_args()


def download_and_extract(release_info, base_dir, args):
    """
    Downloads and extracts a single release into the target structure.
    
    Args:
        release_info: Dictionary containing release configuration
        base_dir: Base directory path for extraction
        args: Parsed command line arguments
    """
    progress_bar = None
    temp_zip_path = None
    
    try:
        release_name = release_info['release_name']
        zip_url = release_info['zip_url']
        expected_subdir = release_info['temp_dir']

        if not args.quiet:
            print("---")
            print(f"Processing: {release_name}")

        # 1. Define Paths
        release_path = base_dir / release_name
        final_data_path = release_path / expected_subdir

        # 2. Check if already exists
        if final_data_path.exists() and final_data_path.is_dir():
            if args.skip_existing and not args.force_redownload:
                if not args.quiet:
                    print(f"Target directory {final_data_path} already exists. Skipping.")
                return True
            elif args.force_redownload:
                if not args.quiet:
                    print(f"Force redownload enabled. Removing existing directory: {final_data_path}")
                import shutil
                shutil.rmtree(final_data_path)

        # 3. Create parent directory for extraction
        final_data_path.mkdir(parents=True, exist_ok=True)
        if not args.quiet:
            print(f"Ensured directory exists: {final_data_path}")

        # 4. Download to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
            temp_zip_path = temp_zip.name
            if not args.quiet:
                print(f"Starting download: {zip_url} ...")
                
            try:
                with requests.get(zip_url, stream=True) as r:
                    r.raise_for_status()
                    
                    # Get total file size for progress bar
                    total_size_in_bytes = int(r.headers.get('content-length', 0))
                    
                    # Initialize progress bar (only if not quiet)
                    if not args.quiet:
                        progress_bar = tqdm(
                            total=total_size_in_bytes, 
                            unit='iB', 
                            unit_scale=True, 
                            desc=f"Downloading {release_name}",
                            leave=True
                        )
                    
                    # Write data to temp file in chunks
                    downloaded_bytes = 0
                    for chunk in r.iter_content(chunk_size=args.chunk_size):
                        if chunk:
                            if progress_bar:
                                progress_bar.update(len(chunk))
                            downloaded_bytes += len(chunk)
                            temp_zip.write(chunk)
                    
                    if progress_bar:
                        progress_bar.close()
                    
                    # Validate download
                    if total_size_in_bytes != 0 and downloaded_bytes != total_size_in_bytes:
                        print(f"Warning: Download for {release_name} may be incomplete. "
                              f"Expected {total_size_in_bytes} bytes, got {downloaded_bytes} bytes.", 
                              file=sys.stderr)

            except Exception:
                if progress_bar:
                    progress_bar.close()
                if os.path.exists(temp_zip_path):
                    os.unlink(temp_zip_path)
                raise

        # 5. Extract
        if not args.quiet:
            print(f"Extracting {temp_zip_path} to: {final_data_path} ...")
            
        with zipfile.ZipFile(temp_zip_path, 'r') as z:
            z.extractall(final_data_path)

        # 6. Fix nested directory structure
        if not args.quiet:
            print("Checking for nested directory...")
            
        extracted_items = list(final_data_path.iterdir())

        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            unwanted_dir = extracted_items[0]
            if not args.quiet:
                print(f"Detected nested directory: {unwanted_dir.name}. Moving contents up.")
            
            for item_to_move in unwanted_dir.iterdir():
                destination = final_data_path / item_to_move.name
                item_to_move.rename(destination)
                
            unwanted_dir.rmdir()
            if not args.quiet:
                print(f"Successfully moved contents and removed {unwanted_dir.name}.")
        else:
            if not args.quiet:
                print("No single nested directory found. Data is as extracted.")

        # 7. Clean up temporary zip file
        os.unlink(temp_zip_path)
        if not args.quiet:
            print(f"Removed temporary file: {temp_zip_path}")
        temp_zip_path = None

        # 8. Verify
        if final_data_path.exists() and final_data_path.is_dir():
            if not args.quiet:
                print(f"Successfully created: {final_data_path}")
            return True
        else:
            print(f"Warning: Extraction finished, but expected path {final_data_path} was not found.", 
                  file=sys.stderr)
            return False

    except requests.RequestException as e:
        print(f"Error downloading {release_name}: {e}", file=sys.stderr)
    except zipfile.BadZipFile:
        print(f"Error: Downloaded file for {release_name} is not a valid zip file.", file=sys.stderr)
    except KeyError as e:
        print(f"Error: Missing key {e} in structure.json for release '{release_info.get('release_name', 'UNKNOWN')}'", 
              file=sys.stderr)
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
    """Main function to read JSON and process all releases."""
    args = parse_args()
    
    # 1. Determine JSON file path
    if args.json_file:
        json_path = pathlib.Path(args.json_file)
    else:
        # Auto-select based on mode
        if args.mode == "full":
            json_path = pathlib.Path("structure_full.json")
        else:  # mini
            json_path = pathlib.Path("structure_mini.json")
    
    if not json_path.exists():
        print(f"Error: {json_path} not found in {os.getcwd()}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Loading configuration from {json_path.resolve()}")
        print(f"Download mode: {args.mode}")
        print(f"Base storage path: {args.base_path}")
        print(f"Data folder: {args.data_folder}")
    
    # 2. Load JSON configuration
    try:
        with open(json_path, 'r') as f:
            structure = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding {json_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Set up main data directory
    main_data_dir = pathlib.Path(args.base_path) / args.data_folder
    main_data_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.quiet:
        print("--- Starting Dataset Download and Extraction ---")
        print(f"All data will be stored in: {main_data_dir.resolve()}")

    # 4. Process all releases
    releases = structure.get('releases', [])
    if not releases:
        print("No 'releases' found in structure JSON. Nothing to do.", file=sys.stderr)
        sys.exit(1)

    success_count = 0
    fail_count = 0
    
    for release in releases:
        if download_and_extract(release, main_data_dir, args):
            success_count += 1
        else:
            fail_count += 1

    # 5. Final report
    if not args.quiet:
        print("---")
        print("--- Processing Complete ---")
        print(f"Successfully processed/skipped: {success_count}")
        print(f"Failed: {fail_count}")

        if fail_count == 0:
            print("All datasets are ready.")
        else:
            print("Some datasets failed to process. Please check the errors above.")
    
    # Exit with error code if any downloads failed
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()