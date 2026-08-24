#!/usr/bin/env python3
"""
Script to download DrugOOD dataset from Google Drive
"""
import os
import sys

def download_drugood_data():
    """Download DrugOOD datasets from Google Drive"""
    
    # Check if gdown is installed
    try:
        import gdown
    except ImportError:
        print("Error: gdown is not installed.")
        print("Please install it with: pip install gdown")
        sys.exit(1)
    
    # Base data directory
    base_dir = os.path.join(os.path.dirname(__file__), "data", "drugood")
    
    # Google Drive folder ID (extracted from the link)
    folder_id = "17nVALCgTz0LV8pVuoM0xQnRqwRH3Bz7a"
    
    print("=" * 60)
    print("DrugOOD Dataset Downloader")
    print("=" * 60)
    print(f"\nDownloading to: {base_dir}")
    print("\nNote: This may take a while depending on your connection...")
    print("\nDownloading from Google Drive folder...")
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Download the entire folder
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    try:
        # Download folder contents
        gdown.download_folder(folder_url, output=base_dir, quiet=False, use_cookies=False)
        print("\n" + "=" * 60)
        print("Download completed successfully!")
        print("=" * 60)
        
        # Check if the assay dataset was downloaded
        assay_raw_dir = os.path.join(base_dir, "lbap_core_ic50_assay", "raw")
        assay_json = os.path.join(assay_raw_dir, "lbap_core_ic50_assay.json")
        
        if os.path.exists(assay_json):
            print(f"\n✓ Assay dataset found at: {assay_json}")
        else:
            print(f"\n⚠ Warning: Assay dataset not found at expected location: {assay_json}")
            print("You may need to manually organize the downloaded files.")
        
        # List what was downloaded
        print(f"\nContents of {base_dir}:")
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
                
    except Exception as e:
        print(f"\n❌ Error downloading: {e}")
        print("\nAlternative: You can manually download from:")
        print("https://drive.google.com/drive/folders/17nVALCgTz0LV8pVuoM0xQnRqwRH3Bz7a")
        print(f"\nThen extract/place the files in: {base_dir}")
        print("\nExpected structure:")
        print("  data/drugood/lbap_core_ic50_assay/raw/lbap_core_ic50_assay.json")
        sys.exit(1)

if __name__ == "__main__":
    download_drugood_data()

