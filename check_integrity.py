import json
import os
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

def clean_jsonl(jsonl_path, dry_run=False):
    print(f"\nChecking integrity of: {jsonl_path}")
    valid_lines = []
    removed_count = 0
    total_count = 0
    
    # Read and verify
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Scanning entries"):
            total_count += 1
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                video_path = data.get('video')
                
                is_valid = False
                if video_path and os.path.exists(video_path):
                    # Check if file is essentially empty/corruped (e.g. < 1KB)
                    if os.path.getsize(video_path) > 1024:
                        is_valid = True
                
                if is_valid:
                    valid_lines.append(line)
                else:
                    removed_count += 1
            except json.JSONDecodeError:
                removed_count += 1
    
    # Report results
    print(f"Total entries: {total_count}")
    print(f"Valid entries: {len(valid_lines)}")
    print(f"Invalid/Missing references: {removed_count}")
    
    if removed_count > 0:
        if dry_run:
            print("[Dry Run] No changes made.")
        else:
            # Create backup
            backup_path = str(jsonl_path) + ".bak"
            shutil.copy2(jsonl_path, backup_path)
            print(f"Backup created at: {backup_path}")
            
            # Write cleaned content
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                f.writelines(valid_lines)
            print("✓ Cleaned file saved.")
    else:
        print("✓ File is already clean.")

def main():
    parser = argparse.ArgumentParser(description="Check and clean JSONL dataset integrity")
    parser.add_argument("--dir", type=str, default=".", help="Root directory to search for jsonl files")
    parser.add_argument("--dry-run", action="store_true", help="Scan without modifying files")
    args = parser.parse_args()
    
    root_dir = Path(args.dir)
    jsonl_files = list(root_dir.glob("**/*.jsonl"))
    
    if not jsonl_files:
        print(f"No .jsonl files found in {root_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files to check...")
    
    for f in jsonl_files:
        clean_jsonl(f, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
