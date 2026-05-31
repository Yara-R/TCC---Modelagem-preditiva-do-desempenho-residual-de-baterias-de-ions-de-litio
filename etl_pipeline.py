import os
import yaml
import pandas as pd
from pathlib import Path

# --- CONFIG & SETUP ---
if not os.path.exists("configs/config.yaml"):
    print("⚠️ config.yaml not found. Using default paths tailored to your folder structure.")
    config = {
        'data_paths': {
            'everlasting': "./data/everlasting",
            'evtol': "./data/eVTOL",
            'forklift': "./data/Forklift",
            'multistage': "./data/Multi-Stage",
            'nasa': "./data/NASA",
            'oxford': "./data/Oxford"
        }
    }
else:
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

def scan_directory(source_name, path_str):
    root = Path(path_str)
    print(f"🔎 Scanning [{source_name}] at: {root}")
    
    if not root.exists():
        print(f"   ❌ Path not found: {root}")
        return []

    files = list(root.rglob("*.csv")) + list(root.rglob("*.CSV")) + list(root.rglob("*.mat"))
    print(f"   found {len(files)} files.")

    data_entries = []
    for p in files:
        name_lower = p.name.lower()
        
        if p.name.startswith("."):
            continue

        if "readme" in name_lower or "meta" in name_lower: 
            continue
            
        if "impedance" in name_lower:
            continue
            
        # Build cell_id including subfolder(s) relative to root, so files
        # nested inside subdirectories (e.g. Forklift/Ageing/data.csv) get
        # a unique, human-readable ID like "forklift_Ageing_data".
        relative_parts = p.relative_to(root).parts  # e.g. ('Ageing', 'data.csv')
        if len(relative_parts) > 1:
            # join all parent folder names + stem, skip the filename itself
            cell_id = f"{source_name}_" + "_".join(list(relative_parts[:-1]) + [p.stem])
        else:
            cell_id = f"{source_name}_{p.stem}"

        entry = {
            "dataset_source": source_name,
            "cell_id": cell_id,
            "path": str(p),
            "meta": "{}" 
        }
        data_entries.append(entry)
    
    return data_entries

if __name__ == "__main__":
    all_data = []
    for source, path in config['data_paths'].items():
        entries = scan_directory(source, path)
        all_data.extend(entries)

    df = pd.DataFrame(all_data)

    if df.empty:
        print("\n❌ CRITICAL: No files found. Please check if your scripts are inside the main folder alongside 'data/'.")
    else:
        expected_cols = ["dataset_source", "cell_id", "path", "meta"]
        df = df[expected_cols] 
        output_file = "master_battery_index.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Success! Index saved to '{output_file}' with {len(df)} main files.")