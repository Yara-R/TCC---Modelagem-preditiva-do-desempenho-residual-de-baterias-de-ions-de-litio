import os
import yaml
import pandas as pd
from pathlib import Path

# --- CONFIG & SETUP ---
# Check if the configuration file exists.
if not os.path.exists("configs/config.yaml"):
    print("⚠️ config.yaml not found. Using default paths.")
    config = {
        'data_paths': {
            'everlasting': "./data/everlasting",
            'evtol': "./data/eVTOL",
            'forklift': "./data/Lithium-ion battery degradation dataset based on a realistic forklift operation profile",
            'multistage': "./data/Multi-Stage_Aging_Study",
            'nasa': "./data/nasa",
            'oxford': "./data/oxford"
        }
    }
else:
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

# --- HELPER FUNCTIONS ---
def parse_generic(path_obj, source_name):
    """
    Helper to generate a unique ID and metadata dict.
    """
    return f"{source_name}_{path_obj.stem}", {"filename": path_obj.name}

# --- MAIN LOADER ---
def scan_directory(source_name, path_str):
    """
    Scans a directory recursively for battery data files (.csv, .mat).
    """
    root = Path(path_str)
    print(f"🔎 Scanning [{source_name}] at: {root}")
    
    if not root.exists():
        print(f"   ❌ Path not found: {root}")
        return []

    # Recursively find files (case insensitive for extensions)
    files = list(root.rglob("*.csv")) + list(root.rglob("*.CSV")) + list(root.rglob("*.mat"))
    print(f"   found {len(files)} files.")

    data_entries = []
    for p in files:
        name_lower = p.name.lower()
        
        # 1. FILTRO DE SISTEMA (CRÍTICO PARA MAC/LINUX)
        # Ignora arquivos que começam com "." (ex: ._VAH01, .DS_Store)
        if p.name.startswith("."):
            continue

        # 2. Filter out documentation/metadata files
        if "readme" in name_lower or "meta" in name_lower: 
            continue
            
        # 3. FILTER OUT IMPEDANCE FILES (New logic for eVTOL)
        # We don't want these in the main list. The app will find them automatically.
        if "impedance" in name_lower:
            continue
            
        # Create a standardized entry for the master index
        entry = {
            "dataset_source": source_name,
            "cell_id": f"{source_name}_{p.stem}",  # Unique Identifier per cell
            "path": str(p),
            "meta": "{}" 
        }
        data_entries.append(entry)
    
    return data_entries
# def scan_directory(source_name, path_str):
#     """
#     Scans a directory recursively for battery data files (.csv, .mat).
#     """
#     root = Path(path_str)
#     print(f"🔎 Scanning [{source_name}] at: {root}")
    
#     if not root.exists():
#         print(f"   ❌ Path not found: {root}")
#         return []

#     # Recursively find files (case insensitive for extensions)
#     files = list(root.rglob("*.csv")) + list(root.rglob("*.CSV")) + list(root.rglob("*.mat"))
#     print(f"   found {len(files)} files.")

#     data_entries = []
#     for p in files:
#         name_lower = p.name.lower()
        
#         # 1. Filter out documentation/metadata files
#         if "readme" in name_lower or "meta" in name_lower: 
#             continue
            
#         # 2. FILTER OUT IMPEDANCE FILES (New logic for eVTOL)
#         # We don't want these in the main list. The app will find them automatically.
#         if "impedance" in name_lower:
#             continue
            
#         # Create a standardized entry for the master index
#         entry = {
#             "dataset_source": source_name,
#             "cell_id": f"{source_name}_{p.stem}",  # Unique Identifier per cell
#             "path": str(p),
#             "meta": "{}" 
#         }
#         data_entries.append(entry)
    
#     return data_entries

# --- EXECUTION ---
if __name__ == "__main__":
    all_data = []
    
    # Iterate through all data sources defined in config.yaml
    for source, path in config['data_paths'].items():
        entries = scan_directory(source, path)
        all_data.extend(entries)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_data)

    if df.empty:
        print("\n❌ CRITICAL: No files found. Please check your 'data' folder location.")
    else:
        # Ensure column order for consistency
        expected_cols = ["dataset_source", "cell_id", "path", "meta"]
        df = df[expected_cols] 
        
        output_file = "master_battery_index.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Success! Index saved to '{output_file}' with {len(df)} main files (impedance files hidden).")