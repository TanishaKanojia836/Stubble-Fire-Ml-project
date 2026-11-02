import os
import glob
import pandas as pd

# 1. Search recursively across modis, modis-2, modis-3... for India CSV files
base_dir = "."
india_files = glob.glob(os.path.join(base_dir, "**", "*India*.csv"), recursive=True)

if not india_files:
    india_files = glob.glob(os.path.join(base_dir, "**", "*india*.csv"), recursive=True)

print(f"Found {len(india_files)} India CSV files:")
for f in sorted(india_files):
    print("  ->", f)

if not india_files:
    raise FileNotFoundError("Could not find any *India*.csv files inside your modis folders. Please check file names.")

# 2. Merge all year-wise India CSVs
df_list = []
for file_path in sorted(india_files):
    temp_df = pd.read_csv(file_path)
    temp_df.columns = temp_df.columns.str.lower()
    df_list.append(temp_df)

raw_df = pd.concat(df_list, ignore_index=True)
print(f"\nTotal raw records across all years: {len(raw_df)}")

# 3. Filter spatial bounding box (Punjab & Haryana) + Stubble months (Oct & Nov)
raw_df['acq_date'] = pd.to_datetime(raw_df['acq_date'])

clean_df = raw_df[
    (raw_df['latitude'].between(27.6, 32.5)) &
    (raw_df['longitude'].between(73.8, 77.6)) &
    (raw_df['acq_date'].dt.month.isin([10, 11]))
].copy()

# Remove low-confidence noise if column exists
if 'confidence' in clean_df.columns:
    if pd.api.types.is_numeric_dtype(clean_df['confidence']):
        clean_df = clean_df[clean_df['confidence'] >= 50]
    else:
        clean_df = clean_df[clean_df['confidence'].str.lower().isin(['n', 'h', 'nominal', 'high'])]

# Save filtered master dataset
output_file = "punjab_haryana_stubble_2020_2025.csv"
clean_df.to_csv(output_file, index=False)
print(f"Successfully saved clean master file: '{output_file}'!")