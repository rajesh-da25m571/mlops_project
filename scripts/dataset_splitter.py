import os
import re
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configurable Paths
DATASET_DIR = Path(r"D:\mlops_files\invoices_dataset_final\images")
TAMPERED_DIR = Path(r"D:\mlops_files\prepared_dataset\test_evaluation\tampered")
OUTPUT_DIR = Path(r"D:\mlops_files\final_invoice_dataset")  # Target folder
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def parse_filename(file_name: str, parent_folder: str):
    """Extract template identifier from file name or parent folder."""
    match = re.search(r"Template(\d+)", file_name, re.IGNORECASE) or re.search(
        r"Template(\d+)", parent_folder, re.IGNORECASE
    )

    if match:
        template_num = int(match.group(1)) - 1
        template_id_str = f"Template{match.group(1)}"
    else:
        template_num = 0
        template_id_str = "TemplateUnknown"

    return template_num, template_id_str
    
def collect_image_records(dataset_dir: Path) -> pd.DataFrame:
    """Scans dataset directory and builds a Pandas DataFrame."""
    file_records = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in VALID_EXTENSIONS:
                full_path = Path(root) / file
                parent_folder = Path(root).name

                label_int, template_str = parse_filename(file, parent_folder)

                file_records.append(
                    {
                        "filepath": full_path,
                        "label": label_int,
                        "strat_key": template_str,
                    }
                )

    return pd.DataFrame(file_records)
def copy_split_files(df: pd.DataFrame, split_name: str, target_dir: Path) -> None:
    """Copies physical image files to the target split directory."""
    split_path = target_dir / split_name
    split_path.mkdir(parents=True, exist_ok=True)

    print(f"Copying {len(df)} images to '{split_path}'...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name} split"):
        src_path = Path(row["filepath"])
        # Prefix filename with template string to prevent file overwrites
        unique_filename = f"{row['strat_key']}_{src_path.name}"
        dest_path = split_path / unique_filename

        shutil.copy2(src_path, dest_path)


def main():
    print(f"Scanning directory: {DATASET_DIR}")
    df = collect_image_records(DATASET_DIR)
    df_tampered = collect_image_records(TAMPERED_DIR)
    if df.empty:
        raise FileNotFoundError(f"No valid images found in {DATASET_DIR}")

    print(f"Total images found: {len(df)}")
    print(f"Unique templates detected: {df['strat_key'].nunique()}")

    # Handle rare templates with < 2 instances so stratify doesn't crash
    counts = df["strat_key"].value_counts()
    
    rare_templates = counts[counts < 2].index
    if not rare_templates.empty:
        print(f"⚠️ Warning: Replacing rare templates (<2 images) with 'TemplateUnknown' for stratification safety.")
        df.loc[df["strat_key"].isin(rare_templates), "strat_key"] = "TemplateUnknown"
    df_drift = df.loc[df['strat_key'].isin(['Template25','Template50'])]
    df = df.loc[~df['strat_key'].isin(['Template25','Template50'])]
    #print(df.value_counts())
    #sys.exit(1)

    # move tampered dir files and some authentic files to drift folder
    df_dataset, extra_df = train_test_split (df, test_size=0.01, random_state=42, stratify =df['strat_key'])
    df_tampered,df_mini_tampered = train_test_split(df_tampered, test_size=0.1, random_state=42, stratify = df_tampered['strat_key'])
    df_drift = pd.concat([df_drift,extra_df,df_mini_tampered],axis=0) 
    
    # Stratified Split: 80% Train, 20% Temp (Val + Test)
    train1_df, temp_df = train_test_split(
        df_dataset, test_size=0.202, random_state=42, stratify=df_dataset["strat_key"]
    )
    
    # Stratified Split: Split Temp evenly into 15% Val, 15% Test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["strat_key"]
    )
    #train_df, temp_df = train_test_split(
    #    train1_df, test_size=0.90, random_state=42, stratify=train1_df["strat_key"]
    #)

    print("\n--- Split Summary ---")
    print(f"Train size: {len(train1_df)} ({len(train1_df)/len(df):.1%})")
    #print(f"Smaller Train size: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Val size:   {len(val_df)} ({len(val_df)/len(df):.1%})")
    print(f"Test size:  {len(test_df)} ({len(test_df)/len(df):.1%})\n")
    print(f"Drift size:  {len(df_drift)}")
    
    # Save physical files to disk
    copy_split_files(train1_df, "train", OUTPUT_DIR)
    #copy_split_files(train_df, "train_small", OUTPUT_DIR)
    copy_split_files(val_df, "val", OUTPUT_DIR)
    copy_split_files(test_df, "test", OUTPUT_DIR)
    copy_split_files(df_drift, "drift", OUTPUT_DIR)
    copy_split_files(df_tampered, "tampered", OUTPUT_DIR)

    print(f"\n Dataset successfully saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()