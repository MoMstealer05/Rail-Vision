import os
import pandas as pd

# CONFIG: Point this to your actual dataset folder
DATASET_ROOT = r"D:\Rail-Vision-Root\data\wagon_dataset"

def count_files(path, is_label_folder=False):
    if not os.path.exists(path): return 0
    
    # FIX: If we are counting labels, look for .txt files!
    if is_label_folder:
        valid_exts = ('.txt')
    else:
        valid_exts = ('.jpg', '.png', '.jpeg')
        
    return len([f for f in os.listdir(path) if f.endswith(valid_exts)])

stats = {
    "Split": ["Training", "Validation", "Test"],
    "Image Count": [
        count_files(os.path.join(DATASET_ROOT, "train", "images"), is_label_folder=False),
        count_files(os.path.join(DATASET_ROOT, "valid", "images"), is_label_folder=False),
        count_files(os.path.join(DATASET_ROOT, "test", "images"),  is_label_folder=False)
    ],
    "Label Count": [
        count_files(os.path.join(DATASET_ROOT, "train", "labels"), is_label_folder=True),
        count_files(os.path.join(DATASET_ROOT, "valid", "labels"), is_label_folder=True),
        count_files(os.path.join(DATASET_ROOT, "test", "labels"),  is_label_folder=True)
    ]
}

df = pd.DataFrame(stats)
# Add a total row for professional look
df.loc['Total'] = df.sum(numeric_only=True)

output_file = "1_Dataset_Statistics.xlsx"
df.to_excel(output_file, index=False)
print(f"✅ Success! Corrected stats saved to: {output_file}")
print("\n--- PREVIEW ---")
print(df)