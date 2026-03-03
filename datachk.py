import os
import pandas as pd

# Path to your image folder
dataset_path = r"D:\Rail-Vision-Root\data\wagon_dataset\train\images"
images = os.listdir(dataset_path)

data = []

for img in images:
    if img.endswith(('.png', '.jpg', '.jpeg')):
        # Example metadata logic
        # You can customize 'Class' based on your folder names or filenames
        data.append({
            "Image_Name": img,
            "File_Path": os.path.join(dataset_path, img),
            "Format": img.split('.')[-1],
            "Dataset_Split": "Training",
            "Project": "Rail-Vision"
        })

# Create DataFrame and Save to Excel
df = pd.DataFrame(data)
df.to_excel("RailVision_Dataset_Catalog.xlsx", index=False)
print("✅ Dataset Excel file generated successfully!")