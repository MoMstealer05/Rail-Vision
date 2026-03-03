from roboflow import Roboflow
import os

# --- CONFIGURATION ---
# We want the data to land inside your project structure
# So it doesn't clutter your user folder
DESTINATION_DIR = "data/wagon_dataset" 

print(f"🚀 Downloading Wagon-Dataset to '{DESTINATION_DIR}'...")

rf = Roboflow(api_key="XPBTTDDbao71y12YzBcF")
project = rf.workspace("wagoncounting").project("wagon-dataset")
version = project.version(1)

# This downloads it directly into the folder we want
dataset = version.download("yolov8", location=DESTINATION_DIR)

print("✅ Download Complete!")
print(f"   ► Data is located at: {DESTINATION_DIR}")
print(f"   ► Config file is at: {DESTINATION_DIR}/data.yaml")