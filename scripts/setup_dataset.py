import os
import zipfile

drive_zip = "/content/drive/MyDrive/phish360_dataset/phish360.zip"
extract_path = "/content/data"

print("Creating dataset folder...")
os.makedirs(extract_path, exist_ok=True)

print("Extracting dataset...")

with zipfile.ZipFile(drive_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset ready!")