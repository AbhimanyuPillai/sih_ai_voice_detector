import os
import urllib.request
import zipfile
import shutil

MODEL_URL = "https://alphacephei.com/kaldi/models/vosk-model-small-en-us-0.15.zip"
ZIP_NAME = "vosk_model.zip"
TARGET_DIR = "model"

print("Downloading Vosk lightweight English model (~40 MB)...")
urllib.request.urlretrieve(MODEL_URL, ZIP_NAME)
print("Download finished. Extracting...")

with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
    zip_ref.extractall("temp_vosk")

# Locate extracted inner directory and move to final target directory
extracted_folder = [os.path.join("temp_vosk", f) for f in os.listdir("temp_vosk") if os.path.isdir(os.path.join("temp_vosk", f))][0]

if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)

shutil.move(extracted_folder, TARGET_DIR)

# Clean up temporary archives
if os.path.exists("temp_vosk"):
    shutil.rmtree("temp_vosk")
if os.path.exists(ZIP_NAME):
    os.remove(ZIP_NAME)

print(f"Vosk model configured at: {os.path.abspath(TARGET_DIR)}")