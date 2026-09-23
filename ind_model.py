import os
import shutil
import urllib.request
import zipfile

MODEL_URL = "https://alphacephei.com/kaldi/models/vosk-model-small-en-in-0.4.zip"
ZIP_NAME = "model.zip"
TARGET_DIR = "model"
EXTRACTED_FOLDER = "vosk-model-small-en-in-0.4"

# Remove existing model directory or stale zip if present
if os.path.exists(TARGET_DIR):
    print(f"[*] Removing existing '{TARGET_DIR}' folder...")
    shutil.rmtree(TARGET_DIR)

if os.path.exists(ZIP_NAME):
    os.remove(ZIP_NAME)

# Download the Indian English model
print(f"[*] Downloading Indian English Vosk model from AlphaCephei...")
urllib.request.urlretrieve(MODEL_URL, ZIP_NAME)

# Unpack the archive
print("[*] Unzipping model archive...")
with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
    zip_ref.extractall(".")

# Rename extracted folder to standard 'model' folder
if os.path.exists(EXTRACTED_FOLDER):
    os.rename(EXTRACTED_FOLDER, TARGET_DIR)

# Clean up zip file
if os.path.exists(ZIP_NAME):
    os.remove(ZIP_NAME)

print(f"[✓] Setup complete: '{TARGET_DIR}' is now configured with the Indian English model.")