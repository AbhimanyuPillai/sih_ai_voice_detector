import os
import sys
import shutil
import urllib.request
import zipfile

MODEL_URL = "https://alphacephei.com/kaldi/models/vosk-model-small-en-in-0.4.zip"
ZIP_NAME = "model.zip"
TARGET_DIR = "model"
EXTRACTED_FOLDER = "vosk-model-small-en-in-0.4"

def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, (downloaded / total_size) * 100)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r[*] Downloading model: {percent:.1f}% ({mb_down:.1f} MB / {mb_total:.1f} MB)")
        sys.stdout.flush()

def setup_vosk_model():
    # Remove existing target directory or stale zip if present
    if os.path.exists(TARGET_DIR):
        print(f"[*] Removing existing '{TARGET_DIR}' folder...")
        shutil.rmtree(TARGET_DIR)

    if os.path.exists(ZIP_NAME):
        os.remove(ZIP_NAME)

    # Configure custom opener with standard User-Agent header to avoid HTTP blocks
    print(f"[*] Fetching Indian English Vosk model (~36 MB)...")
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
    urllib.request.install_opener(opener)

    try:
        urllib.request.urlretrieve(MODEL_URL, ZIP_NAME, reporthook=progress_hook)
        print("\n[*] Download complete.")

        print("[*] Unzipping model archive...")
        with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
            zip_ref.extractall(".")

        # Rename extracted directory to standard 'model' directory
        if os.path.exists(EXTRACTED_FOLDER):
            os.rename(EXTRACTED_FOLDER, TARGET_DIR)

        print(f"[✓] Setup complete: '{TARGET_DIR}/' is ready for unified_app.py.")

    except Exception as err:
        print(f"\n[!] Failed to setup model: {err}")
        sys.exit(1)

    finally:
        # Ensure temporary zip is cleaned up
        if os.path.exists(ZIP_NAME):
            os.remove(ZIP_NAME)

if __name__ == "__main__":
    setup_vosk_model()