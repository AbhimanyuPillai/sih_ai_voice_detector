import os
import tarfile
import urllib.request
import glob
import shutil

OUTPUT_DIR = "dataset_root/val/human"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TAR_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
TAR_PATH = "test-clean.tar.gz"

print("Downloading LibriSpeech test-clean archive...")
if not os.path.exists(TAR_PATH):
    urllib.request.urlretrieve(TAR_URL, TAR_PATH)
    print("Download finished.")

print("Extracting 50 sample files...")
extracted_count = 0
with tarfile.open(TAR_PATH, "r:gz") as tar:
    for member in tar.getmembers():
        if member.name.endswith(".flac"):
            tar.extract(member, path="temp_libri")
            extracted_count += 1
            if extracted_count >= 50:
                break

# Move extracted flac files to final folder
flac_files = glob.glob("temp_libri/**/*.flac", recursive=True)
for idx, src_file in enumerate(flac_files[:50]):
    dst_file = os.path.join(OUTPUT_DIR, f"human_libri_{idx+1:03d}.flac")
    shutil.move(src_file, dst_file)

# Cleanup
if os.path.exists("temp_libri"):
    shutil.rmtree("temp_libri")
if os.path.exists(TAR_PATH):
    os.remove(TAR_PATH)

print(f"Extraction complete. 50 human files ready in: {OUTPUT_DIR}")