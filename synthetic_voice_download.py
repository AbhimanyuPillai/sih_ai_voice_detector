import os
import shutil
from huggingface_hub import HfApi, hf_hub_download

# Suppress the Windows symlink warning in terminal output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

OUTPUT_DIR = "dataset_root/val/synthetic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPO_ID = "garystafford/deepfake-audio-detection"
TARGET_COUNT = 5

print(f"Connecting to Hugging Face repository '{REPO_ID}'...")
api = HfApi()

# List files inside the repository
all_repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

# Filter files inside the fake/ directory
fake_audio_files = [f for f in all_repo_files if f.startswith("fake/") and f.endswith(".flac")]
print(f"Found {len(fake_audio_files)} synthetic files available.")

download_list = fake_audio_files[:TARGET_COUNT]
print(f"Downloading first {len(download_list)} files into '{OUTPUT_DIR}'...\n")

for idx, file_path in enumerate(download_list, start=1):
    # Direct file download
    cached_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=file_path,
        repo_type="dataset"
    )
    
    # Save into destination folder
    base_name = os.path.basename(file_path)
    destination = os.path.join(OUTPUT_DIR, base_name)
    shutil.copyfile(cached_path, destination)
    

    print(f"[{idx:02d}/{TARGET_COUNT}] Downloaded: {base_name}")

print(f"\nAll {len(download_list)} synthetic audio samples ready in: {OUTPUT_DIR}")