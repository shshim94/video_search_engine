import os
from huggingface_hub import snapshot_download

def fetch_mtmc_data():
    print("Initiating connection to NVIDIA Hugging Face repository...")
    
    # Ensure the local data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Download specifically the validation MP4 files from the MTMC 2026 folder
    snapshot_download(
        repo_id="nvidia/PhysicalAI-SmartSpaces",
        repo_type="dataset",
        allow_patterns="MTMC_Tracking_2026/val/*.mp4",
        local_dir="./data",
        local_dir_use_symlinks=False
    )
    
    print("\n✅ NVIDIA MTMC Validation Dataset successfully downloaded to the 'data/' folder.")

if __name__ == "__main__":
    fetch_mtmc_data()
