import modal
import os
from pathlib import Path

# Create or get existing volume
volume = modal.Volume.from_name("flux-model-vol-2", create_if_missing=True)
MODEL_DIR = Path("/data/models")

# Set up image with dependencies
download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # Enable fast Rust download client
)

# Create Modal app
app = modal.App("flux-model-setup")

@app.function(
    volumes={MODEL_DIR: volume},
    image=download_image,
    secrets=[modal.Secret.from_name("huggingface-token")]  # Correct secrets syntax
)
def download_flux():
    from huggingface_hub import snapshot_download
    
    # Get token from environment variable
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in environment variables. Ensure the secret is correctly set.")
    
    repo_id = "black-forest-labs/FLUX.1-dev"
    local_dir = MODEL_DIR / repo_id.split("/")[1]
    
    # Ensure the directory exists
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the model
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token
    )
    print(f"FLUX model downloaded to {local_dir}")

@app.local_entrypoint()
def main():
    download_flux.remote()