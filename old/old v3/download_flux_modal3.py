import modal
import os
from pathlib import Path

# Create or get existing volume
volume = modal.Volume.from_name("flux-model-vol-3", create_if_missing=True)

# Set model storage directory
MODEL_DIR = "/data/models"

# Set up image with dependencies
download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]", "transformers")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # Enables optimized downloads
)

# Create Modal app
app = modal.App("flux-model-setup")

@app.function(
    volumes={"/data/models": volume},  # Fix volume mounting syntax
    image=download_image,
    secrets=[modal.Secret.from_name("huggingface-token")]
)
def download_flux():
    from huggingface_hub import snapshot_download
    import transformers  # Ensure transformers is available

    repo_id = "black-forest-labs/FLUX.1-dev"
    local_dir = f"{MODEL_DIR}/{repo_id.split('/')[-1]}"  # Store model in /data/models/FLUX.1-dev

    # Download the model without large weight files for efficiency
    snapshot_download(
        repo_id,
        local_dir=local_dir,
        revision="main",  # Define revision explicitly
        ignore_patterns=["*.pt", "*.bin"],  # Skip large model weights
    )

    # Ensure proper caching
