import modal
from pathlib import Path

# Create or get existing volume
volume = modal.Volume.from_name("flux-model-vol", create_if_missing=True)
MODEL_DIR = Path("/models")

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
)
def download_flux():
    from huggingface_hub import snapshot_download
    
    repo_id = "black-forest-labs/FLUX.1-dev"
    snapshot_download(
        repo_id=repo_id,
        local_dir=MODEL_DIR / repo_id.split("/")[1]
    )
    print(f"FLUX model downloaded to {MODEL_DIR / repo_id.split('/')[1]}")

@app.local_entrypoint()
def main():
    download_flux()