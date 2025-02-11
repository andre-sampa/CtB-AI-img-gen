################# RUN IT W MODAL RUN TO DOWNLOAD ON MODAL VOLUME ###########


import modal
import os

app = modal.App("flux-model-setup")

# Persistent volume for storing models
volume = modal.Volume.from_name("flux-model-vol", create_if_missing=True)  

# Image with dependencies
download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]", "transformers", "aria2")  # aria2 for ultra-fast parallel downloads
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # Enable fast Rust-based downloads
)

@app.function(
    volumes={"/data": volume},
    image=download_image,
    secrets=[modal.Secret.from_name("huggingface-token")]
)
def download_flux():
    from huggingface_hub import snapshot_download
    import transformers

    repo_id = "city96/FLUX.1-dev-gguf"
    local_dir = "/data/FLUX.1-dev-gguf"

    # **FASTEST METHOD:** Use max_workers for parallel download
    snapshot_download(
        repo_id,
        local_dir=local_dir,
        revision="main",
        #ignore_patterns=["*.pt", "*.bin"],  # Skip large model weights
        max_workers=8  # Higher concurrency for parallel chunk downloads
    )

    transformers.utils.move_cache()
    print(f"Model downloaded to {local_dir}")

@app.local_entrypoint()
def main():
    download_flux.remote()
