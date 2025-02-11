import modal
import os

app = modal.App("flux-model-setup")

# Attach the newly created volume
volume = modal.Volume.from_name("flux-model-vol")

@app.function(
    volumes={"/data/models": volume},  # Mount the volume inside the container
    image=modal.Image.debian_slim().pip_install("huggingface_hub[hf_transfer]", "transformers"),
    secrets=[modal.Secret.from_name("huggingface-token")]
)
def download_flux():
    from huggingface_hub import snapshot_download
    import transformers  # Ensure transformers is available

    repo_id = "black-forest-labs/FLUX.1-dev"
    local_dir = "/data/models/FLUX.1-dev"  # Store model inside mounted volume

    snapshot_download(
        repo_id,
        local_dir=local_dir,
        revision="main",
        ignore_patterns=["*.pt", "*.bin"]  # Skip large model weights
    )

    transformers.utils.move_cache()
    print(f"FLUX model downloaded to {local_dir}")

@app.local_entrypoint()
def main():
    download_flux.remote()
