import os


from huggingface_hub import snapshot_download
import transformers

repo_id = "black-forest-labs/FLUX.1-dev"
local_dir = "models/FLUX.1-dev"

# **FASTEST METHOD:** Use max_workers for parallel download
print("Calling snapshot_download")
snapshot_download(
    repo_id,
    local_dir=local_dir,
    revision="main",
    #ignore_patterns=["*.pt", "*.bin"],  # Skip large model weights
    max_workers=8  # Higher concurrency for parallel chunk downloads
)
print("Called snapshot_download")


transformers.utils.move_cache()
print(f"Model downloaded to {local_dir}")


