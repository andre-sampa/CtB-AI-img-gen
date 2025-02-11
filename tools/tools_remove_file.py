# img_gen.py
#img_gen_modal.py
# img_gen.py
# img_gen_modal.py
import modal
import random
import io
from config.config import prompts, models  # Indirect import
import os


CACHE_DIR = "/model_cache"

# Define the Modal image
image = (
    #modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.9")
    modal.Image.debian_slim(python_version="3.9")  # Base image

    .apt_install(
        "git",
    )
    .pip_install(

    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "HF_HOME", "HF_HUB_CACHE": CACHE_DIR
        }
    )
)

# Create a Modal app
app = modal.App("tools-test-dir", image=image)
with image.imports():
    import os
    from datetime import datetime

flux_model_vol = modal.Volume.from_name("flux-model-vol", create_if_missing=True)  # Reference your volume

@app.function(volumes={"/data": flux_model_vol},
              secrets=[modal.Secret.from_name("huggingface-token")],
              #gpu="a100-80gb"
              )
def test_dir():
    
    import os

    flux_model_vol.remove_file("20250130_185339_flux.1-dev_castle_siege_red.png", recursive = False)
    flux_model_vol.remove_file("20250130_185025_flux.1-dev_castle_siege_red.png", recursive = False)
    flux_model_vol.remove_file("20250130_184952_flux.1-dev_castle_siege_red.png", recursive = False)
    flux_model_vol.remove_file("20250130_184323_flux.1-dev_castle_siege_red.png", recursive = False)





