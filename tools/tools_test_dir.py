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
        "diffusers",
        "transformers",
        "torch",
        "accelerate",
        "gradio>=4.44.1",
        "safetensors",
        "pillow",
        "sentencepiece",
        "hf_transfer",
        "huggingface_hub[hf_transfer]",
        "aria2",  # aria2 for ultra-fast parallel downloads
        f"git+https://github.com/huggingface/transformers.git"
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
    import diffusers
    import os
    import gradio
    from datetime import datetime

flux_model_vol = modal.Volume.from_name("flux-model-vol", create_if_missing=True)  # Reference your volume

@app.function(volumes={"/data": flux_model_vol},
              secrets=[modal.Secret.from_name("huggingface-token")],
              #gpu="a100-80gb"
              )
def test_dir():
    
    #with modal.enable_output():
    print("Hello from TEST DIR!")
 
    #import os
    # Get the current working directory
    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")

    # List the contents of the current directory
    print("Contents of current directory:")
    print(os.listdir(current_directory))

    # Define the Modal volume path (replace with your actual volume mount path)
    modal_volume_path = "/data/"

    # Check if the Modal volume path exists
    if os.path.exists(modal_volume_path):
        print(f"Contents of Modal volume at {modal_volume_path}:")
        print(os.listdir(modal_volume_path))
    else:
        print(f"Modal volume path {modal_volume_path} does not exist.")