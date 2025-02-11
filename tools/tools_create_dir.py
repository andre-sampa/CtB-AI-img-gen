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
    import urllib.request


    # Define the path of the new directory
    new_directory = "FLUX.1-dev-gguf"

    # Create the directory (and parent directories if needed)
    os.makedirs(f"/data/{new_directory}", exist_ok=True)

    url = "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q8_0.gguf"
    urllib.request.urlretrieve(url, "/data/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf")

    print("Download complete!")


    #print(f"Directory created: {new_directory}")

    # Get the current working directory
    current_directory = os.getcwd()
    # List the contents of the current directory
    print("Contents of current modal directory:")
    print(os.listdir(current_directory))

    # VOLUME DIRECTORY
    volume_directory = f"{current_directory}/data/"
    print(f"Current volume directory: {volume_directory}")
    print(os.listdir(volume_directory))


    flux_model_vol.de





