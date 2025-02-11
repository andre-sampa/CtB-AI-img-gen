import modal
from src.gradio_interface import gradio_interface
from config.config import prompts, models  # Indirect import

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

        f"git+https://github.com/huggingface/transformers.git"
    )
    .env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "HF_HOME"
    }
)
)


# Create the Modal app
app = modal.App("ctb-image-generator-modal", image=image)
with image.imports():
    import diffusers
    import os
    import gradio
    import torch
    import sentencepiece
    import importlib
print("Modal app created.")

# Entry point for local execution
@app.local_entrypoint()
def main():
    print("Launching Gradio interface...")
    # demo.launch()
    gradio_interface.launch()


