# Define the Modal image
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.9")
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
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "HF_HOME"
        }
    )
)

# Create a Modal app
app = modal.App("img-gen-modal", image=image)
with image.imports():
    import diffusers
    import os
    import gradio
    import torch
    import sentencepiece
    import transformers
    from huggingface_hub import InferenceClient, login 


@app.function(
    secrets=[modal.Secret.from_name("huggingface-token")],
    gpu="t4",
    timeout=600
)
def generate_image


# Define the Modal image
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.9")
    #modal.Image.debian_slim(python_version="3.9")  # Base image

    .apt_install(
        "git",
    )
    .pip_install(
        "torch"
    )
)