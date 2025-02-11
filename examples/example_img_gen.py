#img_gen_modal.py
import modal
import random
from datetime import datetime
import random
import io
from config.config import prompts, models  # Indirect import
import os
import torch
from huggingface_hub import login
from transformers import AutoTokenizer

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

#flux_model_vol = modal.Volume.from_name("flux-model-vol", create_if_missing=True)  # Reference your volume

@app.function(
        secrets=[modal.Secret.from_name("huggingface-token")],
        #volumes={"/data": flux_model_vol},
        gpu="t4",
        timeout=600
        )

def generate_image():
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

    prompt = "A cat holding a sign that says hello world"
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    image.save("flux-dev.png")

generate_image()

