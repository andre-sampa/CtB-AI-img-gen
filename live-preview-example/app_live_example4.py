import gradio as gr
import numpy as np
import random
import torch
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderTiny, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
from live_preview_helpers import calculate_shift, retrieve_timesteps, flux_pipe_call_that_returns_an_iterable_of_images
import modal
import random
import io
from config.config import prompts, models  # Indirect import
import os
import sentencepiece
from huggingface_hub import login
from transformers import AutoTokenizer
from datetime import datetime
from PIL import Image



CACHE_DIR = "/model_cache"

# Define the Modal image
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.9")
                .pip_install_from_requirements("requirements.txt")
    #modal.Image.debian_slim(python_version="3.9")  # Base image
    # .apt_install(
    #     "git",
    # )
    # .pip_install(
    #     "diffusers",
    #     f"git+https://github.com/huggingface/transformers.git"
    # )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "HF_HOME", "HF_HUB_CACHE": CACHE_DIR
        }
    )
)

# Create a Modal app
app = modal.App("img-gen-modal-live", image=image)
with image.imports():
    import os

flux_model_vol = modal.Volume.from_name("flux-model-vol", create_if_missing=True)  # Reference your volume


# GPU FUNCTION
@app.function(volumes={"/data": flux_model_vol},
              secrets=[modal.Secret.from_name("huggingface-token")],
              gpu="L40S",
              timeout = 300
              )
def main():
    
    def latents_to_rgb(latents):
        weights = (
            (60, -60, 25, -70),
            (60,  -5, 15, -50),
            (60,  10, -5, -35),
        )

        weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
        biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
        rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
        image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

        return Image.fromarray(image_array)

    def decode_tensors(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]

        image = latents_to_rgb(latents[0])
        image.save(f"{step}.png")

        return callback_kwargs
    model_name = "FLUX.1-dev"
    model_path = f"/data/{model_name}" 

    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    ).to("cuda")

    image = pipeline(
        prompt="A croissant shaped like a cute bear.",
        negative_prompt="Deformed, ugly, bad anatomy",
        width=300,
        height=200,
        callback_on_step_end=decode_tensors,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]
    
   