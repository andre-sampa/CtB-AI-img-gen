#img_gen_modal.py
import modal
import random
import io
from config.config import prompts, models  # Indirect import
import os
import gradio as gr

#MOVED FROM IMAGE IMPORT LIST
import torch
import sentencepiece
import torch
from huggingface_hub import login
from transformers import AutoTokenizer
import random
from datetime import datetime
#import xformers

import gradio as gr
import numpy as np
#import spaces
from diffusers import  DiffusionPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderTiny, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
from live_preview_helpers import calculate_shift, retrieve_timesteps, flux_pipe_call_that_returns_an_iterable_of_images


CACHE_DIR = "/model_cache"

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

# Define the Modal image
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.9").pip_install_from_requirements("requirements.txt")
    #modal.Image.debian_slim(python_version="3.9")  # Base image

    # .apt_install(
    #     "git",
    # )
    # .pip_install(
    #     "diffusers",
    #     "transformers",
    #     "xformers",
    #     "torch",
    #     "accelerate",
    #     "gradio>=4.44.1",
    #     "safetensors",
    #     "pillow",
    #     "sentencepiece",
    #     "hf_transfer",
    #     "huggingface_hub[hf_transfer]",
    #     "aria2",  # aria2 for ultra-fast parallel downloads
    #     f"git+https://github.com/huggingface/transformers.git"
    # )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "HF_HOME", "HF_HUB_CACHE": CACHE_DIR
        }
    )
)

# Create a Modal app
app = modal.App("live-preview-test", image=image)
with image.imports():
    import os

flux_model_vol = modal.Volume.from_name("flux-model-vol", create_if_missing=True)  # Reference your volume


# GPU FUNCTION
@app.function(volumes={"/data": flux_model_vol},
              secrets=[modal.Secret.from_name("huggingface-token")],
              gpu="L40S",
              timeout = 300
              )
def infer(prompt, seed=42, randomize_seed=False, width=1024, height=1024, guidance_scale=3.5, num_inference_steps=28, progress=gr.Progress(track_tqdm=True)):

    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    taef1 = AutoencoderTiny.from_pretrained("/data/taef1", torch_dtype=dtype).to(device)
    good_vae = AutoencoderKL.from_pretrained("/data/FLUX.1-dev", subfolder="vae", torch_dtype=dtype).to(device)
    pipe = DiffusionPipeline.from_pretrained("/data/FLUX.1-dev", torch_dtype=dtype, vae=taef1).to(device)
    torch.cuda.empty_cache()

    pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)

    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    
    for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            output_type="pil",
            good_vae=good_vae,
        ):
            yield img, seed
    

