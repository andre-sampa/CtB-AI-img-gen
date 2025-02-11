import gradio as gr
import numpy as np
import random
import torch
from diffusers import  DiffusionPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderTiny, AutoencoderKL
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


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR = "/model_cache"

image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.9")
    .pip_install_from_requirements("requirements.txt")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "HF_HOME", "HF_HUB_CACHE": CACHE_DIR
    })
)

app = modal.App("img-gen-modal-live", image=image)
with image.imports():
    import os

flux_model_vol = modal.Volume.from_name("flux-model-vol", create_if_missing=True)

@app.function(volumes={"/data": flux_model_vol},
              secrets=[modal.Secret.from_name("huggingface-token")],
              gpu="L40S",
              timeout=300)
def infer(prompt, seed=42, randomize_seed=False, width=640, height=360, guidance_scale=3.5, num_inference_steps=28, progress=gr.Progress(track_tqdm=True)):
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

examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

hf_token = os.environ["HF_TOKEN"]
print("Initializing HF TOKEN")
print(hf_token)
print("HF TOKEN:")
login(token=hf_token)

with gr.Blocks(css=css) as demo:
    f = modal.Function.from_name("img-gen-modal-live", "infer")
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX.1 [dev]
12B param rectified flow transformer guidance-distilled from [FLUX.1 [pro]](https://blackforestlabs.ai/)  
[[non-commercial license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)] [[blog](https://blackforestlabs.ai/announcing-black-forest-labs/)] [[model](https://huggingface.co/black-forest-labs/FLUX.1-dev)]
        """)
        
        with gr.Row():
            prompt = gr.Text(label="Prompt", show_label=False, max_lines=1, placeholder="Enter your prompt", container=False)
            run_button = gr.Button("Run", scale=0)
        
        result = gr.Image(label="Result", show_label=False)
        
        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=640)
                height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=360)
            
            with gr.Row():
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=15, step=0.1, value=3.5)
                num_inference_steps = gr.Slider(label="Number of inference steps", minimum=1, maximum=50, step=1, value=28)
        
        gr.Examples(
            examples=examples,
            fn=f.remote,
            inputs=[prompt],
            outputs=[result, seed],
            cache_examples="lazy"
        )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=lambda *args: [next(f.remote_gen(*args)), seed],  # Adjusted to process generator
        inputs=[prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs=[result, seed]
    )

demo.launch()
