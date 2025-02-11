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

def generate_image(prompt_alias, team_color, model_alias, custom_prompt, height=360, width=640, 
                  num_inference_steps=20, guidance_scale=2.0, seed=-1):
    import torch
    from diffusers import StableDiffusionPipeline
    
    # Find the selected prompt and model
    try:
        prompt = next(p for p in prompts if p["alias"] == prompt_alias)["text"]
        model_name = next(m for m in models if m["alias"] == model_alias)["name"]
    except StopIteration:
        return None, "ERROR: Invalid prompt or model selected."

    # Determine the enemy color
    enemy_color = "blue" if team_color.lower() == "red" else "red"
    
    # Print the original prompt and dynamic values for debugging
    print("Original Prompt:")
    print(prompt)
    print(f"Enemy Color: {enemy_color}")
    print(f"Team Color: {team_color.lower()}")

    # Format the prompt
    prompt = prompt.format(team_color=team_color.lower(), enemy_color=enemy_color)
    
    # Print the formatted prompt for debugging
    print("\nFormatted Prompt:")
    print(prompt)

    # Append custom prompt if provided
    if custom_prompt and len(custom_prompt.strip()) > 0:
        prompt += " " + custom_prompt.strip()

    # Randomize seed if needed
    if seed == -1:
        seed = random.randint(0, 1000000)

    # Initialize the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        #torch_dtype=torch.float16,
        use_safetensors=True,
        #variant="fp16"
    )
    pipe.to("cpu")

    # Generate the image
    try:
        image = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]
    except Exception as e:
        return None, f"ERROR: Failed to generate image. Details: {e}"

    # Save the image with a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_{model_alias.replace(' ', '_').lower()}_{prompt_alias.replace(' ', '_').lower()}_{team_color.lower()}.png"
    try:
        image.save(output_filename)
    except Exception as e:
        return None, f"ERROR: Failed to save image. Details: {e}"

    return output_filename, "Image generated successfully!"