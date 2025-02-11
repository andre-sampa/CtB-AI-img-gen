import modal
import random
from datetime import datetime
import os
from config.config import models, prompts

# Define the Modal image (same as in modal_app.py)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "diffusers",
    "transformers",
    "torch>=2.0.1",
    "accelerate",
    "gradio",
    "safetensors",
    "pillow",
)

# Create a Modal app
app = modal.App("ctb-ai-img-gen-modal", image=image)

# Define a volume for caching models
volume = modal.Volume.from_name("flux-model-vol")

@app.cls(
    gpu="H100",  # Use H100 GPU for maximum performance
    container_idle_timeout=20 * 60,  # 20 minutes
    timeout=60 * 60,  # 1 hour
    volumes={"/cache": volume},
)
class Model:
    def __init__(self):
        self.device = "cuda"
        self.torch_dtype = torch.bfloat16
        self.model_dir = "/cache/models"

    @modal.enter()
    def setup(self):
        import torch
        from diffusers import StableDiffusionPipeline

        # Load the model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=self.torch_dtype,
            safety_checker=None,
            feature_extractor=None,
        ).to(self.device)

        # Optimize the model
        self.pipe = self.optimize(self.pipe)

    def optimize(self, pipe):
        import torch

        # Fuse QKV projections
        pipe.unet.fuse_qkv_projections()
        pipe.vae.fuse_qkv_projections()

        # Switch memory layout
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)

        # Compile the model
        pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

        return pipe

    @modal.method()
    def generate(self, prompt_alias, team_color, model_alias, custom_prompt):
        import torch
        from diffusers import StableDiffusionPipeline

        # Find the selected prompt and model
        try:
            prompt = next(p for p in prompts if p["alias"] == prompt_alias)["text"]
            model_name = next(m for m in models if m["alias"] == model_alias)["name"]
        except StopIteration:
            return None, "ERROR: Invalid prompt or model selected."

        # Format the prompt
        enemy_color = "blue" if team_color.lower() == "red" else "red"
        prompt = prompt.format(team_color=team_color.lower(), enemy_color=enemy_color)
        if custom_prompt.strip():
            prompt += " " + custom_prompt.strip()

        # Set seed
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)

        # Generate the image
        try:
            image = self.pipe(
                prompt,
                guidance_scale=2.0,
                num_inference_steps=20,
                width=640,
                height=360,
                generator=torch.Generator(self.device).manual_seed(seed)
            ).images[0]

            # Save the image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{timestamp}_{model_alias.replace(' ', '_').lower()}_{prompt_alias.replace(' ', '_').lower()}_{team_color.lower()}.png"
            image.save(output_filename)

            return output_filename, "Image generated successfully!"
        except Exception as e:
            return None, f"ERROR: Failed to generate image. Details: {e}"

# Function to be called from the Gradio interface
def generate(prompt_alias, team_color, model_alias, custom_prompt):
    model = Model()
    return model.generate.remote(prompt_alias, team_color, model_alias, custom_prompt)