import modal
import random
from datetime import datetime
import io
import os
from config.config import models, prompts

volume = modal.Volume.from_name("flux-model-vol")

# Define the Modal image
image = (modal.Image.debian_slim(python_version="3.9")
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "diffusers",  # For Stable Diffusion
        "transformers",  # For Hugging Face models
        "torch>=2.0.1",  # PyTorch with a minimum version
        "accelerate",  # For distributed training/inference
        "gradio",  # For the Gradio interface
        "safetensors",  # For safe model loading
        "pillow",  # For image processing
        "datasets",  # For datasets (if needed)
    )
)
with image.imports():
    import diffusers
    import torch
    from fastapi import Response

app = modal.App("ctb-ai-img-gen-modal", image=image)

@app.local_entrypoint()
def generate_image(prompt_alias, team_color, model_alias, custom_prompt, height=360, width=640, num_inference_steps=20, guidance_scale=2.0, seed=-1):
        import torch
        from diffusers import StableDiffusionPipeline
        # Debug function to check installed packages


def check_dependencies():
    import importlib 
    # Load the pipeline
    self.model_dir = model_dir
    self.device = "cuda"
    self.torch_dtype = torch.float16

    #@modal.method()
    def run(
        self,
        prompt_alias: str,
        team_color: str,
        model_alias: str,
        custom_prompt: str,
        height: int = 360,
        width: int = 640,
        num_inference_steps: int = 20,
        guidance_scale: float = 2.0,
        seed: int = -1,
    ) -> tuple[str, str]:
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
        
        # Format the prompt
        prompt = prompt.format(team_color=team_color.lower(), enemy_color=enemy_color)
        
        # Append custom prompt if provided
        if custom_prompt and len(custom_prompt.strip()) > 0:
            prompt += " " + custom_prompt.strip()

        # Set seed
        seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
        print("seeding RNG with", seed)
        torch.manual_seed(seed)

        # Load the pipeline
        model_path = os.path.join(self.model_dir, model_name)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            safety_checker=None,  # Disable safety checker
            feature_extractor=None,  # Disable feature extractor
        ).to(self.device)

        # Generate the image
        try:
            image = pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                generator=torch.Generator(self.device).manual_seed(seed)
            ).images[0]
            
            # Save the image with a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{timestamp}_{model_alias.replace(' ', '_').lower()}_{prompt_alias.replace(' ', '_').lower()}_{team_color.lower()}.png"
            image.save(output_filename)
            
            return output_filename, "Image generated successfully!"
        except Exception as e:
            return None, f"ERROR: Failed to generate image. Details: {e}"
        
# Function to be called from the Gradio interface
def generate(prompt_alias, team_color, model_alias, custom_prompt, height=360, width=640, num_inference_steps=20, guidance_scale=2.0, seed=-1):
    try:
        # Generate the image
        image_path, message = generate_image(prompt_alias, team_color, model_alias, custom_prompt, height, width, num_inference_steps, guidance_scale, seed)
        return image_path, message
    except Exception as e:
        return None, f"An error occurred: {e}"
