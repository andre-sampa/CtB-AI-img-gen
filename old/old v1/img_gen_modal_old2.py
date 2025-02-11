#img_gen_modal.py
import modal
import sys
import os
import random
from datetime import datetime
import random
import io
from config.config import models, prompts  # Indirect import
import gradio as gr

volume = modal.Volume.from_name("flux-model-vol")  # Reference your volume

# Define the Modal image
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
    )
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

app = modal.App("ctb-ai-img-gen-mondal", image=image)

f = modal.Function.lookup("ctb-ai-img-gen-mondal", "generate_image")

def generate(prompt_alias, team_color, model_alias, custom_prompt, height=360, width=640, num_inference_steps=20, guidance_scale=2.0, seed=-1):
    import gradio as gr
    try:
        # Generate the image
        image_path, message = f.remote(prompt_alias, team_color, model_alias, custom_prompt, height, width, num_inference_steps, guidance_scale, seed)
        return image_path, message
    except Exception as e:
        return None, f"An error occurred: {e}"

@app.function(
    volumes={"/volume": volume},  # Mount the volume to /volume
    #gpu="T4",
    timeout=600
)
def generate_image(prompt_alias, team_color, model_alias, custom_prompt, height=360, width=640, 
                  num_inference_steps=20, guidance_scale=2.0, seed=-1):
    import torch
    from diffusers import StableDiffusionPipeline

    # Check if the directory exists
    import os
    model_dir = "/volume/FLUX.1-dev"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found at {model_dir}")
    
    # Your image generation code here
    print(f"Model directory found at {model_dir}! Proceeding with image generation...")
    
    # Example: List contents of the directory
    print("Contents of FLUX.1-dev:")
    print(os.listdir(model_dir))

    # Find the selected prompt and model
    try:
        prompt = next(p for p in prompts if p["alias"] == prompt_alias)["text"]
        model_name = next(m for m in models if m["alias"] == model_alias)["name"]
    except StopIteration:
        return None, "ERROR: Invalid prompt or model selected."
    
    # Debug: Check if the model directory exists
    print(f"Debug: Checking if model directory exists: {model_name}")
    if not os.path.exists(model_name):
        return None, f"ERROR: Model directory not found at {model_name}"

    # Initialize the pipeline using the local model
    print("Debug: Loading model...")

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
        torch_dtype=torch.float16,
        use_safetensors=True,
        #variant="fp16"
    )
    pipe.to("cuda")

    # Connect the button to the function
    generate_button.click(
        generate,
        inputs=[prompt_dropdown, team_dropdown, model_dropdown, custom_prompt_input],
        outputs=[output_image, status_text]
    )
    # Generate the image
    try:
        image = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=torch.Generator("cuda").manual_seed(seed)
        ).images[0]
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
    except Exception as e:
        return None, f"ERROR: Failed to generate image. Details: {e}"

    # Save the image with a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_{model_alias.replace(' ', '_').lower()}_{prompt_alias.replace(' ', '_').lower()}_{team_color.lower()}.png"
    try:
        image.save(output_filename)
    except Exception as e:     
        return img_byte_arr, "Image generated successfully!"
    except Exception as e:
        return None, f"ERROR: Failed to generate image. Details: {e}"
    
    return output_filename, "Image generated successfully!"
