# img_gen.py
#img_gen_modal.py
# img_gen.py
# img_gen_modal.py
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

CACHE_DIR = "/model_cache"

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
            "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "HF_HOME", "HF_HUB_CACHE": CACHE_DIR
        }
    )
)

# Create a Modal app
app = modal.App("img-gen-modal-example", image=image)
with image.imports():
    import diffusers
    import os
    import gradio
    import torch
    import sentencepiece

flux_model_vol = modal.Volume.from_name("flux-model-vol", create_if_missing=True)  # Reference your volume

@app.function(
    gpu="t4",  # or "A100" depending on what you're using
    volumes={"/data": flux_model_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=1800  # Add timeout of 30 minutes
)

def generate_image(prompt_alias, team_color, model_alias, custom_prompt, height=50, width=50, num_inference_steps=2, guidance_scale=2.0, seed=-1):
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

    prompt = prompt.format(team_color=team_color.lower(), enemy_color=enemy_color)

    # Print the formatted prompt for debugging
    print("\nFormatted Prompt:")
    print(prompt)

    # Append the custom prompt (if provided)
    if custom_prompt and len(custom_prompt.strip()) > 0:
        prompt += " " + custom_prompt.strip()

    # Randomize the seed if needed
    if seed == -1:
        seed = random.randint(0, 1000000)

    # DOWNLOADING FROM HERE KEEPS THE /MODELS/ DIRECTORY
    # WITH A SCRIPT IT GOES AWAY
    # def download_flux():
    #     from huggingface_hub import snapshot_download
    #     import transformers

    #     repo_id = "black-forest-labs/FLUX.1-schnell"
    #     local_dir = "/data/models/FLUX.1-schnell"

    #     # **FASTEST METHOD:** Use max_workers for parallel download
    #     snapshot_download(
    #         repo_id,
    #         local_dir=local_dir,
    #         revision="main",
    #         #ignore_patterns=["*.pt", "*.bin"],  # Skip large model weights
    #         max_workers=8  # Higher concurrency for parallel chunk downloads
    #     )

    #     transformers.utils.move_cache()
    #     print(f"FLUX model downloaded to {local_dir}")
    # download_flux()

    try:
        from diffusers import FluxPipeline
        print("Initializing HF TOKEN")
        hf_token = os.environ["HF_TOKEN"]
        login(token=hf_token)
        
        local_path = f"/data/{model_name}"
        print(f"Loading model from local path: {local_path}")
        
        # Check model files
        if os.path.exists(local_path):
            print("Directory exists. Contents:")
            required_files = ['model_index.json', 'scheduler', 'vae']
            found_files = os.listdir(local_path)
            print("Found files:", found_files)
            missing_files = [f for f in required_files if f not in found_files]
            if missing_files:
                print(f"Warning: Missing required files: {missing_files}")
        
        print("Initializing pipeline...")
        pipe = FluxPipeline.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            device_map="auto"  # Add this for better GPU utilization
        )
        # Move to GPU and print memory usage
        #pipe = pipe.to("cuda")
        torch.cuda.empty_cache()  # Clear any unused memory
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Error details: {e.__dict__}")
          
    # Verify CUDA availability and model device
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    

    print("Starting image generation...")
    # Set a small test generation first
    test_output = pipe(
        "test",
        num_inference_steps=1,
        width=64,
        height=64
    )
    print("Test generation successful!")
    
    # Now do the actual generation
    print("Starting main generation with parameters:")
    print(f"Prompt: {prompt}")
    print(f"Steps: {num_inference_steps}")
    print(f"Size: {width}x{height}")
    print(f"Guidance scale: {guidance_scale}")
    
    output = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
    )
    
    image = output.images[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"/data/generated_{timestamp}.png"
    
    image.save(output_filename)
    print(f"Image saved to {output_filename}")
    return image, "Success!"

    # # Save the image with a timestamped filename
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_filename = f"{timestamp}_{model_alias.replace(' ', '_').lower()}_{prompt_alias.replace(' ', '_').lower()}_{team_color.lower()}.png"
    # try:
    #     # The pipeline typically returns images in a specific format
    #     # Usually it's image.images[0] for the first generated image
    #     image_output = image.images[0]  # Get the actual PIL Image from the output
    #     image_output.save(output_filename)  # Save using PIL's save method
    # except Exception as e:
    #     return None, f"ERROR: Failed to save image. Details: {e}"
    # print(f"Image output type: {type(image)}")
    # print(f"Image output attributes: {dir(image)}")