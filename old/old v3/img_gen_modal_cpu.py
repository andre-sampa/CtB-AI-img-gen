# img_gen.py
#img_gen_modal.py
# img_gen.py
# img_gen_modal.py
import modal
import random
import io
from config.config import prompts, models  # Indirect import
import os

CACHE_DIR = "/model_cache"

# Define the Modal image
image = (
    #modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.9")


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

# Create a Modal CPU app
app = modal.App("img-gen-modal-cpu", image=image)
with image.imports():
    import diffusers
    import os
    import gradio
    import torch
    import sentencepiece
    import torch
    from huggingface_hub import login
    from transformers import AutoTokenizer
    import random
    from datetime import datetime

flux_model_vol = modal.Volume.from_name("flux-model-vol", create_if_missing=True)  # Reference your volume


# CPU FUNCTION
@app.function(volumes={"/data": flux_model_vol},
              secrets=[modal.Secret.from_name("huggingface-token")],
              #gpu="a100-80gb",
              cpu = 2,
              memory = 160000,
              timeout=6000
              )
# MAIN GENERATE IMAGE FUNCTION
def generate_image(prompt_alias, team_color, model_alias, custom_prompt, height=360, width=640, num_inference_steps=20, guidance_scale=2.0, seed=-1):
        #with modal.enable_output():
        print("Hello from ctb_modal!")
        print("Running debug check...")
        # Debug function to check installed packages
        def check_dependencies():
            packages = [
                "diffusers",  # For Stable Diffusion
                "transformers",  # For Hugging Face models
                "torch",  # PyTorch
                "accelerate",  # For distributed training/inference
                "gradio",  # For the Gradio interface (updated to latest version)
                "safetensors",  # For safe model loading
                "pillow",  # For image processing
                "sentencepiece"
            ]

            for package in packages:
                try:
                    import importlib
                    module = importlib.import_module(package)
                    print(f" {package} is installed. Version:")
                except ImportError:
                    print(f" {package} is NOT installed.")

        check_dependencies()
        
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

        try:
            from diffusers import FluxPipeline
            print("Initializing HF TOKEN")
            hf_token = os.environ["HF_TOKEN"]
            print(hf_token)
            print("HF TOKEN:")
            login(token=hf_token)
            print("model_name:")
            print(model_name)
            
            # Use absolute path with leading slash
            local_path = f"/data/{model_name}"  # Changed from "data/" to "/data/"
            print(f"Loading model from local path: {local_path}")
            
            # Debug: Check if the directory exists and list its contents
            if os.path.exists(local_path):
                print("Directory exists. Contents:")
                for item in os.listdir(local_path):
                    print(f" - {item}")
            else:
                print(f"Directory does not exist: {local_path}")
                print("Contents of /data:")
                print(os.listdir("/data"))

            # INITIALIZING PIPE
            print("Initializing PIPE2")
            pipe = FluxPipeline.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True
            )
            pipe.enable_model_cpu_offload()  # Use official recommended method
            #pipe = pipe.to("cpu")

        except Exception as e:
            print(f"Detailed error: {str(e)}")
            return None, f"ERROR: Failed to initialize PIPE. Details: {e}"
        try:
            print("Sending img gen to pipe")
            image = pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                # seed=seed
            ).images[0]
            print("render done")
            print(image)           
        except Exception as e:
            return f"ERROR: Failed to initialize InferenceClient. Details: {e}"
        

        try:
            print("SAVING")
            # Save the image with a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"/data/images/{timestamp}_{model_alias.replace(' ', '_').lower()}_{prompt_alias.replace(' ', '_').lower()}_{team_color.lower()}.png"
            # Save the image using PIL's save method
            image.save(output_filename)
            #print(f"Image saved! File path: {output_filename}")
            print("Image generated successfully!")
        except Exception as e:
            print(f"ERROR: Failed to save image. Details: {e}")
        # Return the filename and success message
        return image, "Image generated successfully!"
