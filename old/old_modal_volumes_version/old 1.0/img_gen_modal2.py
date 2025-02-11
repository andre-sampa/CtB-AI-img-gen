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
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.9"
    )
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
app = modal.App("img-gen-modal", image=image)
with image.imports():
    import diffusers
    import os
    import gradio
    import torch
    import sentencepiece

flux_model_vol = modal.Volume.from_name("flux-model-vol", create_if_missing=True)  # Reference your volume

@app.function(gpu="t4", volumes={"/models": flux_model_vol},
              secrets=[modal.Secret.from_name("huggingface-token")],
              # gpu="a100-80gb"
              )
def generate_image(prompt_alias, team_color, model_alias, custom_prompt, height=360, width=640, num_inference_steps=20, guidance_scale=2.0, seed=-1):
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
        print(hf_token)
        print("HF TOKEN:")
        login(token=hf_token)
        print("model_name:")
        print(model_name)
        # First check if model exists in the volume
        local_path = "models/" + model_name
        print(f"Loading model from local path: {local_path}")
        # Debug: Check if the directory exists and list its contents
        for item in os.listdir(local_path):
                print(f" - {item}")

        print("Initializing PIPE")
        # Initialize the pipeline
        #cache_dir = "/cache_"
        pipe = FluxPipeline.from_pretrained("data/" + model_name, torch_dtype=torch.bfloat16,local_files_only=True,
            #cache_dir=cache_dir
        )
        pipe = pipe.to("cuda")

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
        image.save("image.png")

    except Exception as e:
        return None, f"ERROR: Failed to generate image. Details: {e}"

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