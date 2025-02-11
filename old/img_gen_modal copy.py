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


########## LIVE PREVIEW TEST 1/3 ##########
#from live_preview_helpers import flux_pipe_call_that_returns_an_iterable_of_images
###########################################


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
app = modal.App("img-gen-modal", image=image)
with image.imports():
    import os

flux_model_vol = modal.Volume.from_name("flux-model-vol", create_if_missing=True)  # Reference your volume

############ LIVE PREVIEW 2/3 ##################
# dtype = torch.bfloat16
# device = "cuda" if torch.cuda.is_available() else "cpu"

# taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=dtype).to(device)
# good_vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype).to(device)
# pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=dtype, vae=taef1).to(device)
# torch.cuda.empty_cache()

# MAX_SEED = np.iinfo(np.int32).max
# MAX_IMAGE_SIZE = 2048

#pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)
#################################################


# GPU FUNCTION
@app.function(volumes={"/data": flux_model_vol},
              secrets=[modal.Secret.from_name("huggingface-token")],
              gpu="L40S",
              timeout = 300
              )
def generate_image_gpu(prompt_alias, team_color, model_alias, custom_prompt):
    image = generate_image(prompt_alias, team_color, model_alias, custom_prompt)
    return image, "Image generated successfully! Call the banners!"


# CPU FUNCTION
@app.function(volumes={"/data": flux_model_vol},
              secrets=[modal.Secret.from_name("huggingface-token")],
              cpu = 1,
              timeout = 300
              )
def generate_image_cpu(prompt_alias, team_color, model_alias, custom_prompt):
    image = generate_image(prompt_alias, team_color, model_alias, custom_prompt)
    return image, "Image generated successfully! Call the banners!"

# MAIN GENERATE IMAGE FUNCTION
def generate_image(
                prompt_alias, 
                team_color, 
                model_alias, 
                custom_prompt, 
                height=360, 
                width=640, 
                num_inference_steps=20, 
                guidance_scale=2.0, 
                seed=-1, 
                progress=gr.Progress(track_tqdm=True)  # Add progress parameter
            ):
    with modal.enable_output():
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
            # CHECK FOR TORCH USING CUDA
            print("CHECK FOR TORCH USING CUDA")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print("inside if")
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current device: {torch.cuda.current_device()}")
                print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            
            ########## INITIALIZING CPU PIPE ##########

            print("-----INITIALIZING PIPE-----")
            pipe = FluxPipeline.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                #torch_dtype=torch.float16,
                #torch_dtype=torch.float32,
                local_files_only=True
            )
            if torch.cuda.is_available():
                print("CUDA available")
                print("using gpu")
                pipe = pipe.to("cuda")
                pipe_message = "CUDA"
            else:
                print("CUDA not available")
                print("using cpu")
                pipe = pipe.to("cpu")
                pipe_message = "CPU"
                # pipe.enable_model_cpu_offload()  # Use official recommended method  
            
            
            print(f"-----{pipe_message} PIPE INITIALIZED-----")
            print(f"Using device: {pipe.device}")
        except Exception as e:
            print(f"Detailed error: {str(e)}")
            return None, f"ERROR: Failed to initialize PIPE2. Details: {e}"
        try:
            print("-----SENDING IMG GEN TO PIPE-----")
            print("-----HOLD ON-----")   
            


            # ################ LIVE PREVIEW TEST 3/3 ####################
            # seed = random.randint(0, MAX_SEED)
            # generator = torch.Generator().manual_seed(seed)
            
            # for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
            #         prompt=prompt,
            #         guidance_scale=guidance_scale,
            #         num_inference_steps=num_inference_steps,
            #         width=width,
            #         height=height,
            #         generator=generator,
            #         output_type="pil",
            #         good_vae=good_vae,
            #     ):
            #         yield img, seed
            # ############################################################


            ########## SENDING IMG GEN TO PIPE - WORKING CODE ##########
            image = pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                max_sequence_length=512,
                # seed=seed
            ).images[0]
            #############################################################

            print("-----IMAGE GENERATED SUCCESSFULLY!-----")
            print(image)  
                
        except Exception as e:
            return f"ERROR: Failed to initialize InferenceClient. Details: {e}"
        
        try:
            print("-----SAVING-----")
            print("-----DONE!-----")
            print("-----CALL THE BANNERS!-----")
            # Save the image with a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"/data/images/{timestamp}_{model_alias.replace(' ', '_').lower()}_{prompt_alias.replace(' ', '_').lower()}_{team_color.lower()}.png"
            # Save the image using PIL's save method
            image.save(output_filename)
            print(f"File path: {output_filename}")
        except Exception as e:
            print(f"ERROR: Failed to save image. Details: {e}")
        # Return the filename and success message
        return image