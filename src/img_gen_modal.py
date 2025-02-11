#img_gen_modal.py
import modal
import random
import io
from config.config import prompts, models_modal  # Indirect import
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
from diffusers.callbacks import SDXLCFGCutoffCallback
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, AutoencoderTiny, AutoencoderKL, DiffusionPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from PIL import Image
from src.check_dependecies import check_dependencies
import numpy as np

#import xformers


from live_preview_helpers import flux_pipe_call_that_returns_an_iterable_of_images

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

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
              timeout = 30000
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
                width=640, 
                height=360, 
                num_inference_steps=20, 
                guidance_scale=2.0, 
                seed=-1, 
                progress=gr.Progress(track_tqdm=True)  # Add progress parameter
            ):
    with modal.enable_output():
        print("Hello from ctb_modal!")

        check_dependencies()

        # Find the selected prompt and model
        try:
            prompt = next(p for p in prompts if p["alias"] == prompt_alias)["text"]
            model_name = next(m for m in models_modal if m["alias"] == model_alias)["name"]
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
            model_path = f"/data/{model_name}"  # Changed from "data/" to "/data/"
            print(f"Loading model from local path: {model_path}")
            
            # Debug: Check if the directory exists and list its contents
            if os.path.exists(model_path):
                print("Directory exists. Contents:")
                for item in os.listdir(model_path):
                    print(f" - {item}")
            else:
                print(f"Directory does not exist: {model_path}")
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
    
            # ########## LIVE PREVIEW FROM REPO DEMO ##########
            # print("-----INITIALIZING LIVE PREVIEW CODE FROM DEMO -----")
            # dtype = torch.bfloat16
            # device = "cuda" if torch.cuda.is_available() else "cpu"

            # taef1 = AutoencoderTiny.from_pretrained("/data/taef1", torch_dtype=dtype).to(device)
            # good_vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device)
            # pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype, vae=taef1).to(device)
            # torch.cuda.empty_cache()
            
            # pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)
            # print("-----INITIALIZING LIVE PREVIEW CODE FROM DEMO PART2-----")
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


            print("-----INITIALIZING PIPE-----")
            pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                #torch_dtype=torch.float16,
                #torch_dtype=torch.float32,
                #vae=taef1,
                local_files_only=True,
            )
            #torch.cuda.empty_cache()

            if torch.cuda.is_available():
                print("CUDA available")
                print("using gpu")
                pipe = pipe.to("cuda")
                pipe_message = "CUDA"
                #pipe.enable_model_cpu_offload()  # official recommended method but is running slower w it

            else:
                print("CUDA not available")
                print("using cpu")
                pipe = pipe.to("cpu")
                pipe_message = "CPU"
            
            
            print(f"-----{pipe_message} PIPE INITIALIZED-----")
            print(f"Using device: {pipe.device}")
        except Exception as e:
            print(f"Detailed error: {str(e)}")
            return None, f"ERROR: Failed to initialize PIPE2. Details: {e}"
        try:
            print("-----SENDING IMG GEN TO PIPE-----")
            print("-----HOLD ON-----")   
            


            # ########## LATENTS ##########
            # # live preview function to get the latents
            # # official reference guideline
            # def latents_to_rgb(latents):
            #     weights = (
            #         (60, -60, 25, -70),
            #         (60,  -5, 15, -50),
            #         (60,  10, -5, -35),
            #     )

            #     weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
            #     biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
            #     rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
            #     image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

            #     return Image.fromarray(image_array)

            # def decode_tensors(pipe, step, timestep, callback_kwargs):
            #     latents = callback_kwargs["latents"]

            #     image = latents_to_rgb(latents[0])
            #     image.save(f"{step}.png")

            #     return callback_kwargs
            # ############################################################




            ########## SENDING IMG GEN TO PIPE - WORKING CODE ##########
            image = pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                max_sequence_length=512,
                #callback_on_step_end=decode_tensors,
                #callback_on_step_end_tensor_inputs=["latents"],
                # seed=seed
            ).images[0]
            #############################################################

            print("-----IMAGE GENERATED SUCCESSFULLY!-----")
            print(image)  
                
        except Exception as e:
            return f"ERROR: Failed to initialize InferenceClient. Details: {e}"
        
        try:
            print("-----SAVING-----")
            # Save the image with a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"/data/images/{timestamp}_{model_alias.replace(' ', '_').lower()}_{prompt_alias.replace(' ', '_').lower()}_{team_color.lower()}.png"
            # Save the image using PIL's save method
            image.save(output_filename)
            print("-----DONE!-----")
            print("-----CALL THE BANNERS!-----")
            print(f"File path: {output_filename}")
        except Exception as e:
            print(f"ERROR: Failed to save image. Details: {e}")
        # Return the filename and success message
        return image