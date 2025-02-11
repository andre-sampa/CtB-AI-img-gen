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
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig, AutoPipelineForText2Image
from src.check_dependecies import check_dependencies

# MAIN GENERATE IMAGE FUNCTION
def generate_image(
                prompt_alias, 
                team_color, 
                custom_prompt, 
                model_alias="FLUX.1-dev", 
                height=36, 
                width=64, 
                num_inference_steps=2, 
                guidance_scale=2.0, 
                seed=-1, 
                progress=gr.Progress(track_tqdm=True)  # Add progress parameter
            ):
    print("Hello from ctb_local!")

    print("Running debug check...")
    # Debug function to check installed packages
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
        print("Initializing HF TOKEN")
        hf_token = os.environ["HF_TOKEN"]
        print(hf_token)
        print("HF TOKEN:")
        login(token=hf_token)
        print("model_name:")
        print(model_name)
        
        # Use absolute path with leading slash
        model_path = f"models/{model_alias}"  
        print(f"Loading model from local path: {model_path}")
        
        # Debug: Check if the directory exists and list its contents
        if os.path.exists(model_path):
            print("Directory exists. Contents:")
            for item in os.listdir(model_path):
                print(f" - {item}")
        else:
            # print(f"Directory does not exist: {local_path}")
             print(f"Contents of {model_path}:")
            # print(os.listdir("/data"))
        # CHECK FOR TORCH USING CUDA
        print("CHECK FOR TORCH USING CUDA")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print("inside if")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        ######### INITIALIZING CPU PIPE ##########
        print("-----LOADING QUANTA-----")
        ckpt_path = (
            "models/FLUX.1-dev-gguf/flux1-dev-Q2_K.gguf"
        )
        transformer = FluxTransformer2DModel.from_single_file(
            ckpt_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
            
        print("-----INITIALIZING PIPE-----")
        pipe = FluxPipeline.from_pretrained(
            model_path,
            transformer = transformer,
            torch_dtype=torch.bfloat16,
            #torch_dtype=torch.float16,
            #torch_dtype=torch.float32,
            local_files_only=True,
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
            #pipe.enable_model_cpu_offload()  # Use official recommended method  
        
        print(f"-----{pipe_message} PIPE INITIALIZED-----")
        print(f"Using device: {pipe.device}")
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        return None, f"ERROR: Failed to initialize PIPE2. Details: {e}"
    try:
        print("-----SENDING IMG GEN TO PIPE-----")
        print("-----HOLD ON-----")   

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
        output_filename = f"/images/{timestamp}_{model_alias.replace(' ', '_').lower()}_{prompt_alias.replace(' ', '_').lower()}_{team_color.lower()}.png"
        # Save the image using PIL's save method
        image.save(output_filename)
        print(f"File path: {output_filename}")
    except Exception as e:
        print(f"ERROR: Failed to save image. Details: {e}")
    # Return the filename and success message
    return image