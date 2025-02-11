#img_gen_modal.py
import modal
import random
from datetime import datetime
import random
import io
from config.config import prompts, models  # Indirect import

# Define the Modal image
image = (
    modal.Image.debian_slim(python_version="3.11")  # Base image
    .pip_install(
        "numpy", 
        "pandas",         
        "diffusers",
        "transformers",
        "torch",
        "accelerate",
        "gradio",
        "safetensors",
        "pillow",
    )  # Install Python packages
    .run_commands("echo 'Image build complete!'")  # Run a shell command
)
# Create a Modal app
app = modal.App("img-gen-modal", image=image)
flux_model_vol = modal.Volume.from_name("flux-model-vol")  # Reference your volume

# def on_button_click():
#     f = modal.Function.from_name("functions-app", "message")
#     messageNEW = "Remote call Hello World!"
#     message.remote((messageNEW))
#     #return message.remote((messageNEW))


@app.function(volumes={"/data": flux_model_vol})
def generate(prompt_alias, team_color, model_alias, custom_prompt, height=360, width=640, num_inference_steps=20, guidance_scale=2.0, seed=-1):
    
    # Debug: Print a message when the function starts
    print("Starting main function inside the container...")

    # Import libraries and print their versions
    import numpy as np
    import pandas as pd
    import torch
    import diffusers
    import transformers
    import gradio as gr
    from PIL import Image as PILImage

    print("Hello from img_gen_modal!")
    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)
    print("PyTorch version:", torch.__version__)
    print("Diffusers version:", diffusers.__version__)  # Corrected: Use the library's __version__
    print("Transformers version:", transformers.__version__)  # Corrected: Use the library's __version__
    print("Gradio version:", gr.__version__)
    print("Pillow version:", PILImage.__version__)
    
    def generate_image(prompt_alias, team_color, model_alias, custom_prompt, height=360, width=640, 
                    num_inference_steps=20, guidance_scale=2.0, seed=-1):
        import torch
        from diffusers import StableDiffusionPipeline
        from config.config import prompts, models  # Indirect import

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
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
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
                generator=torch.Generator("cuda").manual_seed(seed)
            ).images[0]
        except Exception as e:
            return None, f"An error occurred ON PIPE: {e}"
        
        #     # Convert PIL image to bytes
        #     img_byte_arr = io.BytesIO()
        #     image.save(img_byte_arr, format='PNG')
        #     img_byte_arr = img_byte_arr.getvalue()
        # except Exception as e:
        #     return None, f"ERROR: Failed to generate image. Details: {e}"

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
        

# Run the function locally (for testing)
@app.local_entrypoint()
def main():
    print("Running the function locally...")
    generate.remote("horse", "blue", "FLUX.1-dev", "bear", height=360, width=640, num_inference_steps=20, guidance_scale=2.0, seed=-1)