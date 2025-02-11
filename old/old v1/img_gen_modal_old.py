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

app = modal.App("ctb-ai-img-gen-mondal")


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# CtB AI Image Generator")
    with gr.Row():
         # Set default values for dropdowns
        prompt_dropdown = gr.Dropdown(choices=[p["alias"] for p in prompts], label="Select Prompt", value=prompts[0]["alias"])
        team_dropdown = gr.Dropdown(choices=["Red", "Blue"], label="Select Team", value="Red")
        model_dropdown = gr.Dropdown(choices=[m["alias"] for m in models], label="Select Model", value=models[0]["alias"])
    with gr.Row():
            # Add a text box for custom user input (max 200 characters)
        custom_prompt_input = gr.Textbox(label="Custom Prompt (Optional)", placeholder="Enter additional details (max 200 chars)...", max_lines=1, max_length=200)
    with gr.Row():
        generate_button = gr.Button("Generate Image")
    with gr.Row():
        output_image = gr.Image(label="Generated Image")
    with gr.Row():
            status_text = gr.Textbox(label="Status", placeholder="Waiting for input...", interactive=False)


@app.function(
    volumes={"/volume": volume},  # Mount the volume to /volume
    #gpu="T4",
    timeout=600
)
def generate(prompt_alias, team_color, model_alias, custom_prompt, height=360, width=640, num_inference_steps=20, guidance_scale=2.0, seed=-1):
    import gradio as gr
    try:
        # Generate the image
        image_path, message = generate_image(prompt_alias, team_color, model_alias, custom_prompt, height, width, num_inference_steps, guidance_scale, seed)
        return image_path, message
    except Exception as e:
        return None, f"An error occurred: {e}"

def generate_image(prompt_alias, team_color, model_alias, custom_prompt, height=360, width=640, 
                  num_inference_steps=20, guidance_scale=2.0, seed=-1):
    import torch
    from diffusers import StableDiffusionPipeline
    
    # Debug: Check if the volume is mounted correctly
    print("Debug: Checking volume contents...")
    try:
        volume_contents = os.listdir("/volume")
        print(f"Debug: Volume contents: {volume_contents}")
    except Exception as e:
        print(f"Debug: Error checking volume contents: {e}")

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
        variant="fp16"
    )
    pipe.to("cpu")

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
