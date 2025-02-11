# img_gen.py
import sys
import os
import random
from huggingface_hub import InferenceClient
from datetime import datetime
from config.config import models, prompts, api_token  # Direct import
import modal 

# Define the Modal image
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.9")
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
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "HF_HOME"
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
    import transformers
    from huggingface_hub import InferenceClient, login 


@app.function(
    secrets=[modal.Secret.from_name("huggingface-token")],
    gpu="t4",
    timeout=600
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

    # if team.lower() == "red":
    #     winning_team_text = " The winning army is dressed in red armor and banners."
    # elif team.lower() == "blue":
    #     winning_team_text = " The winning army is dressed in blue armor and banners."

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

    # Initialize the InferenceClient
    try:
        print ("starting inference")
        print("token:")
        print (api_token)
        client = InferenceClient(model_name, token=api_token)
    except Exception as e:
        return None, f"ERROR: Failed to initialize InferenceClient. Details: {e}"

     #Generate the image
    try:
        image = client.text_to_image(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            seed=seed
        )
    except Exception as e:
        return None, f"ERROR: Failed to generate image. Details: {e}"

    #return prompt  # For testing purposes, return the formatted prompt

    # Save the image with a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_{model_alias.replace(' ', '_').lower()}_{prompt_alias.replace(' ', '_').lower()}_{team_color.lower()}.png"
    try:
        image.save(output_filename)
    except Exception as e:
        return None, f"ERROR: Failed to save image. Details: {e}"

    return output_filename, "Image generated successfully!"