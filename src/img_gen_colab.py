# img_gen_logic_colab.py
from huggingface_hub import InferenceClient
from PIL import Image
import random
from datetime import datetime

def generate_image(prompt, team, model_name, height, width, num_inference_steps, guidance_scale, seed, custom_prompt, api_token, randomize_seed=True):
    """
    Generate an image using the Hugging Face Inference API.

    Args:
        prompt (str): The base prompt for image generation.
        team (str): The selected team ("Red" or "Blue").
        model_name (str): The name of the model to use.
        height (int): The height of the generated image.
        width (int): The width of the generated image.
        num_inference_steps (int): The number of inference steps.
        guidance_scale (float): The guidance scale for the model.
        seed (int): The seed for random generation.
        custom_prompt (str): Additional custom prompt text.
        api_token (str): The Hugging Face API token.
        randomize_seed (bool): Whether to randomize the seed.

    Returns:
        PIL.Image.Image or str: The generated image or an error message.
    """
    # Determine the enemy color
    enemy_color = "blue" if team.lower() == "red" else "red"

    if team.lower() == "red":
        winning_team_text = " The winning army is dressed in red armor and banners."
    elif team.lower() == "blue":
        winning_team_text = " The winning army is dressed in blue armor and banners."

    # Print the original prompt and dynamic values for debugging
    print("Original Prompt:")
    print(prompt)
    print(f"Enemy Color: {enemy_color}")
    print(f"Winning Team Text: {winning_team_text}")

    prompt = prompt.format(enemy_color=enemy_color, winning_team_text = winning_team_text)

    # Print the formatted prompt for debugging
    print("\nFormatted Prompt:")
    print(prompt)

    # Append the custom prompt if provided
    if custom_prompt.strip():
        prompt += " " + custom_prompt.strip()

    try:
        # Randomize the seed if the checkbox is checked
        if randomize_seed:
            seed = random.randint(0, 1000000)

        print(f"Using seed: {seed}")

        # Debug: Indicate that the image is being generated
        print("Generating image... Please wait.")

        # Initialize the InferenceClient with the selected model
        client = InferenceClient(model_name, token=api_token)

        # Generate the image using the Inference API with parameters
        image = client.text_to_image(
            prompt,
            guidance_scale=guidance_scale,  # Guidance scale
            num_inference_steps=num_inference_steps,  # Number of inference steps
            width=width,  # Width
            height=height,  # Height
            seed=seed  # Random seed
        )
        return image
    except Exception as e:
        return f"An error occurred: {e}"

def save_image(image, model_label, prompt_label, team):
    """
    Save the generated image with a timestamped filename.

    Args:
        image (PIL.Image.Image): The generated image.
        model_label (str): The label of the selected model.
        prompt_label (str): The label of the selected prompt.
        team (str): The selected team ("Red" or "Blue").

    Returns:
        str: The filename of the saved image.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_{model_label.replace(' ', '_').lower()}_{prompt_label.replace(' ', '_').lower()}_{team.lower()}.png"
    image.save(output_filename)
    return output_filename