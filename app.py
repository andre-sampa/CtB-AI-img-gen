
import os
import random
from huggingface_hub import InferenceClient
from PIL import Image
from IPython.display import display, clear_output
import ipywidgets as widgets
from datetime import datetime

# Retrieve the Hugging Face token from Colab secrets
api_token = os.environ.get("HF_CTB_TOKEN")

# List of models with aliases
models = [
    {
        "alias": "FLUX.1-dev",
        "name": "black-forest-labs/FLUX.1-dev"
    },
    {
        "alias": "Stable Diffusion 3.5 turbo",
        "name": "stabilityai/stable-diffusion-3.5-large-turbo"
    },
    {
        "alias": "Midjourney",
        "name": "strangerzonehf/Flux-Midjourney-Mix2-LoRA"
    }
]

# Initialize the InferenceClient with the default model
client = InferenceClient(models[0]["name"], token=api_token)

# List of 10 prompts with intense combat
prompts = [
    {
        "alias": "Castle Siege",
        "text": "A medieval castle under siege, with archers firing arrows from the walls, knights charging on horses, and catapults launching fireballs. The enemy army, dressed in {enemy_color} armor, is fiercely attacking the castle, with soldiers scaling ladders and clashing swords with the defenders. Arrows fly through the air, explosions light up the battlefield, and injured knights lie on the ground. Fire engulfs parts of the castle, and the air is thick with smoke and chaos. Unreal Engine render style, photorealistic, realistic fantasy style."
    },
    {
        "alias": "Forest Battle",
        "text": "A fierce battle between two armies in a dense forest, with knights wielding swords and axes, horses rearing, and the ground covered in mud and blood. The enemy army, dressed in {enemy_color} armor, is locked in brutal combat, with soldiers fighting hand-to-hand amidst the trees. Arrows whiz past, and the sounds of clashing steel echo through the forest. Injured soldiers scream in pain, and the forest is littered with broken weapons and shields. Unreal Engine render style, photorealistic, realistic fantasy style."
    },
    {
        "alias": "Boiling Oil Defense",
        "text": "A dramatic moment in a medieval siege, with a knight leading a charge against a castle gate, while defenders pour boiling oil from the walls. The enemy army, dressed in {enemy_color} armor, is relentlessly attacking, with soldiers screaming as they are hit by the oil. Knights clash swords at the gate, and arrows rain down from above. The ground is littered with the bodies of fallen soldiers, and the air is filled with the smell of burning flesh. Unreal Engine render style, photorealistic, realistic fantasy style."
    },
    {
        "alias": "Burning Castle Battle",
        "text": "A chaotic battlefield with knights on horseback clashing with infantry, archers firing volleys of arrows, and a castle burning in the background. The enemy army, dressed in {enemy_color} armor, is fighting fiercely, with soldiers engaging in brutal melee combat. Flames light up the scene as knights charge through the chaos. Injured soldiers crawl on the ground, and the air is filled with the sounds of clashing steel and screams of pain. Unreal Engine render style, photorealistic, realistic fantasy style."
    },
    {
        "alias": "Heroic Last Stand",
        "text": "A heroic last stand of a small group of knights defending a bridge against a massive army, with arrows flying and swords clashing. The enemy army, dressed in {enemy_color} armor, is overwhelming the defenders, but the knights fight bravely, cutting down enemy soldiers as they advance. The bridge is littered with bodies and broken weapons. Blood stains the ground, and the air is thick with the sounds of battle. Unreal Engine render style, photorealistic, realistic fantasy style."
    },
    {
        "alias": "Siege Tower Attack",
        "text": "A medieval siege tower approaching a castle wall, with knights scaling ladders and defenders throwing rocks and shooting arrows. The enemy army, dressed in {enemy_color} armor, is fighting desperately to breach the walls, with soldiers clashing swords on the battlements. Arrows fly in all directions, and the siege tower is engulfed in flames. Injured soldiers fall from the ladders, and the ground is littered with the bodies of the fallen. Unreal Engine render style, photorealistic, realistic fantasy style."
    },
    {
        "alias": "Knight Duel",
        "text": "A dramatic duel between two knights in the middle of a battlefield, with their armies watching and the castle in the background. The enemy army, dressed in {enemy_color} armor, is engaged in fierce combat all around, with soldiers clashing swords and firing arrows. The duelists fight with skill and determination, their blades flashing in the sunlight. Injured soldiers lie on the ground, and the air is filled with the sounds of battle. Unreal Engine render style, photorealistic, realistic fantasy style."
    },
    {
        "alias": "Night Battle",
        "text": "A night battle during a medieval siege, with torches lighting the scene, knights fighting in the shadows, and the castle walls looming in the background. The enemy army, dressed in {enemy_color} armor, is locked in brutal combat, with soldiers clashing swords and firing arrows in the dim light. Flames from burning siege equipment illuminate the chaos. Injured soldiers scream in pain, and the ground is littered with the bodies of the fallen. Unreal Engine render style, photorealistic, realistic fantasy style."
    },
    {
        "alias": "Marching Army",
        "text": "A massive army of knights and infantry marching towards a distant castle, with banners flying and the sun setting behind them. The enemy army, dressed in {enemy_color} armor, is engaging in skirmishes along the way, with soldiers clashing swords and firing arrows. The battlefield is alive with the sounds of combat and the clash of steel. Injured soldiers lie on the ground, and the air is thick with the smell of blood and smoke. Unreal Engine render style, photorealistic, realistic fantasy style."
    },
    {
        "alias": "Snowy Battlefield",
        "text": "A medieval battle in a snowy landscape, with knights in heavy armor fighting on a frozen lake, and the castle visible in the distance. The enemy army, dressed in {enemy_color} armor, is locked in fierce combat, with soldiers slipping on the ice as they clash swords. Arrows fly through the air, and the snow is stained red with blood. Injured soldiers crawl on the ground, and the air is filled with the sounds of battle. Unreal Engine render style, photorealistic, realistic fantasy style."
    }
]

# Dropdown menu for model selection
model_dropdown = widgets.Dropdown(
    options=[(model["alias"], model["name"]) for model in models],
    description="Select Model:",
    style={"description_width": "initial"}
)

# Dropdown menu for prompt selection
prompt_dropdown = widgets.Dropdown(
    options=[(prompt["alias"], prompt["text"]) for prompt in prompts],
    description="Select Prompt:",
    style={"description_width": "initial"}
)

# Dropdown menu for team selection
team_dropdown = widgets.Dropdown(
    options=["Red", "Blue"],
    description="Select Team:",
    style={"description_width": "initial"}
)

# Input for height
height_input = widgets.IntText(
    value=360,
    description="Height:",
    style={"description_width": "initial"}
)

# Input for width
width_input = widgets.IntText(
    value=640,
    description="Width:",
    style={"description_width": "initial"}
)

# Input for number of inference steps
num_inference_steps_input = widgets.IntSlider(
    value=20,
    min=10,
    max=100,
    step=1,
    description="Inference Steps:",
    style={"description_width": "initial"}
)

# Input for guidance scale
guidance_scale_input = widgets.FloatSlider(
    value=2,
    min=1.0,
    max=20.0,
    step=0.5,
    description="Guidance Scale:",
    style={"description_width": "initial"}
)

# Input for seed
seed_input = widgets.IntText(
    value=random.randint(0, 1000000),
    description="Seed:",
    style={"description_width": "initial"}
)

# Checkbox to randomize seed
randomize_seed_checkbox = widgets.Checkbox(
    value=True,
    description="Randomize Seed",
    style={"description_width": "initial"}
)

# Button to generate image
generate_button = widgets.Button(
    description="Generate Image",
    button_style="success"
)

# Output area to display the image
output = widgets.Output()

# Function to generate images based on the selected prompt, team, and model
def generate_image(prompt, team, model_name, height, width, num_inference_steps, guidance_scale, seed):
    # Determine the enemy color
    enemy_color = "blue" if team.lower() == "red" else "red"
    
    # Replace {enemy_color} in the prompt
    prompt = prompt.format(enemy_color=enemy_color)
    
    if team.lower() == "red":
        prompt += " The winning army is dressed in red armor and banners."
    elif team.lower() == "blue":
        prompt += " The winning army is dressed in blue armor and banners."
    else:
        return "Invalid team selection. Please choose 'Red' or 'Blue'."

    try:
        # Randomize the seed if the checkbox is checked
        if randomize_seed_checkbox.value:
            seed = random.randint(0, 1000000)
            seed_input.value = seed  # Update the seed input box

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

# Function to handle button click event
def on_generate_button_clicked(b):
    with output:
        clear_output(wait=True)  # Clear previous output
        selected_prompt = prompt_dropdown.value
        selected_team = team_dropdown.value
        selected_model = model_dropdown.value
        height = height_input.value
        width = width_input.value
        num_inference_steps = num_inference_steps_input.value
        guidance_scale = guidance_scale_input.value
        seed = seed_input.value

        # Debug: Show selected parameters
        print(f"Selected Model: {model_dropdown.label}")
        print(f"Selected Prompt: {prompt_dropdown.label}")
        print(f"Selected Team: {selected_team}")
        print(f"Height: {height}")
        print(f"Width: {width}")
        print(f"Inference Steps: {num_inference_steps}")
        print(f"Guidance Scale: {guidance_scale}")
        print(f"Seed: {seed}")

        # Generate the image
        image = generate_image(selected_prompt, selected_team, selected_model, height, width, num_inference_steps, guidance_scale, seed)

        if isinstance(image, str):
            print(image)
        else:
            # Debug: Indicate that the image is being displayed and saved
            print("Image generated successfully!")
            print("Displaying image...")

            # Display the image in the notebook
            display(image)

            # Save the image with a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{timestamp}_{model_dropdown.label.replace(' ', '_').lower()}_{prompt_dropdown.label.replace(' ', '_').lower()}_{selected_team.lower()}.png"
            print(f"Saving image as {output_filename}...")
            image.save(output_filename)
            print(f"Image saved as {output_filename}")

# Attach the button click event handler
generate_button.on_click(on_generate_button_clicked)

# Display the widgets
#display(model_dropdown, prompt_dropdown, team_dropdown, height_input, width_input, num_inference_steps_input, guidance_scale_input, seed_input, randomize_seed_checkbox, generate_button, output)

display(prompt_dropdown, team_dropdown, generate_button, output)