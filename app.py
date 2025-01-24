import os
import random
from huggingface_hub import InferenceClient
from PIL import Image
import gradio as gr
from datetime import datetime

# Retrieve the Hugging Face token from environment variables
api_token = os.getenv("HF_CTB_TOKEN")

# Debugging: Check if the Hugging Face token is available
if not api_token:
    print("ERROR: Hugging Face token (HF_CTB_TOKEN) is missing. Please set it as an environment variable.")
else:
    print("Hugging Face token loaded successfully.")

# List of models with aliases
models = [
    {"alias": "FLUX.1-dev", "name": "black-forest-labs/FLUX.1-dev"},
    {"alias": "Midjourney", "name": "strangerzonehf/Flux-Midjourney-Mix2-LoRA"}
]

# List of prompts with intense combat
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

# Debugging: Print prompt and model options
print("Prompt Options:", [p["alias"] for p in prompts])
print("Model Options:", [m["alias"] for m in models])

# Function to generate images
def generate_image(prompt_alias, team, model_alias, height, width, num_inference_steps, guidance_scale, seed):
    # Debugging: Check if the token is available
    if not api_token:
        return None, "ERROR: Hugging Face token (HF_CTB_TOKEN) is missing. Please set it as an environment variable."

    # Find the selected prompt and model
    try:
        prompt = next(p for p in prompts if p["alias"] == prompt_alias)["text"]
        model_name = next(m for m in models if m["alias"] == model_alias)["name"]
    except StopIteration:
        return None, "ERROR: Invalid prompt or model selected."

    # Determine the enemy color
    enemy_color = "blue" if team.lower() == "red" else "red"
    prompt = prompt.format(enemy_color=enemy_color)

    if team.lower() == "red":
        prompt += " The winning army is dressed in red armor and banners."
    elif team.lower() == "blue":
        prompt += " The winning army is dressed in blue armor and banners."

    # Randomize the seed if needed
    if seed == -1:
        seed = random.randint(0, 1000000)

    # Initialize the InferenceClient
    try:
        client = InferenceClient(model_name, token=api_token)
    except Exception as e:
        return None, f"ERROR: Failed to initialize InferenceClient. Details: {e}"

    # Generate the image
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

    # Save the image with a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_{model_alias.replace(' ', '_').lower()}_{prompt_alias.replace(' ', '_').lower()}_{team.lower()}.png"
    try:
        image.save(output_filename)
    except Exception as e:
        return None, f"ERROR: Failed to save image. Details: {e}"

    return output_filename, "Image generated successfully!"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# CtB AI Image Generator")
    with gr.Row():
        # Set default values for dropdowns
        prompt_dropdown = gr.Dropdown(choices=[p["alias"] for p in prompts], label="Select Prompt", value=prompts[0]["alias"])
        team_dropdown = gr.Dropdown(choices=["Red", "Blue"], label="Select Team", value="Red")
        #model_dropdown = gr.Dropdown(choices=[m["alias"] for m in models], label="Select Model", value=models[0]["alias"])
    #with gr.Row():
        # Commented-out dialog boxes (can be re-enabled later)
        # height_input = gr.Number(value=360, label="Height")
        # width_input = gr.Number(value=640, label="Width")
        # num_inference_steps_input = gr.Slider(minimum=10, maximum=100, value=20, label="Inference Steps")
        # guidance_scale_input = gr.Slider(minimum=1.0, maximum=20.0, value=2.0, step=0.5, label="Guidance Scale")
        # seed_input = gr.Number(value=-1, label="Seed (-1 for random)")
    with gr.Row():
        generate_button = gr.Button("Generate Image")
    with gr.Row():
        output_image = gr.Image(label="Generated Image")
    with gr.Row():
        status_text = gr.Textbox(label="Status", placeholder="Waiting for input...", interactive=False)

    # Function to handle button click
    def generate(prompt_alias, team, model_alias, height=360, width=640, num_inference_steps=20, guidance_scale=2.0, seed=-1):
        try:
            # Generate the image
            image_path, message = generate_image(prompt_alias, team, model_alias, height, width, num_inference_steps, guidance_scale, seed)
            return image_path, message
        except Exception as e:
            return None, f"An error occurred: {e}"

    # Connect the button to the function
    generate_button.click(
        generate,
        inputs=[prompt_dropdown, team_dropdown, model_dropdown],  # Removed commented-out inputs
        outputs=[output_image, status_text]
    )

# Launch the Gradio app  p
demo.launch()