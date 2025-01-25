# gradio_interface.py (HuggingFace Spaces)
import gradio as gr
from config.config import prompts, models  # Direct import
from src.img_gen import generate

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

    # Connect the button to the function
    generate_button.click(
        generate,
        inputs=[prompt_dropdown, team_dropdown, model_dropdown, custom_prompt_input],
        outputs=[output_image, status_text]
    )