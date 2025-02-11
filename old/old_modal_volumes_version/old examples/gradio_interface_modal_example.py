# gradio_interface.py
import gradio as gr
import modal
from config.config import prompts, models  # Indirect import
#from src.img_gen_modal import generate

print("Hello from gradio_interface_head!")

# Modal remote function synchronously
def generate(prompt_dropdown, team_dropdown, model_dropdown, custom_prompt_input):
    # Debug: Print a message instead of generating an image
    debug_message = f"Debug: Button clicked! Inputs - Prompt: {prompt_dropdown}, Team: {team_dropdown}, Model: {model_dropdown}, Custom Prompt: {custom_prompt_input}"
    print(debug_message)  # Print to console for debugging
    try:
        # Import the remote function
        f = modal.Function.from_name("img-gen-modal-example", "generate_image")
        image_path, message = f.remote(prompt_dropdown, team_dropdown, model_dropdown, custom_prompt_input)
        return image_path, message
    except Exception as e:
        return None, f"An error occurred: {e}"
   

def gradio_interface_modal():
    with modal.enable_output():
        from config.config import prompts, models  # Indirect import
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
                output = gr.Textbox(label="Output")
            with gr.Row():
                output_image = gr.Image(label="Generated Image")
            with gr.Row():
                status_text = gr.Textbox(label="Status", placeholder="Waiting for input...", interactive=False)
            print("Building cudasdasrer...")

            #Connect the button to the call_generate function 
            #had do do it to handle gradio/modal interaction)
            generate_button.click(
            generate,
            inputs=[prompt_dropdown, team_dropdown, model_dropdown, custom_prompt_input],
            outputs=[output_image, status_text]
            )
            demo.launch()





