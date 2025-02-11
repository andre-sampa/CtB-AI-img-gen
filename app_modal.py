# app_modal.py 
import gradio as gr
import modal
from config.config import models, models_modal, prompts, api_token  # Direct import
from config.config import prompts, models, models_modal  # Indirect import
#from img_gen import generate_image

print("Hello from gradio_interface_head!")

# Modal remote function synchronously
def generate(cpu_gpu, prompt_dropdown, team_dropdown, model_dropdown, custom_prompt_input):
    # Debug: 
    debug_message = f"Debug: Button clicked! Inputs - Prompt: {prompt_dropdown}, Team: {team_dropdown}, Model: {model_dropdown}, Custom Prompt: {custom_prompt_input}"
    print(debug_message)  # Print to console for debugging
    try:
        # Check for CPU/GPU dropdown option
        if cpu_gpu == "GPU":
            f = modal.Function.from_name("img-gen-modal", "generate_image_gpu")
        else:
            f = modal.Function.from_name("img-gen-modal", "generate_image_cpu")

        # Import the remote function
        image_path, message = f.remote(
                            prompt_dropdown, 
                            team_dropdown, 
                            model_dropdown, 
                            custom_prompt_input,
                            )
        return image_path, message
    except Exception as e:
        return None, f"Error calling generate_image function: {e}"
    
def gradio_interface_modal():
    try:
        with open("config/layout.css", "r") as f:
            custom_css = f.read()
    except FileNotFoundError:
        print("Error: aaa.css not found!")
        custom_css = ""  # Or provide default CSS

    with modal.enable_output():
        #from config.config import prompts, models  # Indirect import
        # Gradio Interface
        with gr.Blocks(
                    css=custom_css
                    ) as demo:
            gr.Markdown("# CtB AI Image Generator - Cloud version (Modal volume)")
            with gr.Row():
                # Set default values for dropdowns
                prompt_dropdown = gr.Dropdown(choices=[p["alias"] for p in prompts], label="Select Prompt", value=prompts[0]["alias"])
                team_dropdown = gr.Dropdown(choices=["Red", "Blue"], label="Select Team", value="Red")
                model_dropdown = gr.Dropdown(choices=[m["alias"] for m in models_modal], label="Select Model", value=models_modal[0]["alias"])
            with gr.Row():
                # Add a text box for custom user input (max 200 characters)
                custom_prompt_input = gr.Textbox(label="Custom Prompt (Optional)", placeholder="Enter additional details (max 200 chars)...", max_lines=1, max_length=200)
            with gr.Row(elem_classes="row-class"):
                cpu_gpu = gr.Dropdown(choices=["CPU", "GPU"], label="Select CPU/GPU", value="GPU")
                generate_button = gr.Button("Generate Image")
            with gr.Row():
                output_image = gr.Image(elem_classes="output-image", label="Generated Image", show_label=False, scale=1)
            with gr.Row():
                status_text = gr.Textbox(label="Status", placeholder="Waiting for input...", interactive=False)
            print("Building cudasdasrer...")

            ##Connect the button to the call_generate function 
            ##had do do it to handle gradio/modal interaction)
            generate_button.click(
            generate,
            inputs=[
                cpu_gpu, 
                prompt_dropdown, team_dropdown, model_dropdown, custom_prompt_input],
            outputs=[output_image, status_text],
            )
    return demo

# Create the demo instance
demo = gradio_interface_modal()

# Only launch if running directly
if __name__ == "__main__":
    with modal.enable_output():
        demo.queue().launch()
