# modal_app.py
import modal
#IMPORT gradio_interface
from src.gradio_interface import demo

# Create a Modal app
app = modal.App("ctb-image-generator")

# Define the Modal image
image = modal.Image.debian_slim().pip_install(
    "diffusers", 
    "transformers",
    "torch",
    "accelerate",
    "gradio"
)

@app.local_entrypoint()
def main():
    with modal.enable_output():
        demo.launch()

if __name__ == "__main__":
    main()