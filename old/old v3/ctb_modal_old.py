# modal_app.py
import modal
#IMPORT gradio_interface
from src.gradio_interface import demo

# Create a Modal app
app = modal.App("ctb-image-generator")

image = (
    modal.Image.debian_slim()  # Start with a lightweight Debian-based image
    .apt_install("git")  # Install system-level dependencies (if needed)
    .pip_install(
        "diffusers",  # For Stable Diffusion
        "transformers",  # For Hugging Face models
        "torch",  # PyTorch
        "accelerate",  # For distributed training/inference
        "gradio",  # For the Gradio interface
        "safetensors",  # For safe model loading
        "pillow",  # For image processing
        "datasets",  # For datasets (if needed)
    )
)

@app.local_entrypoint()
def main():
    with modal.enable_output():
        demo.launch()

if __name__ == "__main__":
    main()