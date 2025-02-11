import modal
from src.gradio_interface_modal import gradio_interface_modal
from config.config import prompts, models  # Indirect import
# Define the Modal image
image = (
    modal.Image.debian_slim(python_version="3.9")  # Base image

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
        f"git+https://github.com/huggingface/transformers.git"
    )
    .env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "HF_HOME"
    }
)
)
# Create the Modal app
app = modal.App("ctb-image-generator-modal", image = image)
print("Modal app created.")

# Entry point for local execution
@app.function(secrets=[modal.Secret.from_name("huggingface-token")])
def main():
    with modal.enable_output():
        print("Hello from ctb_modal!")
        print("Running debug check...")
        # Debug function to check installed packages
        def check_dependencies():
            packages = [
                "diffusers",  # For Stable Diffusion
                "transformers",  # For Hugging Face models
                "torch",  # PyTorch
                "accelerate",  # For distributed training/inference
                "gradio",  # For the Gradio interface (updated to latest version)
                "safetensors",  # For safe model loading
                "pillow",  # For image processing
                "sentencepiece"
            ]

            for package in packages:
                try:
                    import importlib
                    module = importlib.import_module(package)
                    print(f" {package} is installed. Version:")
                except ImportError:
                    print(f" {package} is NOT installed.")

        check_dependencies()
        print("Launching Gradio interface...")
        # demo.launch()
        gradio_interface_modal()


