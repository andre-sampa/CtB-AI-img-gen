import modal
from src.gradio_interface_modal_example import gradio_interface_modal
from config.config import prompts, models  # Indirect import

# Create the Modal app
app = modal.App("ctb-image-generator")
print("Modal app created.")

# Entry point for local execution
@app.local_entrypoint()
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
                    print(f" {package} is installed. Version: {module.__version__}")
                except ImportError:
                    print(f" {package} is NOT installed.")

        check_dependencies()
        print("Launching Gradio interface...")
        # demo.launch()
        gradio_interface_modal()


