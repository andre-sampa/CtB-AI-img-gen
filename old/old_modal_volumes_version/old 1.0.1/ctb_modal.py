import modal
from src.gradio_interface_modal import demo 
from config.config import prompts, models, api_token  # Indirect import
import gradio as gr
import importlib

#FUNZIONA MA NON SI VEDE L'IMMAGINE!!

#Entry point for local execution
#Create the Modal app
app = modal.App("ctb-image-generator-modal", secrets=[modal.Secret.from_name("huggingface-token")])
@app.local_entrypoint()
def main():
    #with modal.enable_output():
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
        demo.launch()