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
        "sentencepiece",
        "gguf",
    ]

    for package in packages:
        try:
            import importlib
            module = importlib.import_module(package)
            print(f" {package} is installed. Version:")
        except ImportError:
            print(f" {package} is NOT installed.")
