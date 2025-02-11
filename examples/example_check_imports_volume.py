    def check_dependencies():
     import importlib
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

    def check_volume_contents():
        model_path = "/data/FLUX.1-dev"
        if os.path.exists(model_path):
            print(f"Contents of {model_path}:")
            print(os.listdir(model_path))
        else:
            print(f"Model path {model_path} does not exist.")

    check_volume_contents()