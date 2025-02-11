from src.gradio_interface import demo
import modal


# Define the Modal image
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "diffusers",  # For Stable Diffusion
        "transformers",  # For Hugging Face models
        "torch>=2.0.1",  # PyTorch with a minimum version
        "accelerate",  # For distributed training/inference
        "gradio",  # For the Gradio interface
        "safetensors",  # For safe model loading
        "pillow",  # For image processing
        "datasets",  # For datasets (if needed)
    )
)

# Create a Modal app
app = modal.App("ctb-image-generator", image=image)

# Debug function to check installed packages
def check_dependencies():
    import importlib
    packages = [
        "diffusers",  # For Stable Diffusion
        "transformers",  # For Hugging Face models
        "torch",  # PyTorch
        "accelerate",  # For distributed training/inference
        "gradio>=4.44.1",  # For the Gradio interface (updated to latest version)
        "safetensors",  # For safe model loading
        "pillow",  # For image processing
    ]

    for package in packages:
        try:
            module = importlib.import_module(package)
            print(f"✅ {package} is installed. Version: {module.__version__}")
        except ImportError:
            print(f"❌ {package} is NOT installed.")

@app.local_entrypoint()
def main():
    print("🚀 Starting Modal app...")
    with modal.enable_output():
        print("🔍 Running debug check...")
        check_dependencies()
        print("🎨 Launching Gradio interface...")
        demo.launch()
    with modal.enable_output():
        demo.launch()

if __name__ == "__main__":
    main()