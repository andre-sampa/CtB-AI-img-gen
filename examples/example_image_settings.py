 image = (
modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "numpy", 
        "pandas",         
        "diffusers",
        "transformers",
        "torch",
        "accelerate",
        "gradio",
        "safetensors",
        "pillow",
    )  # Install Python packages
    .run_commands("echo 'Image build complete!'")  # Run a shell command
)


# CHECK INSTALLS
def function():
# Import libraries and print their versions
    import numpy as np
    import pandas as pd
    import torch
    import diffusers
    import transformers
    import gradio as gr
    from PIL import Image as PILImage

    print("Hello from ctb_modal!")
    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)
    print("PyTorch version:", torch.__version__)
    print("Diffusers version:", diffusers.__version__)  # Corrected: Use the library's __version__
    print("Transformers version:", transformers.__version__)  # Corrected: Use the library's __version__
    print("Gradio version:", gr.__version__)
    print("Pillow version:", PILImage.__version__)


    # # Run the function locally (for testing)
# if __name__ == "__main__":
#     print("Running the function locally...")
#     main.local()






image = (
    modal.Image.debian_slim(python_version="3.9")  # Base image