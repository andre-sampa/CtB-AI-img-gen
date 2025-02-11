import modal

image = (
    modal.Image.debian_slim(python_version="3.11")  # Base image
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

app = modal.App("functions-call-app", image=image)
@app.function()
def main():
    #Import libraries and print their versions
    import numpy as np
    import pandas as pd
    import torch
    import diffusers
    import transformers
    import gradio as gr
    from PIL import Image as PILImage
    
    print("def main function")
    print("Hello from Modal!")
    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)
    print("PyTorch version:", torch.__version__)
    print("Diffusers version:", diffusers.__version__)  # Corrected: Use the library's __version__
    print("Transformers version:", transformers.__version__)  # Corrected: Use the library's __version__
    print("Gradio version:", gr.__version__)
    print("Pillow version:", PILImage.__version__)

    f = modal.Function.from_name("functions-app", "message_func")
    messageNEW = "Remote call Hello World!"
    messageTEMP = "TEMP"
    result = f.remote(messageNEW)
    print(result)

# # Run the function locally (for testing)
if __name__ == "__main__":
    print("Running the function locally...")
    main.local()
    main.remote()
