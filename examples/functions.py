import modal

# Print debug information
print("Importing Modal and setting up the app...")

# Define a custom image with Python and some dependencies
print("Building custom image...")
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

# Define a function to run inside the container
#@app.function(image=image)

# Define the Modal app
app = modal.App("functions-app")

@app.function()
def message_func (message = "default"):
    print("message function")
    new_message = message + " ok, it works!"
    return new_message



@app.local_entrypoint()
def main():
    # Import libraries and print their versions
    # import numpy as np
    # import pandas as pd
    # import torch
    # import diffusers
    # import transformers
    # import gradio as gr
    # from PIL import Image as PILImage
    
    # print("def main function")
    # print("Hello from Modal!")
    # print("NumPy version:", np.__version__)
    # print("Pandas version:", pd.__version__)
    # print("PyTorch version:", torch.__version__)
    # print("Diffusers version:", diffusers.__version__)  # Corrected: Use the library's __version__
    # print("Transformers version:", transformers.__version__)  # Corrected: Use the library's __version__
    # print("Gradio version:", gr.__version__)
    # print("Pillow version:", PILImage.__version__)

    remote_message = "remote message!"
    local_message = "local message"
    message_func.remote(remote_message)
    message_func.local(local_message)


# # # Run the function locally (for testing)
# if __name__ == "__main__":
#     print("Running the function locally...")
#     main.local()
#     main.remote()
