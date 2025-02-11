import modal
from src.gradio_interface import demo

# Print debug information
print("Importing Modal and setting up the app...")

# Define the Modal app
app = modal.App(name="example-app")

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
@app.function(image=image)
def main():
    # Debug: Print a message when the function starts
    print("Starting main function inside the container...")

    # Import libraries and print their versions
    import numpy as np
    import pandas as pd
    import torch
    import diffusers
    import transformers
    import gradio as gr
    from PIL import Image as PILImage
    


    print("Hello from Modal!")
    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)
    print("PyTorch version:", torch.__version__)
    print("Diffusers version:", diffusers.__version__)  # Corrected: Use the library's __version__
    print("Transformers version:", transformers.__version__)  # Corrected: Use the library's __version__
    print("Gradio version:", gr.__version__)
    print("Pillow version:", PILImage.__version__)

    # Create a simple DataFrame
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    print("DataFrame:\n", df)

    # Test PyTorch
    tensor = torch.tensor([1, 2, 3])
    print("PyTorch tensor:", tensor)

    # Test Diffusers (load a simple pipeline)
    print("Loading Diffusers pipeline...")
    pipe = diffusers.DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    print("Diffusers pipeline loaded successfully!")

    # Test Gradio (create a simple interface)
    def greet(name):
        return f"Hello {name}!"

    print("Creating Gradio interface...")
    iface = gr.Interface(fn=greet, inputs="text", outputs="text")
    print("Gradio interface created successfully!")

    # Debug: Print a message when the function ends
    print("Main function execution complete!")

    # Launch gradio-interface
    demo.launch()

# Run the function locally (for testing)
if __name__ == "__main__":
    print("Running the function locally...")
    main.local()