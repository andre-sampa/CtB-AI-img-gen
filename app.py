# app.py ANDRE
import sys
import os

# Add the src folder to the Python path
# Solves all problems w subfolders - option2
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

# Import gradio_interface
from gradio_interface import demo
from config.config import api_token
from huggingface_hub import InferenceClient
from config.models import models

# Initialize the InferenceClient with the default model
client = InferenceClient(models[0]["name"], token=api_token)


if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch()
