# main.py
import sys
import os

# Add the /src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import necessary modules
from src.gradio_interface import demo

if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch()