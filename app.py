# app.py
import sys
import os

# Add the src folder to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

# Import gradio_interface
from gradio_interface import demo


if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch()
