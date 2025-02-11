# app.py 
#IMPORT gradio_interface
from src.gradio_interface_diffusers import demo
from config.config import models, prompts, api_token  # Direct import
import sys
import os

# Launch the Gradio app
demo.queue().launch()