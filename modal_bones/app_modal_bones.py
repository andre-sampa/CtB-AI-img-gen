# app.py 
#IMPORT gradio_interface
from src.gradio_interface_modal import demo
from config.config import models, prompts, api_token  # Direct import


# Launch the Gradio app
demo.queue().launch()