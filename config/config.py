# config.py
import os
from config.prompts import prompts 
from config.models import models, models_modal

# Retrieve the Hugging Face token
api_token = os.getenv("HF_TOKEN")

# Debugging: Print prompt and model options
print("##### IMPORTING CONFIG #####")
print("Prompt Options:", [p["alias"] for p in prompts])
print("Model Options:", [m["alias"] for m in models])
