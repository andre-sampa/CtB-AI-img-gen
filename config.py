# config.py
import os
#from google.colab import userdata  # Only for Colab
##

# Retrieve the Hugging Face token
#try:
    # Try to get the token from Colab secrets
#    api_token = userdata.get("HF_CTB_TOKEN")
#except ImportError:
    # Fall back to environment variable (for local execution)
api_token = os.getenv("HF_CTB_TOKEN")

# Debugging: Check if the Hugging Face token is available
if not api_token:
    print("ERROR: Hugging Face token (HF_CTB_TOKEN) is missing. Please set it as an environment variable or in Colab secrets.")
else:
    print("Hugging Face token loaded successfully.")


# List of models with aliases
models = [
    {"alias": "FLUX.1-dev", "name": "black-forest-labs/FLUX.1-dev"},
    {"alias": "Midjourney", "name": "strangerzonehf/Flux-Midjourney-Mix2-LoRA"}
]

# Debugging: Print prompt and model options
print("Prompt Options:", [p["alias"] for p in prompts])
print("Model Options:", [m["alias"] for m in models])

