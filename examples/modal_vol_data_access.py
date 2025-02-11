# img_gen.py
#img_gen_modal.py
import modal
import os
import shutil


# Define the Modal image
image = (
    modal.Image.debian_slim(python_version="3.9")
    )

# Create a Modal app
app = modal.App("img-see-vol-data", image=image)
        
flux_model_vol = modal.Volume.from_name("flux-model-vol",create_if_missing=False)  # Reference your volume

@app.function(volumes={"/data": flux_model_vol}
              )
def main():
    
    # Define where to store the model
    download_model_name = "black-forest-labs/FLUX.1-dev"  # e.g., "stabilityai/stable-diffusion-2"
    local_dir = "data/FLUX.1-dev"


    model_path = "/data"
    if os.path.exists(model_path):
        print(f"Contents of {model_path}:")
        print(os.listdir(model_path))
    else:
        print(f"Model path {model_path} does not exist.")



        
