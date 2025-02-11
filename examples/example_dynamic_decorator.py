
import modal

# Define the Modal image and app
image = modal.Image.debian_slim(python_version="3.9")
app = modal.App("example-app", image=image)

# Define the volume
flux_model_vol = modal.Volume.from_name("flux-model-vol", create_if_missing=True)

# Load configuration (e.g., from a config file or environment variables)
cpu = 8  # Set to 0 to disable CPU
memory = 70000  # Memory in MB
gpu = "a100-80gb"  # Set to None to disable GPU

# Dynamically construct the decorator arguments
decorator_args = {
    "volumes": {"/data": flux_model_vol},
    "secrets": [modal.Secret.from_name("huggingface-token")],
    "cpu": cpu,
    "memory": memory,
}

# Remove GPU if CPU is set
if cpu > 0:
    print("CPU is set, removing GPU parameter.")
    decorator_args.pop("gpu", None)  # Remove 'gpu' if it exists
else:
    print("CPU is not set, keeping GPU parameter.")
    decorator_args["gpu"] = gpu

# Debug: Print the final decorator arguments
print("Decorator arguments:", decorator_args)

# Apply the decorator dynamically
@app.function(**decorator_args)
def my_function():
    import os

    # Example: List the contents of the volume
    print("Contents of /data:")
    print(os.listdir("/data"))

    # Your function code here
    return f"Function executed with CPU={cpu}, Memory={memory}, GPU={gpu if 'gpu' in decorator_args else 'None'}"

# Call the function
result = my_function.remote()
print(result)