
import modal
import os

app = modal.App("test-volume")

@app.function(volumes={"/my_vol": modal.Volume.from_name("flux-model-vol")})
def test_func():
    print("Contents of the volume:", os.listdir("/my_vol"))

@app.local_entrypoint()
def main():
    test_func.call()
