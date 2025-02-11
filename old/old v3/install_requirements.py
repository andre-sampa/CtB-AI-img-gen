import modal

# Create an image and install dependencies directly
image = modal.Image.from_pip([
    "gradio",
    "torch",
    "diffusers"
])

@modal.function(image=image)
def test_function():
    import gradio
    print("Gradio is installed successfully!")

if __name__ == "__main__":
    test_function()
