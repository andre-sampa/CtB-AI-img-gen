import gradio as gr
import modal
from modal import App, Image, Volume
from transformers import AutoModel, AutoTokenizer
import os

app = App("gradio-app")
volume = Volume.from_name("flux-model-vol-2")
image = Image.debian_slim().pip_install("transformers", "torch", "sentencepiece", "gradio")

@app.function(image=image, volumes={"/data": volume})
def load_model():
    model_name = "FLUX.1-dev"
    cache_dir = f"/data/{model_name}"

    print(f"Loading model {model_name} from cache...")
    model = AutoModel.from_pretrained(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(cache_dir)

    print(f"Model {model_name} loaded successfully!")
    return model, tokenizer

def predict(input_text):
    model, tokenizer = load_model.remote()
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)
    return tokenizer.decode(outputs.logits.argmax(dim=-1)[0])

if __name__ == "__main__":
    with app.run():
        iface = gr.Interface(fn=predict, inputs="text", outputs="text")
        iface.launch()