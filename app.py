import streamlit as st
import torch
import os
from transformers import LlamaForCausalLM, LlamaTokenizer
import requests

# Function to download the model file from Google Drive
@st.cache_resource
def download_model():
    url = "https://drive.google.com/uc?export=download&id=1ms0G8on7sD8drXnJgyTajiATpjeykohc"  # Replace with your direct download link
    model_path = "models/model.safetensors"
    
    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists(model_path):
        st.write("Downloading the model. This may take a while...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.write("Model downloaded successfully!")
    
    return model_path

# Function to load the model and tokenizer
@st.cache_resource
def load_model():
    model_path = download_model()
    model = LlamaForCausalLM.from_pretrained("models", torch_dtype=torch.bfloat16)
    tokenizer = LlamaTokenizer.from_pretrained("models")
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit User Interface
st.title("Llama 3.2 1B Model on Streamlit Cloud")
prompt = st.text_area("Enter your prompt:", "")

if st.button("Generate Response"):
    if prompt:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("**Response:**")
        st.write(response)
    else:
        st.warning("Please enter a prompt to get a response.")
