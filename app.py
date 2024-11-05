import streamlit as st
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model = LlamaForCausalLM.from_pretrained("models/llama-3.2-1b")
    tokenizer = LlamaTokenizer.from_pretrained("models/llama-3.2-1b")
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI
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
