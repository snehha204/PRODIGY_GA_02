import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Page config
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="🤖",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .title {
        font-size: 40px;
        font-weight: 700;
        color: #4A00E0;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4A00E0, #8E2DE2);
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- HEADER ----------
st.markdown('<div class="title">🤖 AI Text Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Fine-tuned GPT-2 model for smart text generation ✨</div>',
    unsafe_allow_html=True
)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

# ---------- INPUT ----------
st.markdown("### ✍️ Enter your prompt")
prompt = st.text_area(
    "",
    placeholder="Example: Artificial intelligence is transforming the world because...",
    height=120
)

col1, col2 = st.columns(2)

with col1:
    max_length = st.slider("📏 Max length", 50, 200, 100)

with col2:
    temperature = st.slider("🎨 Creativity", 0.5, 1.5, 0.8)

# ---------- GENERATE ----------
if st.button("🚀 Generate Text"):
    if prompt.strip() == "":
        st.warning("⚠️ Please enter a prompt to generate text.")
    else:
        with st.spinner("🤖 Thinking... generating text"):
            tokenizer, model = load_model()
            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=True
                )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        st.markdown("### 📝 Generated Output")
        st.success(generated_text)

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "💡 *Built using GPT-2, Hugging Face Transformers, and Streamlit*  \n"
    
)