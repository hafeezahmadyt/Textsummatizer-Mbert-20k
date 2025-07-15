# app.py
import streamlit as st
from utils import Summarizer

st.set_page_config(page_title="Urdu Summarizer", layout="centered")

st.title("🧠 Urdu Text Summarizer")
st.markdown("Summarize long Urdu text using your trained mBERT model.")

text_input = st.text_area("✍️ Paste Urdu text below:", height=300)

# Load the model once
if "model" not in st.session_state:
    st.session_state.model = Summarizer("model/model.bin")

if st.button("🔍 Generate Summary"):
    if text_input.strip() == "":
        st.warning("Please enter some Urdu text.")
    else:
        with st.spinner("Generating summary..."):
            summary = st.session_state.model.predict(text_input)
            st.success("✅ Summary:")
            st.text_area("📋 Summary", summary, height=150)
