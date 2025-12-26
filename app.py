import torch
torch.set_num_threads(1)

import streamlit as st
import easyocr
import numpy as np
from PIL import Image
from openai import OpenAI
import os

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="OCR + GPT", layout="centered")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

# -----------------------------
# OCR cache
# -----------------------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['it', 'en'])

reader = load_ocr()

# -----------------------------
# Session state
# -----------------------------
if "raw_text" not in st.session_state:
    st.session_state.raw_text = None

if "clean_text" not in st.session_state:
    st.session_state.clean_text = None

st.title("📸 OCR + GPT")

# -----------------------------
# Camera
# -----------------------------
img_file = st.camera_input("Scatta una foto")

if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption="Foto acquisita", use_column_width=True)

    # OCR (automatico)
    with st.spinner("Riconoscimento testo..."):
        img_np = np.array(image)
        ocr_result = reader.readtext(img_np, detail=0)
        raw_text = "\n".join(ocr_result)[:3000]  # limite token
        st.session_state.raw_text = raw_text
        st.session_state.clean_text = None

    st.subheader("📄 Testo OCR grezzo")
    st.text(st.session_state.raw_text)

    # -----------------------------
    # GPT SOLO SU CLICK
    # -----------------------------
    if st.button("🧠 Pulisci testo con GPT"):

        with st.spinner("Pulizia testo con GPT..."):
            try:
                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=f"""
                    Il seguente testo è stato estratto da una foto tramite OCR.
                    Correggi errori, sistema la formattazione e restituisci SOLO il testo finale pulito.

                    TESTO OCR:
                    {st.session_state.raw_text}
                    """
                )

                st.session_state.clean_text = response.output_text

            except Exception as e:
                st.error("Errore GPT (rate limit o quota). Riprova tra qualche secondo.")
                st.stop()

# -----------------------------
# Output finale
# -----------------------------
if st.session_state.clean_text:
    st.subheader("✅ Testo pulito")
    st.text(st.session_state.clean_text)
