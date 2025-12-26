import torch
torch.set_num_threads(1)

import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import re

st.set_page_config(page_title="OCR Locale", layout="centered")

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

st.title("📸 OCR Locale + Pulizia Testo")

# -----------------------------
# Camera input
# -----------------------------
img_file = st.camera_input("Scatta una foto")

if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption="Foto acquisita", use_column_width=True)

    # OCR automatico
    with st.spinner("Riconoscimento testo..."):
        img_np = np.array(image)
        ocr_result = reader.readtext(img_np, detail=0)
        raw_text = "\n".join(ocr_result)[:3000]  # limita lunghezza
        st.session_state.raw_text = raw_text
        st.session_state.clean_text = None

    st.subheader("📄 Testo OCR grezzo")
    st.text(st.session_state.raw_text)

    # Pulizia testo con regex (senza spellchecker)
    if st.button("🧹 Pulisci testo"):
        with st.spinner("Pulizia testo..."):
            clean_text = re.sub(r'\s+', ' ', st.session_state.raw_text).strip()
            st.session_state.clean_text = clean_text

# -----------------------------
# Output finale
# -----------------------------
if st.session_state.clean_text:
    st.subheader("✅ Testo pulito")
    st.text_area("Testo finale", st.session_state.clean_text, height=200)

    # Download TXT
    st.download_button(
        label="⬇️ Scarica TXT",
        data=st.session_state.clean_text,
        file_name="testo_ocr.txt",
        mime="text/plain"
    )
