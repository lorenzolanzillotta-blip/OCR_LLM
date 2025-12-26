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
# OCR (cache)
# -----------------------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['it', 'en'])

reader = load_ocr()

# -----------------------------
# Session state
# -----------------------------
if "clean_text" not in st.session_state:
    st.session_state.clean_text = None

if "last_image_id" not in st.session_state:
    st.session_state.last_image_id = None

st.title("📸 OCR + GPT")

# -----------------------------
# Camera
# -----------------------------
img_file = st.camera_input("Scatta una foto")

if img_file is not None:

    # Identifica nuova immagine
    image_id = hash(img_file.getvalue())

    # Se foto nuova → reset stato
    if st.session_state.last_image_id != image_id:
        st.session_state.clean_text = None
        st.session_state.last_image_id = image_id

    image = Image.open(img_file)
    st.image(image, caption="Foto acquisita", use_column_width=True)

    # OCR
    with st.spinner("Riconoscimento testo..."):
        img_np = np.array(image)
        ocr_result = reader.readtext(img_np, detail=0)
        raw_text = "\n".join(ocr_result)

        # sicurezza token
        raw_text = raw_text[:3000]

    # GPT (UNA SOLA VOLTA)
    if st.session_state.clean_text is None:
        with st.spinner("Pulizia testo con GPT..."):
            response = client.responses.create(
                model="gpt-4o-mini",
                input=f"""
                Il seguente testo è stato estratto da una foto tramite OCR.
                Correggi errori, sistema la formattazione e restituisci SOLO il testo finale pulito.

                TESTO OCR:
                {raw_text}
                """
            )

            st.session_state.clean_text = response.output_text

    # Output
    st.subheader("📄 Testo riconosciuto")
    st.text(st.session_state.clean_text)
