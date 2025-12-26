import torch
torch.set_num_threads(1)

import streamlit as st
import easyocr
import numpy as np
from PIL import Image
from openai import OpenAI

st.set_page_config(page_title="OCR + GPT", layout="centered")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])



# Inizializza OCR (una sola volta)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['it', 'en'])

reader = load_ocr()

st.title("📸 OCR + GPT")

# -----------------------------
# Camera
# -----------------------------
img_file = st.camera_input("Scatta una foto")

if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption="Foto acquisita", use_column_width=True)

    with st.spinner("Riconoscimento testo..."):
        img_np = np.array(image)
        ocr_result = reader.readtext(img_np, detail=0)
        raw_text = "\n".join(ocr_result)

    with st.spinner("Pulizia testo con GPT..."):
        prompt = f"""
        Il seguente testo è stato estratto da una foto tramite OCR.
        Correggi errori, sistema la formattazione e restituisci SOLO il testo finale pulito.

        TESTO OCR:
        {raw_text}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sei un assistente che pulisce testo OCR."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        clean_text = response.choices[0].message.content

    st.subheader("📄 Testo riconosciuto")
    st.text(clean_text)
