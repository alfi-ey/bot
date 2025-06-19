import pandas as pd
import numpy as np
import streamlit as st
import pickle
import time
from keras.models import load_model

st.set_page_config(page_title="AgroAI 🌾", layout="centered")

@st.cache_resource
def load_model_and_tools():
    model = load_model("trained_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model_and_tools()

def analyze_soil(n, p, k, temp, hum, ph, rain):
    def kategori_npk(x):
        if x < 50:
            return "rendah"
        elif x <= 100:
            return "sedang"
        else:
            return "tinggi"

    def kategori_ph(ph):
        if ph < 5.5:
            return "asam"
        elif ph <= 7.0:
            return "netral"
        else:
            return "basa"

    def kategori_suhu(temp):
        if temp < 20:
            return "dingin"
        elif temp <= 30:
            return "optimal"
        else:
            return "panas"

    def kategori_kelembaban(hum):
        if hum < 30:
            return "kering"
        elif hum <= 70:
            return "cukup"
        else:
            return "lembab"

    def kategori_hujan(rain):
        if rain < 50:
            return "rendah"
        elif rain <= 150:
            return "sedang"
        else:
            return "tinggi"

    kondisi = {
        "Nitrogen": kategori_npk(n),
        "Fosfor": kategori_npk(p),
        "Kalium": kategori_npk(k),
        "pH": kategori_ph(ph),
        "Suhu": kategori_suhu(temp),
        "Kelembaban": kategori_kelembaban(hum),
        "Curah Hujan": kategori_hujan(rain)
    }

    saran = []

    if kondisi["Nitrogen"] == "rendah":
        saran.append("🌱 Tambahkan pupuk urea untuk meningkatkan kadar nitrogen.")
    elif kondisi["Nitrogen"] == "tinggi":
        saran.append("⚠️ Kurangi pupuk nitrogen agar tidak merusak tanaman.")

    if kondisi["Fosfor"] == "rendah":
        saran.append("🌾 Gunakan pupuk TSP atau SP-36 untuk meningkatkan fosfor.")
    elif kondisi["Fosfor"] == "tinggi":
        saran.append("⚠️ Batasi pupuk fosfor agar keseimbangan unsur tetap terjaga.")

    if kondisi["Kalium"] == "rendah":
        saran.append("🌿 Gunakan pupuk KCl untuk meningkatkan kalium.")
    elif kondisi["Kalium"] == "tinggi":
        saran.append("⚠️ Kalium tinggi bisa mengganggu penyerapan magnesium.")

    if kondisi["pH"] == "asam":
        saran.append("🔧 Gunakan kapur dolomit untuk menetralkan tanah asam.")
    elif kondisi["pH"] == "basa":
        saran.append("🧪 Gunakan belerang atau bahan organik untuk menurunkan pH.")

    if kondisi["Suhu"] == "dingin":
        saran.append("🔥 Gunakan mulsa untuk menjaga suhu tetap hangat.")
    elif kondisi["Suhu"] == "panas":
        saran.append("💧 Tingkatkan irigasi dan tambahkan peneduh.")

    if kondisi["Kelembaban"] == "kering":
        saran.append("🚿 Perbanyak penyiraman dan gunakan irigasi tetes.")
    elif kondisi["Kelembaban"] == "lembab":
        saran.append("🛠 Pastikan drainase baik agar tidak tergenang.")

    if kondisi["Curah Hujan"] == "rendah":
        saran.append("🌧 Tambahkan sistem irigasi tambahan.")
    elif kondisi["Curah Hujan"] == "tinggi":
        saran.append("🔧 Perbaiki saluran air untuk mencegah genangan.")

    return kondisi, saran

st.markdown("""
    <div style='
        background-color: #e8f5e9;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        text-align: center;
    '>
        <h1 style='color: #2e7d32; font-size: 3em;'>🌾 AgroAI</h1>
        <p style='font-size: 1.2em; color: #4e944f;'>
            Selamat datang di <b>AgroAI</b> — Konsultan lahan pintar berbasis <i>AI</i> untuk membantu kamu menentukan <b>tanaman terbaik, analisis kondisi tanah</b> dan <b>perbaikan untuk tanah kamu.</b> 🌱🤖
        </p>
        <p style='font-size: 1.05em; color: #6b8e23;'>
            Yuk, isi data lahanmu di bawah ini dan temukan rekomendasi terbaik dari AgroAI 💡🌍
        </p>
    </div>
""", unsafe_allow_html=True)

n = st.slider("🌿 Nitrogen (N)", 0, 150, 111)
p = st.slider("🌿 Fosfor (P)", 0, 150, 27)
k = st.slider("🌿 Kalium (K)", 0, 150, 31)
temp = st.slider("🌡 Suhu (°C)", 0.0, 50.0, 23.59, step=0.01)
hum = st.slider("💧 Kelembaban (%)", 0.0, 100.0, 55.28, step=0.01)
ph = st.slider("🧪 pH", 3.0, 9.0, 6.04, step=0.001)
rain = st.slider("🌧 Curah Hujan (mm)", 0.0, 300.0, 191.40, step=0.01)

if st.button("🔍 Konsultasikan"):
    with st.spinner("Sedang menganalisis lahan kamu... 🌍"):
        time.sleep(2)

    input_data = np.array([[n, p, k, temp, hum, ph, rain]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(np.max(prediction))

    st.chat_message("assistant").markdown(f"""
    ✅ Berdasarkan data lahanmu, **tanaman yang cocok** adalah:
    ### 🌾 `{predicted_label.upper()}`
    Dengan tingkat keyakinan: **{confidence:.2%}**
    """)

    kondisi, saran = analyze_soil(n, p, k, temp, hum, ph, rain)

    with st.expander("🧪 Analisis Lengkap Kondisi Tanah", expanded=False):
        st.markdown("""
        <div style='background-color: #f0f4c3; padding: 15px; border-radius: 10px;'>
            <h4 style='color: #558b2f;'>🌿 Kondisi Tanah di Lahan Kamu:</h4>
            <ul style='line-height: 1.8; font-size: 1.05em; color: #33691e;'>
        """, unsafe_allow_html=True)

        for faktor, nilai in kondisi.items():
            st.markdown(f"<li><b>{faktor}</b>: {nilai.capitalize()}</li>", unsafe_allow_html=True)

        st.markdown("</ul></div>", unsafe_allow_html=True)

    with st.expander("🌱 Rekomendasi Perbaikan Tanah & Tips", expanded=False):
        if saran:
            st.markdown("""
            <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px;'>
                <h4 style='color: #2e7d32;'>💡 Saran dan Tindakan yang Direkomendasikan Untuk Lahan Kamu:</h4>
                <ul style='line-height: 1.8; font-size: 1.05em; color: #1b5e20;'>
            """, unsafe_allow_html=True)

            for tips in saran:
                st.markdown(f"<li>{tips}</li>", unsafe_allow_html=True)

            st.markdown("</ul></div>", unsafe_allow_html=True)
        else:
            st.success("✨ Kondisi tanah sudah optimal! Tidak ada saran tambahan. 🌿")
