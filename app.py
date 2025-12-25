import streamlit as st
import numpy as np
import pandas as pd
import json, joblib
from tensorflow.keras.models import load_model

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# LOAD ARTIFACTS
# =====================================================
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("models/scaler.pkl")
    model = load_model("models/mlp_health.keras")
    with open("models/feature_columns.json") as f:
        feature_cols = json.load(f)
    return scaler, model, feature_cols

scaler, model, feature_cols = load_artifacts()

# =====================================================
# HEADER
# =====================================================
st.title("Sistem Prediksi Risiko Diabetes")
st.caption("Analisis Risiko Kesehatan Berbasis AI")

# =====================================================
# INTRO
# =====================================================
st.info("""
Aplikasi ini menggunakan machine learning untuk memprediksi risiko diabetes berdasarkan indikator kesehatan. Hasil prediksi hanya untuk informasi awal dan **bukan diagnosis medis**. Konsultasikan dengan tenaga medis profesional untuk evaluasi kesehatan yang akurat.
""")

# =====================================================
# SKENARIO PRESET
# =====================================================
st.subheader("Skenario Uji Cepat")
col_scenario, col_info = st.columns([1, 2])
with col_scenario:
    scenario = st.selectbox(
        "Pilih skenario preset",
        ["Input Manual", "Risiko Rendah", "Risiko Sedang", "Risiko Tinggi"],
        help="Pilih preset atau isi data manual"
    )
with col_info:
    if scenario != "Input Manual":
        st.info(f"Skenario **{scenario}** dipilih. Data akan terisi otomatis di bawah.")

def preset(s):
    if s == "Risiko Rendah":
        return dict(
            HighBP=0, HighChol=0, CholCheck=1, BMI=21,
            Smoker=0, Stroke=0, HeartDiseaseorAttack=0,
            PhysActivity=1, Fruits=1, Veggies=1,
            HvyAlcoholConsump=0, AnyHealthcare=1,
            NoDocbcCost=0, GenHlth=1,
            MentHlth=0, PhysHlth=0,
            DiffWalk=0, Sex=0,
            Age=2, Education=5, Income=7
        )
    if s == "Risiko Sedang":
        return dict(
            HighBP=1, HighChol=1, CholCheck=1, BMI=30,
            Smoker=1, Stroke=0, HeartDiseaseorAttack=0,
            PhysActivity=0, Fruits=0, Veggies=0,
            HvyAlcoholConsump=0, AnyHealthcare=1,
            NoDocbcCost=0, GenHlth=4,
            MentHlth=10, PhysHlth=10,
            DiffWalk=1, Sex=1,
            Age=8, Education=3, Income=3
        )
    if s == "Risiko Tinggi":
        return dict(
            HighBP=1, HighChol=1, CholCheck=1, BMI=38,
            Smoker=1, Stroke=1, HeartDiseaseorAttack=1,
            PhysActivity=0, Fruits=0, Veggies=0,
            HvyAlcoholConsump=1, AnyHealthcare=0,
            NoDocbcCost=1, GenHlth=5,
            MentHlth=25, PhysHlth=25,
            DiffWalk=1, Sex=1,
            Age=12, Education=2, Income=1
        )
    return {}

preset_values = preset(scenario)

# =====================================================
# FORM INPUT
# =====================================================
st.subheader("Data Kesehatan Pasien")
st.caption("Isi data kesehatan atau gunakan preset di atas")
col1, col2 = st.columns(2)
input_data = {}
def yesno(label, key):
    return st.selectbox(label, ["Tidak", "Ya"], index=1 if preset_values.get(key,0)==1 else 0)
with col1:
    st.markdown("**Kondisi Medis**")
    input_data["HighBP"] = 1 if yesno("Tekanan Darah Tinggi", "HighBP")=="Ya" else 0
    input_data["HighChol"] = 1 if yesno("Kolesterol Tinggi", "HighChol")=="Ya" else 0
    input_data["Stroke"] = 1 if yesno("Riwayat Stroke", "Stroke")=="Ya" else 0
    input_data["HeartDiseaseorAttack"] = 1 if yesno("Penyakit Jantung/Serangan Jantung", "HeartDiseaseorAttack")=="Ya" else 0
    st.markdown("**Gaya Hidup**")
    input_data["Smoker"] = 1 if yesno("Perokok Aktif", "Smoker")=="Ya" else 0
    input_data["PhysActivity"] = 1 if yesno("Aktivitas Fisik Rutin", "PhysActivity")=="Ya" else 0
    input_data["DiffWalk"] = 1 if yesno("Kesulitan Berjalan", "DiffWalk")=="Ya" else 0
with col2:
    st.markdown("**Metrik Kesehatan**")
    input_data["BMI"] = st.slider("Indeks Massa Tubuh (BMI)", 15.0, 50.0, float(preset_values.get("BMI",25)), help="Berat(kg) / Tinggi(m)²")
    input_data["GenHlth"] = st.slider("Kesehatan Umum (1=Sangat Baik, 5=Buruk)", 1, 5, preset_values.get("GenHlth",3))
    input_data["MentHlth"] = st.slider("Hari Kesehatan Mental Buruk (30 hari terakhir)", 0, 30, preset_values.get("MentHlth",0))
    input_data["PhysHlth"] = st.slider("Hari Kesehatan Fisik Buruk (30 hari terakhir)", 0, 30, preset_values.get("PhysHlth",0))
    st.markdown("**Demografi**")
    input_data["Age"] = st.slider("Kelompok Usia (1=18-24 ... 13=80+)", 1, 13, preset_values.get("Age",5))
    input_data["Education"] = st.slider("Tingkat Pendidikan (1-6)", 1, 6, preset_values.get("Education",4))
    input_data["Income"] = st.slider("Tingkat Pendapatan (1-8)", 1, 8, preset_values.get("Income",4))
# Fitur tetap
input_data["CholCheck"] = 1
input_data["Fruits"] = preset_values.get("Fruits",1)
input_data["Veggies"] = preset_values.get("Veggies",1)
input_data["HvyAlcoholConsump"] = preset_values.get("HvyAlcoholConsump",0)
input_data["AnyHealthcare"] = preset_values.get("AnyHealthcare",1)
input_data["NoDocbcCost"] = preset_values.get("NoDocbcCost",0)
input_data["Sex"] = preset_values.get("Sex",0)
# Threshold
with st.expander("Pengaturan Lanjutan - Threshold Risiko", expanded=False):
    st.markdown("**Atur ambang batas klasifikasi risiko:**")
    threshold = st.slider(
        "Threshold Risiko Tinggi", 
        0.05, 0.95, 0.5,
        help="Probabilitas minimum untuk klasifikasi risiko tinggi"
    )
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.info(f"**Risiko Rendah**\n\n< {threshold*0.6:.0%}")
    with col_t2:
        st.info(f"**Risiko Sedang**\n\n{threshold*0.6:.0%} - {threshold:.0%}")
    with col_t3:
        st.info(f"**Risiko Tinggi**\n\n≥ {threshold:.0%}")
# =====================================================
# PREDIKSI
# =====================================================
st.subheader("Prediksi Risiko Diabetes")
if st.button("Analisis Risiko Diabetes", type="primary"):
    X = pd.DataFrame([input_data])[feature_cols]
    X_scaled = scaler.transform(X)
    prob = model.predict(X_scaled)[0][0]
    st.metric("Probabilitas Risiko Diabetes", f"{prob*100:.2f}%")
    if prob >= threshold:
        st.error("Risiko Tinggi! Segera konsultasi ke dokter dan lakukan pemeriksaan lebih lanjut.")
    elif prob >= threshold*0.6:
        st.warning("Risiko Sedang. Lakukan monitoring dan perbaiki gaya hidup.")
    else:
        st.success("Risiko Rendah. Pertahankan gaya hidup sehat dan lakukan pemeriksaan rutin.")
    st.info("""
    **Catatan Penting:**
    - Prediksi ini berbasis model statistik dan data populasi
    - **Bukan diagnosis medis**
    - Gunakan sebagai alat skrining awal
    - Konsultasikan dengan tenaga medis profesional untuk diagnosis akurat
    - Faktor risiko dapat berubah, lakukan pemeriksaan berkala
    """)
# =====================================================
# FOOTER
# =====================================================
st.caption("Sistem Prediksi Risiko Diabetes | 2025 | Untuk Edukasi dan Skrining Awal")
