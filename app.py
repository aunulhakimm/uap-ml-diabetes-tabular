import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import tensorflow as tf
from pathlib import Path

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Diabetes (Tabular ML)",
    page_icon="🩺",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    models_dir = BASE_DIR / "models"
    scaler = joblib.load(models_dir / "scaler.pkl")
    with open(models_dir / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)
    model = tf.keras.models.load_model(models_dir / "mlp_health.keras")
    return scaler, feature_cols, model

scaler, feature_cols, model = load_artifacts()

# =========================
# HEADER
# =========================
st.title("🩺 Sistem Prediksi Penyakit Diabetes (Tabular ML)")
st.markdown("""
Aplikasi ini merupakan **sistem prediksi risiko diabetes** berbasis **pembelajaran mesin**
menggunakan **data tabular** dari *Diabetes Health Indicators (BRFSS)*.

⚠️ **Catatan Penting**  
Aplikasi ini **bukan alat diagnosis medis**, melainkan **simulasi akademik**
untuk membantu memahami bagaimana model Machine Learning memprediksi risiko diabetes
berdasarkan data kesehatan.
""")

st.divider()

# =========================
# PENJELASAN FITUR
# =========================
with st.expander("ℹ️ Penjelasan Singkat Fitur Input"):
    st.markdown("""
- **HighBP / HighChol** → Riwayat tekanan darah / kolesterol tinggi  
- **BMI** → Body Mass Index  
- **Smoker / HvyAlcoholConsump** → Kebiasaan merokok & konsumsi alkohol  
- **PhysActivity / DiffWalk** → Aktivitas fisik & kesulitan berjalan  
- **GenHlth** → Penilaian kesehatan umum (1 = sangat baik, 5 = sangat buruk)  
- **Age, Education, Income** → Faktor demografis (dalam bentuk kategori)
""")

# =========================
# FORM INPUT
# =========================
st.subheader("🧾 Input Data Pengguna")

defaults = {
    "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 25.0,
    "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
    "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 3, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0,
    "Sex": 0, "Age": 8, "Education": 4, "Income": 5
}

inputs = {}

with st.form("input_form"):
    col1, col2 = st.columns(2)

    for i, col in enumerate(feature_cols):
        target_col = col1 if i % 2 == 0 else col2

        with target_col:
            if col in defaults:
                if col == "BMI":
                    inputs[col] = st.number_input(col, 10.0, 70.0, defaults[col], step=0.1)
                elif col in ["MentHlth", "PhysHlth"]:
                    inputs[col] = st.slider(col, 0, 30, defaults[col])
                elif col in ["GenHlth"]:
                    inputs[col] = st.slider(col, 1, 5, defaults[col])
                elif col in ["Age"]:
                    inputs[col] = st.slider(col, 1, 13, defaults[col])
                elif col in ["Education"]:
                    inputs[col] = st.slider(col, 1, 6, defaults[col])
                elif col in ["Income"]:
                    inputs[col] = st.slider(col, 1, 8, defaults[col])
                else:
                    inputs[col] = st.selectbox(col, [0, 1], index=defaults[col])

    threshold = st.slider(
        "Threshold Prediksi (Default = 0.50)",
        0.05, 0.95, 0.50, 0.05,
        help="Semakin rendah threshold → semakin sensitif terhadap risiko diabetes"
    )

    submitted = st.form_submit_button("🔍 Prediksi Risiko Diabetes")

# =========================
# PREDIKSI
# =========================
if submitted:
    x_df = pd.DataFrame([[inputs[c] for c in feature_cols]], columns=feature_cols)
    x_scaled = scaler.transform(x_df)

    prob = float(model.predict(x_scaled, verbose=0)[0][0])
    pred = 1 if prob >= threshold else 0

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    st.write(f"**Probabilitas Risiko Diabetes:** `{prob:.4f}`")

    if pred == 1:
        st.error("⚠️ Model memprediksi **RISIKO DIABETES**")
        st.markdown("""
        Artinya, berdasarkan data yang dimasukkan, pengguna memiliki **risiko lebih tinggi**
        terhadap diabetes menurut model Machine Learning.
        """)
    else:
        st.success("✅ Model memprediksi **RISIKO RENDAH / NON-DIABETES**")
        st.markdown("""
        Artinya, berdasarkan data yang dimasukkan, risiko diabetes **relatif rendah**
        menurut model.
        """)

    st.info("""
    💡 **Ingat:**  
    Hasil ini hanyalah **prediksi berbasis data**, bukan diagnosis.
    Untuk keputusan medis, selalu konsultasikan dengan tenaga kesehatan profesional.
    """)
