# UAP Machine Learning  

## 🌐 Demo Online
[Akses aplikasi Streamlit di sini](https://uap-ml-diabetes-tabular.streamlit.app/)
## Sistem Prediksi Penyakit Diabetes Berbasis Data Tabular

Proyek ini merupakan tugas **Ujian Akhir Praktikum (UAP)** mata kuliah Machine Learning.
Sistem dibangun untuk memprediksi **risiko penyakit diabetes** menggunakan **data tabular**
dan membandingkan performa beberapa model Machine Learning.

---

## Dataset
1. **Pima Indians Diabetes Dataset**  
   https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  
   Digunakan sebagai baseline dan validasi awal.

2. **Diabetes Health Indicators (BRFSS)**  
   https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset  
   Digunakan sebagai dataset utama (>5000 data).  
   Label dikonversi menjadi klasifikasi biner:
   - 0 → Non-diabetes
   - 1 → Risiko diabetes

---

## Model yang Digunakan
1. **Neural Network Base (Non-Pretrained)**  
   - Multilayer Perceptron (MLP)

2. **Model Pretrained 1**  
   - TabNet

3. **Model Pretrained 2**  
   - FT-Transformer

---

## Evaluasi dan Analisis Model

### Baseline (Pima - MLP)
- Accuracy: 0.75  
- Recall kelas diabetes: 0.57

### Dataset Utama (Health Indicators)

| Model | Akurasi | Precision (Diabetes) | Recall (Diabetes) | F1-Score (Diabetes) | Analisis |
|------|--------|----------------------|-------------------|---------------------|---------|
| MLP | 0.85 | 0.61 | 0.16 | 0.26 | Akurasi tinggi namun recall rendah akibat ketidakseimbangan kelas |
| TabNet | 0.85 | 0.63 | 0.14 | 0.23 | Performa mirip MLP, belum signifikan meningkatkan recall |
| FT-Transformer | 0.85 | 0.58 | 0.20 | 0.30 | Recall terbaik untuk kelas diabetes, namun komputasi lebih berat |

---

## Sistem Prediksi (Streamlit)
Aplikasi Streamlit dibuat sebagai **sistem prediksi interaktif**.
Pengguna dapat memasukkan data kesehatan dan memperoleh prediksi risiko diabetes.

### Menjalankan Aplikasi
```bash
pip install -r requirements.txt
streamlit run app.py
