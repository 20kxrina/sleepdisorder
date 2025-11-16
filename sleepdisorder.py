import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

MODEL_PATH = 'sleep_disorder.keras'        
PREPROCESSOR_PATH = 'preprocessor.pkl' 

@st.cache_resource
def load_resources(model_path, preprocessor_path):
    """Memuat model Keras dan preprocessor pkl."""

    try:
        model = tf.keras.models.load_model(model_path)
        
        preprocessor=joblib.load('preprocessor.pkl')
            
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"⚠️ Error: File tidak ditemukan. Pastikan **{e.filename}** ada di direktori yang benar.")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error saat memuat sumber daya: {e}")
        return None, None

model, preprocessor = load_resources(MODEL_PATH, PREPROCESSOR_PATH)

def get_user_input_dict():
    """Mengumpulkan semua input fitur dari sidebar ke dalam sebuah dictionary."""
    
    st.sidebar.header("⚙️ Input Fitur")

    input_data = {}

    input_data['Gender'] = st.sidebar.selectbox("Gender", ('Male', 'Female'))
    input_data['Gender']=1 if input_data['Gender']=='Male' else 0
    input_data['Age'] = st.sidebar.slider("Age (Usia)", 18, 100, 45)

    occupations = [
        'Software Engineer', 'Doctor', 'Sales Representative', 
        'Teacher', 'Nurse', 'Engineer', 'Accountant', 
        'Scientist', 'Lawyer', 'Salesperson', 'Manager'
    ] 
    input_data['Occupation'] = st.sidebar.selectbox("Occupation (Pekerjaan)", occupations)

    input_data['Sleep Duration'] = st.sidebar.number_input("Sleep Duration (Durasi Tidur, jam)", min_value=1.0, max_value=10.0, value=7.0, step=0.1)

    input_data['Quality of Sleep'] = st.sidebar.slider("Quality of Sleep (Kualitas Tidur, 1-10)", 1, 10, 7)

    input_data['Physical Activity Level'] = st.sidebar.slider("Physical Activity Level (Menit/hari)", 0, 150, 60)

    input_data['Stress Level'] = st.sidebar.slider("Stress Level (Tingkat Stres, 1-10)", 1, 10, 5)

    bmi_categories = ['Normal', 'Overweight', 'Obese', 'Normal Weight'] # SESUAIKAN KATEGORI INI
    input_data['BMI Category'] = st.sidebar.selectbox("BMI Category", bmi_categories)

    input_data['Heart Rate'] = st.sidebar.slider("Heart Rate (Detak Jantung, bpm)", 50, 100, 70)

    input_data['Daily Steps'] = st.sidebar.slider("Daily Steps (Langkah Harian)", 1000, 10000, 5000, step=500)

    input_data['Sistolic'] = st.sidebar.slider("Sistolik (Tekanan Darah)", 90, 180, 120)

    input_data['Diastolic'] = st.sidebar.slider("Diastolik (Tekanan Darah)", 50, 120, 80)
    
    st.sidebar.markdown("---")
    
    return input_data

def make_prediction(data, model, preprocessor):
    """Mengubah input pengguna (dictionary) menjadi DataFrame, memproses, dan memprediksi."""
    num_cols=['Age','Sleep Duration','Quality of Sleep','Physical Activity Level','Stress Level',
          'Heart Rate','Daily Steps','Sistolic','Diastolic']
    cat_cols=['Occupation','BMI Category']
    binary_cols=['Gender']

    input_df = pd.DataFrame([data])
    input_df[num_cols]=input_df[num_cols].astype(float)
    input_df[cat_cols]=input_df[cat_cols].astype(str)
    input_df[binary_cols]=input_df[binary_cols].astype(int)

    feature_order = [
        'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 
        'Physical Activity Level', 'Stress Level', 'BMI Category', 
        'Heart Rate', 'Daily Steps', 'Sistolic', 'Diastolic'
    ]
    input_df = input_df[feature_order]

    processed_data = preprocessor.transform(input_df)
    
    prediction_proba = model.predict(processed_data)
    
    predicted_class_index = np.argmax(prediction_proba, axis=1)[0]
    
    class_mapping = {
        0: 'No Disorder (Tidak Ada Gangguan)',
        1: 'Insomnia',
        2: 'Sleep Apnea'
    }
    
    predicted_class = class_mapping.get(predicted_class_index, "Kelas tidak dikenal")
    
    return predicted_class, prediction_proba[0], class_mapping


if __name__ == "__main__":
    
    st.set_page_config(
        page_title="Prediksi Gangguan Tidur",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("😴 Aplikasi Prediksi Gangguan Tidur")
    st.markdown("Masukkan nilai fitur-fitur di **Sidebar** di sebelah kiri untuk memprediksi **Sleep Disorder**.")

    if model is None or preprocessor is None:
        st.stop()

    input_data = get_user_input_dict() 

    st.subheader("Data Input Anda")
    st.dataframe(pd.DataFrame([input_data]).T.rename(columns={0: 'Nilai'}), use_container_width=True)
    st.markdown("---")

    if st.button("🔮 **Prediksi Sleep Disorder**", use_container_width=True):
        
        predicted_disorder, probabilities, class_mapping = make_prediction(input_data, model, preprocessor)
        
        st.subheader("🎉 Hasil Prediksi")
        st.success(f"Berdasarkan input, prediksi **Sleep Disorder** adalah: **{predicted_disorder}**")
        
        st.markdown("### Detail Probabilitas")
        
        labels = list(class_mapping.values())
        
        prob_df = pd.DataFrame({
            'Gangguan': labels,
            'Probabilitas (%)': (probabilities * 100).round(2)
        }).sort_values(by='Probabilitas (%)', ascending=False)
        
        st.dataframe(prob_df, hide_index=True)