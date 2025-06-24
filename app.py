import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set judul
st.title("🧪 Deteksi Penyakit Daun Tomat")
st.write("Upload gambar daun tomat, dan model akan memprediksi penyakitnya menggunakan MobileNetV2.")

# Load model dan label (hanya sekali)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model_fixed.h5')
        with open("labels.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        return model, class_names
    except Exception as e:
        st.error(f"❌ Gagal memuat model atau label: {e}")
        return None, None

model, class_names = load_model()

# Fungsi prediksi
def predict(image: Image.Image):
    img = image.resize((224, 224)).convert('RGB')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence = float(np.max(predictions[0]))
    return predicted_label, confidence, predictions[0]

# Upload Gambar
uploaded_file = st.file_uploader("📁 Upload gambar daun tomat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Gambar yang di-upload", use_column_width=True)

    if model is not None and class_names is not None:
        with st.spinner("⏳ Memprediksi..."):
            label, confidence, scores = predict(image)
        
        st.success(f"✅ Prediksi: **{label}** ({confidence * 100:.2f}%)")

        st.subheader("🔍 Semua Skor Kelas:")
        for i, score in enumerate(scores):
            st.write(f"{class_names[i]}: {score:.4f}")
    else:
        st.error("❌ Model belum dimuat.")
