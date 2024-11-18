import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load your pre-trained model
model = tf.keras.models.load_model('model.h5')  # Ganti dengan path model Anda jika berbeda

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    image = np.asarray(image)
    image = image / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    return image

# Main Streamlit app
def main():
    st.title("Klasifikasi Gambar")
    st.write("Unggah gambar untuk diklasifikasikan oleh model.")

    uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        
        st.write("\nMemproses gambar...")
        processed_image = preprocess_image(image)

        # Prediksi gambar
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction)

        # Anda dapat mengganti label ini dengan label yang sesuai dengan model Anda
        labels = ['Amphiprion clarkii', 'Chaetodon lunulatus', 'Chaetodon trifascialis']  # Sesuaikan dengan label spesifik model Anda
        predicted_label = labels[class_index]
        
        st.write(f"Prediksi Label : {predicted_label}")
        st.write(f"Probabilitas : {prediction[0][class_index]:.2f}")

if __name__ == "__main__":
    main()

# # Main Streamlit app
# def main():
#     st.title("Klasifikasi Gambar")
#     st.write("Unggah gambar untuk melihat tampilannya.")

#     uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Tampilkan gambar yang diunggah
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        
#         st.write("\nGambar berhasil diunggah. Model klasifikasi belum tersedia.")

# if __name__ == "__main__":
#     main()
