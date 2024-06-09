import streamlit as st
from utils import load_model, load_dataframe

def home_page():
    st.title("Klasifikasi Teks Kecelakaan Kerja dengan Metode Multinomial Naive Bayes")
    st.write("""
    Proyek ini merupakan skripsi S1 di Universitas Islam Negeri Maulana Malik Ibrahim Malang. 
    Tujuan dari proyek ini adalah untuk mengklasifikasikan teks kecelakaan kerja menggunakan metode Multinomial Naive Bayes.
    """)

    st.subheader("Input Data Text")
    input_text = st.text_area("Masukkan teks kecelakaan kerja di sini:")

    if st.button("Predict"):
        if input_text.strip() == "":
            st.error("Mohon masukkan teks terlebih dahulu.")
        else:
            model, vectorizer = load_model()
            with st.spinner("Melakukan prediksi..."):
                input_vec = vectorizer.transform([input_text])
                prediction = model.predict(input_vec)
                st.success(f"Kelas Prediksi: **{prediction[0]}**")

    st.subheader("Proses")
    process = st.selectbox("Pilih Proses", ["Teks Input", "Hasil Preprocessing", "TF-IDF", "MNB - Prior", "MNB - Likelihoods", "MNB - Posteriors"])
    
    if process == "Teks Input":
        st.write(input_text)
    elif process == "Hasil Preprocessing":
        df = load_dataframe('processed_dataframe')
        st.dataframe(df)
    elif process == "TF-IDF":
        df = load_dataframe('tfidf_dataframe')
        st.dataframe(df)
    elif process == "MNB - Prior":
        st.write("Prior probabilities:")
        model, _ = load_model()
        st.write(model.class_log_prior_)
    elif process == "MNB - Likelihoods":
        st.write("Feature log probabilities:")
        model, _ = load_model()
        st.write(model.feature_log_prob_)
    elif process == "MNB - Posteriors":
        if input_text.strip() == "":
            st.error("Mohon masukkan teks terlebih dahulu untuk menghitung posteriors.")
        else:
            model, vectorizer = load_model()
            input_vec = vectorizer.transform([input_text])
            st.write("Posterior probabilities:")
            st.write(model.predict_proba(input_vec))
