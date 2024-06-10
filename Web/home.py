import streamlit as st
import pandas as pd
import numpy as np
from utils import preprocess_text, load_pickle

# Judul dan Deskripsi
st.title("Klasifikasi Teks Kecelakaan Kerja dengan Metode Multinomial Naive Bayes")
st.markdown("""
Projek ini merupakan skripsi S1 pada Universitas Islam Negeri Maulana Malik Ibrahim Malang.
Projek ini bertujuan untuk mengklasifikasikan teks kecelakaan kerja menggunakan metode Multinomial Naive Bayes.
""")

# Input Data
st.header("Masukkan Data Teks")
input_text = st.text_area("Masukkan teks kecelakaan kerja di sini:")

# Tombol Predict
if st.button("Predict"):
    if input_text:
        # Preprocessing
        preprocessed = preprocess_text(input_text)

        # Load TF-IDF Vectorizer
        tfidf_vectorizer = load_pickle('../Model/pickle_files/tfidf_vectorizer.pkl')
        
        # Transform input text
        input_tfidf = tfidf_vectorizer.transform([preprocessed["processed_text"]])
        
        # Load Trained Model
        model = load_pickle('../Model/pickle_files/mnb_model.pkl')
        
        # Predict
        prediction = model.predict(input_tfidf)[0]
        
        # Display Prediction
        st.subheader("Prediksi Kelas")
        st.write(f"Kelas Prediksi: **{prediction}**")
        
        # Expanders for Preprocessing Steps
        with st.expander("Teks Input"):
            st.write(input_text)
            
        with st.expander("Hasil Preprocessing"):
            st.write("Cleaning:", preprocessed["Cleaning"])
            st.write("Case Folding:", preprocessed["CaseFolding"])
            st.write("Tokenize:", preprocessed["Tokenize"])
            st.write("Remove Stopwords:", preprocessed["RemoveStopwords"])
            st.write("Stemming:", preprocessed["Stemming"])
            st.write("Processed Text:", preprocessed["processed_text"])

        with st.expander("TF-IDF"):
            st.write(pd.DataFrame(input_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()))
            
        with st.expander("Model Multinomial Naive Bayes"):
            # Prior probabilities
            st.write("#### Prior Probabilities")
            class_prior = model.class_log_prior_
            class_labels = model.classes_
            for label, prior in zip(class_labels, class_prior):
                st.write(f"{label}: {np.exp(prior):.4f}")
            
            # Likelihoods
            st.write("#### Likelihoods")
            feature_log_prob = model.feature_log_prob_
            for label, log_prob in zip(class_labels, feature_log_prob):
                st.write(f"{label}: {np.exp(log_prob)}")
            
            # Posteriors
            st.write("#### Posteriors")
            pred_log_proba = model.predict_log_proba(input_tfidf)[0]
            pred_proba = np.exp(pred_log_proba)
            for label, proba in zip(class_labels, pred_proba):
                st.write(f"{label}: {proba:.4f}")
