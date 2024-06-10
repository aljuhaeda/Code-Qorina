import streamlit as st
import pandas as pd
import numpy as np
from utils import preprocess_text, load_pickle
import os

def show_home():
    # Judul dan Deskripsi
    st.title(":construction_worker: Klasifikasi Kecelakaan Kerja pada Berita Online Menggunakaan Multinomial Naive Bayes.")
    multi = """
        Aplikasi web sederhana untuk mengklasifikasikan teks berita kecelakaan kerja sesuai ketentuan ILO (International Labour Organization) menggunakan metode Multinomial Naive Bayes, oleh :blue-background[Qorina Setyaningrum]. &mdash; :factory::hospital:
    """
    st.write(multi)

    # Input Data
    st.header("Input Text Berita Kecelakaan Kerja")
    input_text = st.text_area("Masukkan teks berita kecelakaan kerja disini:", placeholder='Input teks berita kecelakaan kerja disini...')

    # Tombol Predict
    if st.button("Predict"):
        if input_text:
            # Preprocessing
            preprocessed = preprocess_text(input_text)

            # Build absolute paths
            base_dir = os.path.dirname(os.path.abspath(__file__))
            tfidf_vectorizer_path = os.path.join(base_dir, '../Model/pickle_files/tfidf_vectorizer.pkl')
            mnb_model_path = os.path.join(base_dir, '../Model/pickle_files/mnb_model.pkl')

            # Load TF-IDF Vectorizer
            tfidf_vectorizer = load_pickle(tfidf_vectorizer_path)
            
            # Transform input text
            input_tfidf = tfidf_vectorizer.transform([preprocessed["processed_text"]])
            
            # Load Trained Model
            model = load_pickle(mnb_model_path)
            
            # Predict
            prediction = model.predict(input_tfidf)[0]
            
            # Display Prediction
            st.subheader("Prediksi Kelas")
            st.write(f"Kelas Prediksi: **{prediction}**")

            # Display Input Text
            st.subheader("Teks Input")
            st.write(input_text)
            
            # Expander for Detailed Process
            with st.expander("Proses"):
                st.write("### Hasil Preprocessing")
                st.write("**Cleaning:**", preprocessed["Cleaning"])
                st.write("---")
                st.write("**Case Folding:**", preprocessed["CaseFolding"])
                st.write("---")
                st.write("**Tokenize:**", preprocessed["Tokenize"])
                st.write("---")
                st.write("**Remove Stopwords:**", preprocessed["RemoveStopwords"])
                st.write("---")
                st.write("**Stemming:**", preprocessed["Stemming"])
                st.write("---")
                st.write("**Processed Text:**", preprocessed["processed_text"])

                st.write("### TF-IDF")
                tfidf_df = pd.DataFrame(input_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
                non_zero_tfidf = tfidf_df.loc[:, (tfidf_df != 0).any(axis=0)]  # Menampilkan hanya kolom dengan nilai TF-IDF non-zero
                st.dataframe(non_zero_tfidf)

                st.write("---")
                
                st.write("### Model Multinomial Naive Bayes")
                
                # Prior probabilities
                st.write("#### Prior Probabilities")
                class_prior = model.class_log_prior_
                class_labels = model.classes_
                for label, prior in zip(class_labels, class_prior):
                    st.write(f"{label}: {np.exp(prior):.4f}")
                st.write("---")
                
                # Likelihoods
                st.write("#### Likelihoods")
                feature_log_prob = model.feature_log_prob_
                normalized_likelihoods = {}
                for label, log_prob in zip(class_labels, feature_log_prob):
                    likelihoods = np.exp(log_prob)
                    normalized_likelihoods[label] = (likelihoods - likelihoods.min()) / (likelihoods.max() - likelihoods.min())
                
                for label, likelihoods in normalized_likelihoods.items():
                    non_zero_likelihoods = likelihoods[likelihoods > 0]  # Menghapus nilai 0
                    st.write(f"{label}: {non_zero_likelihoods[:10]} ...")  # Display only first 10 values for brevity
                st.write("---")
                
                # Posteriors
                st.write("#### Posteriors")
                pred_log_proba = model.predict_log_proba(input_tfidf)[0]
                pred_proba = np.exp(pred_log_proba)
                for label, proba in zip(class_labels, pred_proba):
                    st.write(f"{label}: {proba:.4f}")

if __name__ == "__main__":
    show_home()
