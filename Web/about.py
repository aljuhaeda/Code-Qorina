import streamlit as st

def about_page():
    st.title("Tentang Proyek Ini")
    st.write("""
    Proyek ini menggunakan dataset teks kecelakaan kerja yang dikumpulkan dari berbagai laporan kecelakaan kerja di Indonesia.
    Pentingnya klasifikasi teks kecelakaan kerja ini adalah untuk membantu dalam analisis data kecelakaan kerja dan 
    mempermudah dalam pengambilan keputusan untuk meningkatkan keselamatan kerja.
    """)
    st.write("Penjelasan Model:")
    st.write("""
    Model yang digunakan adalah Multinomial Naive Bayes yang cocok untuk tugas klasifikasi teks. 
    Data diproses melalui beberapa tahap seperti preprocessing, transformasi TF-IDF, dan akhirnya klasifikasi menggunakan model Naive Bayes.
    """)
