import streamlit as st

def show_tentang():
    st.title("Tentang Projek")
    st.markdown("""
    ### Penjelasan Model
    Projek ini menggunakan metode Multinomial Naive Bayes untuk mengklasifikasikan teks kecelakaan kerja. 
    Model ini dilatih menggunakan dataset kecelakaan kerja yang telah dipreproses dan diubah menjadi representasi TF-IDF.

    ### Dataset
    Dataset yang digunakan dalam projek ini terdiri dari teks laporan kecelakaan kerja yang dikumpulkan dari berbagai sumber. 
    Setiap teks diklasifikasikan ke dalam beberapa kategori berdasarkan jenis kecelakaan yang terjadi.

    ### Pentingnya Klasifikasi Ini
    Klasifikasi teks kecelakaan kerja sangat penting untuk mengidentifikasi pola-pola umum dalam insiden kecelakaan, 
    yang dapat membantu dalam upaya pencegahan dan pengembangan kebijakan keselamatan kerja yang lebih baik. 
    Dengan otomatisasi klasifikasi ini, perusahaan dapat menghemat waktu dan sumber daya dalam menganalisis laporan kecelakaan.

    ### Tujuan dan Manfaat
    Projek ini bertujuan untuk:
    - Mengembangkan model klasifikasi yang akurat untuk teks kecelakaan kerja.
    - Meningkatkan efisiensi dalam pengolahan laporan kecelakaan kerja.
    - Menyediakan alat yang dapat digunakan oleh perusahaan untuk analisis lebih lanjut.

    ### Cara Kerja
    1. **Input Teks**: Pengguna memasukkan teks laporan kecelakaan kerja.
    2. **Preprocessing**: Teks dipreproses untuk menghapus karakter yang tidak perlu, mengubah teks menjadi huruf kecil, tokenisasi, penghapusan stopwords, dan stemming.
    3. **TF-IDF**: Teks yang telah dipreproses diubah menjadi representasi TF-IDF.
    4. **Klasifikasi**: Model Multinomial Naive Bayes digunakan untuk mengklasifikasikan teks ke dalam kategori yang sesuai.
    5. **Output**: Kategori kecelakaan kerja yang diprediksi ditampilkan kepada pengguna.

    ### Kesimpulan
    Dengan menggunakan metode Multinomial Naive Bayes, projek ini berhasil mengklasifikasikan teks kecelakaan kerja dengan akurasi yang baik. 
    Hasil ini diharapkan dapat membantu perusahaan dalam mengelola dan menganalisis laporan kecelakaan kerja secara lebih efisien dan efektif.
    """)
