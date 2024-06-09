from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib  # Untuk menyimpan dan memuat model

def calculate_tfidf(data, save_path='tfidf_vectorizer.joblib', **kwargs):
    """
    Menghitung TF-IDF pada data pelatihan dan menyimpan vectorizer.

    Parameters:
    data (array-like): Data teks untuk dihitung TF-IDF.
    save_path (str): Jalur untuk menyimpan vectorizer. Default adalah 'tfidf_vectorizer.joblib'.
    kwargs: Argumen tambahan untuk TfidfVectorizer.

    Returns:
    pd.DataFrame: DataFrame dengan matriks TF-IDF.
    """
    vectorizer = TfidfVectorizer(**kwargs)
    tfidf_matrix = vectorizer.fit_transform(data)
    terms = vectorizer.get_feature_names_out()
    joblib.dump(vectorizer, save_path)  # Menyimpan vectorizer
    return pd.DataFrame(data=tfidf_matrix.toarray(), columns=terms)

def transform_new_data(new_data, load_path='tfidf_vectorizer.joblib'):
    """
    Menggunakan vectorizer yang telah dilatih pada data baru.

    Parameters:
    new_data (array-like): Data teks baru untuk di-transformasi.
    load_path (str): Jalur untuk memuat vectorizer yang telah dilatih. Default adalah 'tfidf_vectorizer.joblib'.

    Returns:
    pd.DataFrame: DataFrame dengan matriks TF-IDF dari data baru.
    """
    try:
        vectorizer = joblib.load(load_path)  # Memuat vectorizer yang telah dilatih
    except FileNotFoundError:
        print(f"Vectorizer file not found at path: {load_path}")
        return pd.DataFrame()  # Mengembalikan DataFrame kosong jika file tidak ditemukan

    tfidf_matrix = vectorizer.transform(new_data)
    terms = vectorizer.get_feature_names_out()
    return pd.DataFrame(data=tfidf_matrix.toarray(), columns=terms)

# Contoh penggunaan
if __name__ == "__main__":
    df = pd.DataFrame({'text': ["Ini adalah contoh teks", "Teks lain dengan kata-kata yang berbeda"]})
    tfidf_df = calculate_tfidf(df['text'], max_features=5000)

    new_texts = ["Teks baru untuk di-transformasi", "Contoh teks lainnya"]
    new_tfidf_df = transform_new_data(new_texts)
    print(new_tfidf_df)
