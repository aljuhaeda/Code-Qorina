import pickle
import pandas as pd

# Fungsi untuk memuat model dan vectorizer
def load_model():
    with open('C:\\Code Qorina\\Model\\pickle_files\\mnb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('C:\\Code Qorina\\Model\\pickle_files\\tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Fungsi untuk memuat dataframe hasil
def load_dataframe(file_name):
    return pd.read_csv(f'C:\\Code Qorina\\Model\\intermediate_csv_files\\{file_name}.csv')
