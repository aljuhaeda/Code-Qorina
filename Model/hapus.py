import os
import pickle
import pandas as pd

# Jalur absolut direktori tempat file CSV dan pickle berada
csv_directory = 'C:\\Code Qorina\\Model\\intermediate_csv_files'
pickle_directory = 'C:\\Code Qorina\\Model\\pickle_files'

# Daftar file CSV dan pickle yang diperlukan (sesuaikan dengan kebutuhan Anda)
required_csv_files = [
    'processed_dataframe.csv',
    'tfidf_dataframe.csv',
    'X_train_70_30.csv',
    'X_test_70_30.csv',
    'y_train_70_30.csv',
    'y_test_70_30.csv',
    'X_train_80_20.csv',
    'X_test_80_20.csv',
    'y_train_80_20.csv',
    'y_test_80_20.csv',
    'X_train_90_10.csv',
    'X_test_90_10.csv',
    'y_train_90_10.csv',
    'y_test_90_10.csv',
    'hasil_prediksi_library_70_30.csv',
    'hasil_prediksi_library_80_20.csv',
    'hasil_prediksi_library_90_10.csv',
    'hasil_prediksi_manual_70_30.csv',
    'hasil_prediksi_manual_80_20.csv',
    'hasil_prediksi_manual_90_10.csv',
    'kfold_results.csv',
    'top_words_per_class.csv'
]

required_pickle_files = [
    'processed_dataframe.pkl',
    'tfidf_vectorizer.pkl',
    'X_train_70_30.pkl',
    'X_test_70_30.pkl',
    'y_train_70_30.pkl',
    'y_test_70_30.pkl',
    'y_pred_l2_70_30.pkl',
    'X_train_80_20.pkl',
    'X_test_80_20.pkl',
    'y_train_80_20.pkl',
    'y_test_80_20.pkl',
    'y_pred_l2_80_20.pkl',
    'X_train_90_10.pkl',
    'X_test_90_10.pkl',
    'y_train_90_10.pkl',
    'y_test_90_10.pkl',
    'y_pred_l2_90_10.pkl',
    'hasil_prediksi_library_70_30.pkl',
    'hasil_prediksi_library_80_20.pkl',
    'hasil_prediksi_library_90_10.pkl',
    'hasil_prediksi_manual_70_30.pkl',
    'hasil_prediksi_manual_80_20.pkl',
    'hasil_prediksi_manual_90_10.pkl',
    'kfold_results.pkl',
    'top_words_per_class.pkl',
    'mnb_model.pkl',
    'nb_model_manual.pkl'
]

# Fungsi untuk menghapus file yang tidak diperlukan
def delete_unnecessary_files(directory, required_files):
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    all_files = set(os.listdir(directory))
    required_files_set = set(required_files)
    unnecessary_files = all_files - required_files_set
    
    for file in unnecessary_files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f'Deleted: {file_path}')

# Hapus file CSV yang tidak diperlukan
delete_unnecessary_files(csv_directory, required_csv_files)

# Hapus file pickle yang tidak diperlukan
delete_unnecessary_files(pickle_directory, required_pickle_files)
