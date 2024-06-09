import numpy as np

class MultinomialNaiveBayes:
    def __init__(self, alpha=1):
        """
        Inisialisasi model dengan parameter smoothing Laplace.
        
        Parameters:
        alpha (float): Parameter untuk smoothing Laplace. Default adalah 1.
        """
        self.alpha = alpha  # Parameter untuk smoothing Laplace
        self.class_probs = {}  # Probabilitas prior
        self.feature_probs = {}  # Probabilitas likelihood

    def fit(self, X, y):
        """
        Melatih model dengan data training.
        
        Parameters:
        X (array-like): Matriks TF-IDF.
        y (array-like): Label kelas.
        """
        num_docs, num_features = X.shape
        unique_classes = np.unique(y)

        # Perhitungan probabilitas prior
        for label in unique_classes:
            self.class_probs[label] = np.sum(y == label) / num_docs

        # Perhitungan probabilitas likelihood
        for label in unique_classes:
            class_docs = X[y == label]
            total_word_count = np.sum(class_docs)
            self.feature_probs[label] = (np.sum(class_docs, axis=0) + self.alpha) / (total_word_count + self.alpha * num_features)

    def predict(self, X):
        """
        Memprediksi label untuk data uji.
        
        Parameters:
        X (array-like): Matriks TF-IDF uji.
        
        Returns:
        predictions (list): Daftar prediksi label.
        """
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        
        predictions = []
        for doc in X:
            posterior_probs = {}
            for label, class_prob in self.class_probs.items():
                # Perhitungan probabilitas posterior tanpa normalisasi
                # Gunakan dot product untuk menghitung perkalian matriks
                likelihood = self.feature_probs[label]
                likelihood = np.where(likelihood == 0, 1e-10, likelihood)  # Menghindari log(0)
                posterior_prob = np.log(class_prob) + np.sum(doc * np.log(likelihood))
                posterior_probs[label] = posterior_prob

            # Pilih kelas dengan probabilitas posterior tertinggi
            predicted_label = max(posterior_probs, key=posterior_probs.get)
            predictions.append(predicted_label)

        return predictions
