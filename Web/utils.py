import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

nltk.download('stopwords')

# Membuat stopword list sekali saja
stopword_list = set(stopwords.words('indonesian'))
stopword_list.update(["yg", "dg", "rt", "di", "mrk", "nya", "dgn", "org", 'yang', 'untuk', 'dari', 'dengan', 'ada', 'ini', 'itu', 'dan', 'di'])

# Membuat stemmer sekali saja
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def cleaning(text):
    text = re.sub(r'#\S+', '', text)  # Menghapus hashtag
    text = re.sub(r'\'\w+', '', text)  # Menghapus karakter setelah apostrof
    text = re.sub(r'\w*\d+\w*', '', text)  # Menghapus angka
    text = re.sub(r'\s{2,}', ' ', text)  # Menghapus spasi berlebih
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca
    text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Menghapus karakter tunggal
    return text.strip()

def case_folding(text):
    return text.lower()

def tokenize(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stopword_list]

def stemming_text(tokens):
    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    cleaned = cleaning(text)
    case_folded = case_folding(cleaned)
    tokenized = tokenize(case_folded)
    no_stopwords = remove_stopwords(tokenized)
    stemmed = stemming_text(no_stopwords)
    processed_text = ' '.join(stemmed)
    return {
        "Cleaning": cleaned,
        "CaseFolding": case_folded,
        "Tokenize": tokenized,
        "RemoveStopwords": no_stopwords,
        "Stemming": stemmed,
        "processed_text": processed_text
    }
