import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from custom_stemming import custom_stemming  # Import custom stemming function

nltk.download('stopwords')

stopword_list = set(stopwords.words('indonesian'))
stopword_list.update(["yg", "dg", "rt", "di", "mrk", "nya", "dgn", "org", 'yang', 'untuk', 'dari', 'dengan', 'ada', 'ini', 'itu', 'dan', 'di'])

def cleaning(text):
    text = re.sub(r'http\S+', '', text)  # Menghapus URL
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
    return [custom_stemming(token) for token in tokens]

def preprocess_dataframe(df, text_column, label_column):
    df['Cleaning'] = df[text_column].apply(cleaning)
    df['CaseFolding'] = df['Cleaning'].apply(case_folding)
    df['Tokenize'] = df['CaseFolding'].apply(tokenize)
    df['RemoveStopwords'] = df['Tokenize'].apply(remove_stopwords)
    df['Stemming'] = df['RemoveStopwords'].apply(stemming_text)
    df['ProcessedText'] = df['Stemming'].apply(lambda x: ' '.join(x))
    return df[[text_column, label_column, 'Cleaning', 'CaseFolding', 'Tokenize', 'RemoveStopwords', 'Stemming', 'ProcessedText']]
