import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import numpy as np

# Téléchargement des mots vides (quiet=True pour ne pas spammer la console)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """Nettoie le texte : minuscules, ponctuation et stopwords."""
    text = str(text).lower()
    # Enlever la ponctuation et les chiffres
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", " ", text)
    # Enlever les stopwords et les mots trop courts
    words = text.split()
    clean_words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(clean_words)




def apply_bow(text_series, max_features=1500):
    """
    Applique la vectorisation Bag of Words (BoW).
    Retourne la matrice transformée et le vectorizer.
    """
    vectorizer = CountVectorizer(max_features=max_features)
    X_bow = vectorizer.fit_transform(text_series)
    return X_bow, vectorizer





def apply_tfidf(text_series, max_features=1500):
    """
    Applique la vectorisation TF-IDF.
    Retourne la matrice transformée et le vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(text_series)
    return X_tfidf, vectorizer





def apply_word2vec(text_series, vector_size=100, window=5, min_count=1):
    """
    Applique la vectorisation Word2Vec.
    Retourne les vecteurs moyens pour chaque document.
    """
    # Tokeniser les textes
    tokenized_texts = [text.split() for text in text_series]
    # Entraîner le modèle Word2Vec
    model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count)
    # Obtenir les vecteurs moyens pour chaque document
    X_word2vec = []
    for tokens in tokenized_texts:
        word_vectors = [model.wv[word] for word in tokens if word in model.wv]
        if word_vectors:
            X_word2vec.append(np.mean(word_vectors, axis=0))
        else:
            X_word2vec.append(np.zeros(vector_size))
    return np.array(X_word2vec), model





def apply_bert(text_series, model_name='all-MiniLM-L6-v2'):
    """
    Applique la vectorisation avec un modèle type BERT (Sentence Transformers).
    Retourne la matrice des embeddings pour chaque document.
    
    'all-MiniLM-L6-v2' est une version légère et très rapide de BERT, 
    parfaite pour extraire des features pour le Machine Learning.
    """
    print(f"Chargement du modèle BERT ({model_name})... Cela peut prendre quelques secondes.")
    model = SentenceTransformer(model_name)
    
    print("Encodage des textes avec BERT en cours...")
    # L'encodage transforme la liste de textes en une matrice numpy (embeddings)
    # show_progress_bar=True est très pratique pour suivre l'avancement !
    X_bert = model.encode(text_series.tolist(), show_progress_bar=True)
    
    return X_bert, model
