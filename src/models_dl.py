import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from src.evaluation import evaluate_performance

def prepare_dl_data(X_train_text, X_test_text, y_train, y_test, max_words=5000, max_len=200):
    """
    Tokenize le texte, applique le padding et encode les catégories en nombres.
    """
    # 1. Tokenization (Création du dictionnaire)
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train_text)
    
    # 2. Transformation en séquences de nombres et Padding
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=max_len)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=max_len)
    
    # 3. Encodage des labels (Ex: 'tech' -> 0, 'business' -> 1)
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    
    return X_train_seq, X_test_seq, y_train_enc, y_test_enc, len(tokenizer.word_index) + 1, len(label_encoder.classes_)

def train_evaluate_dl(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
    """
    Entraîne un modèle Keras, chronomètre et évalue les performances.
    """
    start_time = time.time()
    
    # Entraînement du modèle (verbose=1 pour voir la barre de progression)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    
    training_time = time.time() - start_time
    
    # Prédictions (Keras retourne des probabilités, on prend la classe max)
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Évaluation via ta fonction générique !
    metrics = evaluate_performance(y_test, y_pred, training_time)
    
    return model, metrics

# ==========================================
# Définition des Architectures (CNN et LSTM)
# ==========================================

def run_cnn(X_train, y_train, X_test, y_test, vocab_size, num_classes, max_len=200):
    """Architecture CNN pour la classification de texte"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return train_evaluate_dl(model, X_train, y_train, X_test, y_test)

def run_lstm(X_train, y_train, X_test, y_test, vocab_size, num_classes, max_len=200):
    """Architecture LSTM pour la classification de texte"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len),
        # On utilise un petit LSTM pour que ça ne prenne pas des heures à tourner
        LSTM(64, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return train_evaluate_dl(model, X_train, y_train, X_test, y_test)