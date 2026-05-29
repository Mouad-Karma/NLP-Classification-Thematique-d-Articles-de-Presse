import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

# On importe ta nouvelle fonction d'évaluation depuis le fichier séparé !
from src.evaluation import evaluate_performance

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Entraîne le modèle, chronomètre le temps, fait les prédictions,
    et appelle le fichier d'évaluation pour calculer les scores.
    """
    # 1. Entraînement et chronométrage
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # 2. Prédictions
    y_pred = model.predict(X_test)
    
    # 3. Évaluation (en appelant notre autre fichier)
    metrics = evaluate_performance(y_test, y_pred, training_time)
    
    return model, metrics


# Initialisation des Modèles

def run_naive_bayes(X_train, y_train, X_test, y_test):
    model = MultinomialNB()
    return train_and_evaluate(model, X_train, y_train, X_test, y_test)

def run_svm(X_train, y_train, X_test, y_test):
    model = SVC(kernel='linear', random_state=42)
    return train_and_evaluate(model, X_train, y_train, X_test, y_test)

def run_adaboost(X_train, y_train, X_test, y_test):
    model = AdaBoostClassifier(n_estimators=50, random_state=42)
    return train_and_evaluate(model, X_train, y_train, X_test, y_test)