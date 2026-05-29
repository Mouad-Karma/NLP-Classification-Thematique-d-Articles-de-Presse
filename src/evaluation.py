from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_performance(y_test, y_pred, training_time):
    """
    Calcule toutes les métriques d'évaluation pour un modèle donné.
    Retourne un dictionnaire contenant les résultats.
    """
    # Utilisation de average='macro' pour la classification multiclasse
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Stockage dans un dictionnaire structuré
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'training_time': training_time,
        'confusion_matrix': conf_matrix
    }
    
    return metrics