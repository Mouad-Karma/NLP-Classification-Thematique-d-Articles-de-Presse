from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import LatentDirichletAllocation

def select_chi2(X_train, y_train, X_test, k=1000):
    """
    Applique le test Chi-Deux pour sélectionner les 'k' meilleures features.
    Attention: X_train doit être positif (ex: BoW ou TF-IDF).
    """
    # Initialisation de l'outil de sélection
    selector = SelectKBest(score_func=chi2, k=k)
    
    # On apprend les meilleures features sur le Train et on transforme
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # On applique la même sélection sur le Test
    X_test_selected = selector.transform(X_test)
    
    return X_train_selected, X_test_selected, selector





def select_mutual_info(X_train, y_train, X_test, k=1000):
    """
    Applique l'Information Mutuelle pour sélectionner les 'k' meilleures features.
    """
    # L'information mutuelle est plus lente à calculer que le Chi-2
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    return X_train_selected, X_test_selected, selector





def apply_lda(X_train, X_test, n_components=100):
    """
    Applique la Latent Dirichlet Allocation (LDA) pour réduire la dimensionnalité
    en créant 'n_components' topics (thèmes).
    """
    # LDA crée des thèmes à partir des mots
    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    
    X_train_lda = lda.fit_transform(X_train)
    X_test_lda = lda.transform(X_test)
    
    return X_train_lda, X_test_lda, lda