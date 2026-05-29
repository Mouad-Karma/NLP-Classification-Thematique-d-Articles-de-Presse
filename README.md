# Projet NLP : Classification Thematique d'Articles de Presse

## Contexte et valeur
Ce projet traite un probleme central en NLP applique a la presse : classer automatiquement des articles
par thematique pour faciliter la veille, la recherche et l'exploitation d'un flux d'actualites.
La valeur est double :
- Operationaliser une chaine complete de traitement du texte, du nettoyage a l'evaluation.
- Comparer des familles de modeles (classiques, embeddings, deep learning) pour identifier la meilleure
  combinaison en precision, temps d'apprentissage et robustesse.

## Objectif
Transformer des donnees textuelles non structurees en informations categorisees exploitables.
Le projet explore une palette de methodes allant des approches statistiques classiques (BoW, TF-IDF)
aux modeles pre-entraines de type Transformer (BERT), avec des variantes de selection de features
et des architectures deep learning.

## Donnees
- BBC News
- AG News

## Approches et modeles
### Representations textuelles
- BoW
- TF-IDF
- Word2Vec (moyenne de vecteurs)
- BERT (Sentence Transformers)

### Machine Learning
- Naive Bayes
- SVM (linear)
- AdaBoost

### Deep Learning
- CNN 1D
- LSTM

## Structure du projet
- notebooks/ : analyses, EDA et experimentations
- src/ : preprocess, selection de caracteristiques et modeles
- data/ : donnees brutes et traitees

## Installation
1) Creer un environnement Python (venv ou conda).
2) Installer les dependances a partir de requirements.txt.

## Utilisation
1) Ouvrir le notebook principal et executer les cellules dans l'ordre.
2) Les notebooks EDA permettent d'analyser les jeux de donnees et de produire les fichiers traites.
3) Les scripts dans src/ contiennent les fonctions reutilisables pour preprocess et modeles.

## Tester le projet
Le projet se teste en executant le pipeline complet dans le notebook :
1) Charger les donnees traitees.
2) Lancer les blocs ML, embeddings et deep learning.
3) Comparer les tableaux de resultats et les matrices de confusion.

## Resultats attendus
- Un tableau comparatif des performances (accuracy, precision, recall, f1).
- La meilleure combinaison representation + selection + modele.
- Des matrices de confusion pour analyser les erreurs.
