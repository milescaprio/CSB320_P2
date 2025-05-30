import config

import numpy as np
import pandas as pd

import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

#CuML is a drop-in alternative to Scikit-learn for GPU-accelerated machine learning.

if config.USE_CUML:
    from cuml.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from cuml.linear_model import PCA
    from cuml.linear_model import LogisticRegression
    from cuml.model_selection import train_test_split
    from cuml.metrics import classification_report, accuracy_score, confusion_matrix
    from cuml.metrics import roc_auc_score, roc_curve
    from cuml.metrics import f1_score, precision_score, recall_score
    from cuml.metrics import make_scorer
    from cuml.pipeline import Pipeline
    from cuml.pipeline import make_pipeline 

    from cuml.decomposition import PCA
    from cuml.model_selection import GridSearchCV
else:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.metrics import make_scorer
    from sklearn.pipeline import Pipeline
    from sklearn.pipeline import make_pipeline 
    from sklearn.model_selection import GridSearchCV
    
    


grs = 42 
"""Global Random State"""

def grid_search_report(report_name, score_type, grid_search, Xtest, ytest):
    print("Report for: ", report_name)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Print classification report for the best model
    print("Linear SVM Report:")
    ypred = grid_search.predict(Xtest)
    print(classification_report(ytest, ypred))

    print("Confusion Matrix:")
    print(confusion_matrix(ytest, ypred))

    print("Best Parameters:", best_params)
    print(score_type, "Score of Best Model:", round(best_score * 100, 2), "%")

def preprocess(text, do_lemmatization=True):
    """
    Preprocess text: lowercase; remove non-alphabetic characters; remove stopwords;
    tokenize; strip whitespace; lemmatize.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text, language='english', preserve_line=True)
    stop_words = stopwords.words()
    tokens = [t.strip() for t in tokens if t not in stop_words]
    
    if not do_lemmatization:
        return ' '.join([t for t in tokens])
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join([t for t in tokens])