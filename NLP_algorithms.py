from lib import *

def get_textvec_pipeline(tf_idf = True, model = 'LogisticRegression', pca = False):
    if tf_idf and not pca:
        return make_pipeline(TfidfVectorizer(min_df=0.01, max_df=1.0, stop_words='english', ngram_range=(1, 2), preprocessor=preprocess),
                             #LogisticRegression(max_iter=5000, solver='saga', n_jobs=-1))
                             RandomForestClassifier(n_estimators=100, max_depth=10, random_state=grs, n_jobs=-1))
    if not tf_idf and not pca:
        return make_pipeline(CountVectorizer(min_df=0.01, max_df=1.0, stop_words='english', ngram_range=(1, 2), preprocessor=preprocess),
                             #LogisticRegression(max_iter=5000, solver='saga', n_jobs=-1))
                            RandomForestClassifier(n_estimators=100, max_depth=10, random_state=grs, n_jobs=-1))
    if not tf_idf and pca:
        return make_pipeline(CountVectorizer(min_df=0.01, max_df=1.0, stop_words='english', ngram_range=(1, 2), preprocessor=preprocess),
                             PCA(n_components=0.1),
                             #LogisticRegression(max_iter=5000, solver='saga', n_jobs=-1))
                             RandomForestClassifier(n_estimators=100, max_depth=10, random_state=grs, n_jobs=-1))
        
def get_textvect_grid_search_params(tf_idf = True, model = 'LogisticRegression', pca = False, size = "small"):
    if not tf_idf and not pca:
        if size == "small":
            return {'countvectorizer__ngram_range': [(1, 1), (1, 2)],
                    'countvectorizer__min_df': [0.01, 0.05],
                    #'logisticregression__C': [0.01, 1, 100],
                    #'randomforest'
                    }
        if size == "large":
            return {'countvectorizer__ngram_range': [(1, 1), (1, 2)],
                    'countvectorizer__min_df': [0.01, 0.05],
                    'countvectorizer__max_df': [0.9, 1.0],
                    #'logisticregression__C': [0.01, 0.1, 1, 10, 100],
                    #'logisticregression__penalty': ['l2', 'l1']
                    }
    if tf_idf and not pca:
        # #Grid search over pca and not pca:
        return {'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
                'tfidfvectorizer__min_df': [0.01, 0.05],
                'tfidfvectorizer__max_df': [0.9, 1.0],
                #'logisticregression__C': [0.01, 1, 100],
                #'logisticregression__penalty': ['l2', 'l1']
                }
#                'pca': ['passthrough', PCA(n_components=0.1)],
#                ''
                # }
    