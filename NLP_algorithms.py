from lib import (
    pd,
    BaseEstimator,
    TransformerMixin,
    Pipeline,
    TfidfVectorizer,
    CountVectorizer,
    ColumnTransformer,
    OneHotEncoder,
    RandomForestClassifier,
    np,
    TextBlob,
    preprocess,
    grs,
)


def get_textvec_pipeline(
    tf_idf=True,
    model="LogisticRegression",
    pca=False,
    text_column="text",
    categorical_columns=["subject"],
    target_column="label",
):
    class SentimentFeatures(BaseEstimator, TransformerMixin):
        """Custom Transformer for Sentiment Analysis"""

        def __init__(self, text_column):
            self.text_column = text_column

        def fit(self, X, y=None):
            return self

        def get_sentiment(self, text):
            blob = TextBlob(text)
            overall = blob.sentiment.polarity
            sentences = blob.sentences
            first = sentences[0].sentiment.polarity if sentences else 0
            last = (
                sentences[-1].sentiment.polarity if len(sentences) > 1 else 0
            )
            return [overall, first, last]

        def transform(self, X):
            print(
                f"[DEBUG] type(X): \
                    {type(X)}, shape: {getattr(X, 'shape', None)}"
            )
            if isinstance(X, pd.DataFrame):
                text_series = X.iloc[:, 0]
            elif isinstance(X, np.ndarray):
                # fall back to column index 0
                text_series = pd.Series(X[:, 0])
            else:
                raise ValueError(
                    "SentimentFeatures expected DataFrame or 2D array"
                )

            sentiments = text_series.apply(self.get_sentiment)
            return pd.DataFrame(
                sentiments.tolist(),
                columns=[
                    "overall_sentiment",
                    "first_sentiment",
                    "last_sentiment",
                ],
            )

    if tf_idf:
        text_vect = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        min_df=0.01,
                        max_df=1.0,
                        ngram_range=(1, 2),
                        preprocessor=preprocess,
                        max_features=5000,
                    ),
                )
            ]
        )
    else:
        text_vect = Pipeline(
            [
                (
                    "countvect",
                    CountVectorizer(
                        min_df=0.01,
                        max_df=1.0,
                        ngram_range=(1, 2),
                        preprocessor=preprocess,
                    ),
                )
            ]
        )

    categorical = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    sentiment = Pipeline(
        [("sentiment_extractor", SentimentFeatures(text_column=text_column))]
    )

    # Combine all features
    processor = ColumnTransformer(
        transformers=[
            (
                ("tfidf", text_vect, text_column)
                if tf_idf
                else ("countvect", text_vect, text_column)
            ),
            ("cat", categorical, categorical_columns),
            ("sentiment", sentiment, [text_column]),
        ]
    )

    if not pca:
        # Final Pipeline
        model = Pipeline(
            [
                ("process", processor),
                (
                    "regressor",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=grs,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        return model

    # TODO: Try PCA ( See previous code attempts in
    # first draft file and in first draft's recycled.py file)


def get_textvect_grid_search_params(
    tf_idf=True, model="LogisticRegression", pca=False, size="small"
):
    if not tf_idf and not pca:
        if size == "small":
            return {
                "process__countvect__countvect__ngram_range": [(1, 1), (1, 2)],
                "process__countvect__countvect__min_df": [0.01, 0.05],
                # Use grid search on random forest?
            }
        if size == "large":
            return {
                "process__countvect__countvect__ngram_range": [(1, 1), (1, 2)],
                "process__countvect__countvect__min_df": [0.01, 0.05],
                "process__countvect__countvect__max_df": [0.9, 1.0],
            }
    if tf_idf and not pca:
        # #Grid search over pca and not pca:
        return {
            "process__tfidf__tfidf__ngram_range": [(1, 1), (1, 2)],
            "process__tfidf__tfidf__min_df": [0.01, 0.05],
            "process__tfidf__tfidf__max_df": [0.9, 1.0],
        }
