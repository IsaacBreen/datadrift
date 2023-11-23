from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gensim
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')

@dataclass
class FeatureEngineering:
    text_features: list
    categorical_features: list
    numerical_features: list
    boolean_features: list
    label_encoders: Optional[dict] = None
    tfidf_vectorizers: Optional[dict] = None
    word2vec_models: Optional[dict] = None

    def fit_transform(self, data):
        # Initialize dictionaries
        self.label_encoders = {}
        self.tfidf_vectorizers = {}
        self.word2vec_models = {}

        # Categorical features
        for feature in self.categorical_features:
            le = LabelEncoder()
            data[f'{feature}_encoded'] = le.fit_transform(data[feature])
            self.label_encoders[feature] = le

        # Textual features
        for feature in self.text_features:
            # TF-IDF
            tfidf = TfidfVectorizer(max_features=10)
            tfidf_matrix = tfidf.fit_transform(data[feature]).toarray()
            tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f'tfidf_{feature}_{i}' for i in range(tfidf_matrix.shape[1])])
            data = pd.concat([data, tfidf_df], axis=1)
            self.tfidf_vectorizers[feature] = tfidf

            # Word2Vec
            data[f'{feature}_cleaned'] = data[feature].str.lower().apply(word_tokenize)
            w2v_model = gensim.models.Word2Vec(sentences=data[f'{feature}_cleaned'], vector_size=100, window=5, min_count=1, workers=4)
            data = self.create_sentence_embeddings(data, feature, w2v_model)
            self.word2vec_models[feature] = w2v_model

        # Process boolean features
        for feature in self.boolean_features:
            data[feature] = data[feature].astype(int)

        # TODO: Numerical features - can add any preprocessing if needed

        return data

    def transform(self, data):
        # Apply transformations to new data
        for feature in self.categorical_features:
            le = self.label_encoders[feature]
            data[f'{feature}_encoded'] = le.transform(data[feature])

        for feature in self.text_features:
            tfidf = self.tfidf_vectorizers[feature]
            tfidf_matrix = tfidf.transform(data[feature]).toarray()
            tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f'tfidf_{feature}_{i}' for i in range(tfidf_matrix.shape[1])])
            data = pd.concat([data, tfidf_df], axis=1)
            w2v_model = self.word2vec_models[feature]
            data = self.create_sentence_embeddings(data, feature, w2v_model)

        # Process boolean features in new data
        for feature in self.boolean_features:
            data[feature] = data[feature].astype(int)

        return data

    @staticmethod
    def create_sentence_embeddings(data, feature, w2v_model):
        def get_sentence_embedding(sentence):
            embeddings = [w2v_model.wv[word] for word in sentence if word in w2v_model.wv]
            return np.mean(embeddings, axis=0) if embeddings else np.zeros(w2v_model.vector_size)
        data[f'{feature}_cleaned'] = data[feature].str.lower().apply(word_tokenize)
        embedding_list = data[f'{feature}_cleaned'].apply(get_sentence_embedding).tolist()
        embeddings_df = pd.DataFrame(embedding_list, columns=[f'{feature}_embedding_{i}' for i in range(w2v_model.vector_size)])
        data = pd.concat([data, embeddings_df], axis=1)
        return data
