from dataclasses import dataclass

import gensim
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance, gaussian_kde
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

nltk.download('punkt')


@dataclass
class DataGenerator:
    n_rows: int
    categories: list
    category_type: pd.CategoricalDtype

    def generate_data(self):
        half_n_rows = self.n_rows // 2
        bool_col_systemic = np.random.choice([True, False], size=half_n_rows, p=[0.8, 0.2])
        bool_col_no_systemic = np.random.choice([True, False], size=half_n_rows, p=[0.2, 0.8])

        float_col_systemic = np.random.normal(50, 10, half_n_rows).astype(np.float32)
        float_col_no_systemic = np.random.normal(20, 10, half_n_rows).astype(np.float32)

        cat_col_systemic = np.random.choice(self.categories, size=half_n_rows, p=np.random.dirichlet(np.ones(len(self.categories)), size=1)[0])
        cat_col_no_systemic = np.random.choice(self.categories, size=half_n_rows, p=np.random.dirichlet(np.ones(len(self.categories)), size=1)[0])

        data = pd.DataFrame({
            "float_feature": np.concatenate([float_col_systemic, float_col_no_systemic]),
            "bool_feature": np.concatenate([bool_col_systemic, bool_col_no_systemic]),
            "category_feature": pd.Categorical(np.concatenate([cat_col_systemic, cat_col_no_systemic]), categories=self.categories, ordered=True),
            "is_systemic_risk": np.array([True] * half_n_rows + [False] * half_n_rows)
        })

        data['verbatim_text'] = data['is_systemic_risk'].apply(self.generate_verbatim)
        return data

    def generate_verbatim(self, is_systemic):
        if is_systemic:
            topics = ["security breach", "rate changes", "hidden fees", "misleading schemes", "service responsiveness"]
        else:
            topics = ["delayed response", "minor errors", "application processing", "terms clarification", "platform usability"]
        issue = np.random.choice(topics)
        return f"Systemic Risk: Major issue with {issue}. Detailed investigation required." if is_systemic else f"Non-Systemic: Concerns about {issue}. Standard follow-up suggested."


@dataclass
class FeatureEngineering:
    # Assuming no initial parameters
    text_features: list = None
    categorical_features: list = None
    numerical_features: list = None
    label_encoders: dict = None
    tfidf_vectorizers: dict = None
    word2vec_models: dict = None

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

        # Numerical features - can add any preprocessing if needed
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

        return data

    def create_sentence_embeddings(self, data, feature, w2v_model):
        def get_sentence_embedding(sentence):
            embeddings = [w2v_model.wv[word] for word in sentence if word in w2v_model.wv]
            return np.mean(embeddings, axis=0) if embeddings else np.zeros(w2v_model.vector_size)

        data[f'{feature}_cleaned'] = data[feature].str.lower().apply(word_tokenize)
        embedding_list = data[f'{feature}_cleaned'].apply(get_sentence_embedding).tolist()

        # Creating separate columns for each dimension of the embedding
        for i in range(w2v_model.vector_size):
            data[f'{feature}_embedding_{i}'] = [embedding[i] for embedding in embedding_list]

        return data

@dataclass
class ModelTraining:
    model: XGBClassifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


@dataclass
class DriftDetection:
    @staticmethod
    def ks_test(data1, data2):
        return ks_2samp(data1, data2)

    @staticmethod
    def chi_squared_test(data1, data2):
        contingency_table = pd.crosstab(data1, data2)
        return chi2_contingency(contingency_table)

    @staticmethod
    def wasserstein_distance(data1, data2):
        return wasserstein_distance(data1, data2)

    @staticmethod
    def kld(p_data, q_data):
        p_density = gaussian_kde(p_data)
        q_density = gaussian_kde(q_data)
        x = np.linspace(min(min(p_data), min(q_data)), max(max(p_data), max(q_data)), num=1000)
        p_x = p_density(x)
        q_x = q_density(x)
        q_x = np.where(p_x != 0, q_x, np.min(p_x[p_x > 0]))
        return np.sum(p_x * np.log(p_x / q_x)) * (x[1] - x[0])

    @staticmethod
    def calculate_psi_with_smoothing(expected, actual, buckets=10, axis=0, smoothing_value=0.001):
        breakpoints = np.linspace(
            np.min([np.min(expected), np.min(actual)]),
            np.max([np.max(expected), np.max(actual)]),
            num=buckets + 1
        )
        expected_counts = np.histogram(expected, breakpoints)[0] + smoothing_value
        actual_counts = np.histogram(actual, breakpoints)[0] + smoothing_value

        expected_percents = expected_counts / np.sum(expected_counts)
        actual_percents = actual_counts / np.sum(actual_counts)

        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi_value


# Example Usage

# Data Generation (assuming this is specific and remains as provided)
data_gen = DataGenerator(n_rows=20, categories=["A", "B", "C"], category_type=pd.CategoricalDtype(categories=["A", "B", "C"], ordered=True))
data = data_gen.generate_data()

# Define feature types
text_features = ['verbatim_text']  # Update as per your data
categorical_features = ['category_feature']  # Update as per your data
numerical_features = ['float_feature']  # Update as per your data

# Feature Engineering
feat_eng = FeatureEngineering(text_features=text_features, categorical_features=categorical_features, numerical_features=numerical_features)
data = feat_eng.fit_transform(data)

# Model Training
model_training = ModelTraining()

# Define features based on transformed data
features = [col for col in data.columns if data[col].dtype in [np.float64, np.float32, np.int64, np.int32, np.uint8, np.uint16, np.uint32, np.uint64, np.bool_]]
target = 'is_systemic_risk'

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=0)
model_training.train_model(X_train, y_train)
y_pred = model_training.predict(X_test)
print("Model Accuracy:", model_training.evaluate(y_test, y_pred))

# New Data for Drift Detection
new_data = data_gen.generate_data()
new_data = feat_eng.transform(new_data)

def drift_detection_report(original_data, new_data, drift_detector, feature_types):
    report = []

    # Numerical Features Drift Detection
    for feature in feature_types['numerical']:
        ks_stat, ks_p = drift_detector.ks_test(original_data[feature], new_data[feature])
        wasserstein_dist = drift_detector.wasserstein_distance(original_data[feature], new_data[feature])
        kld = drift_detector.kld(original_data[feature], new_data[feature])
        psi = drift_detector.calculate_psi_with_smoothing(original_data[feature], new_data[feature])

        report.append({
            'Feature': feature,
            'Type': 'Numerical',
            'KS Statistic': ks_stat,
            'KS P-Value': ks_p,
            'Wasserstein Distance': wasserstein_dist,
            'KLD': kld,
            'PSI': psi
        })

    # Categorical Features Drift Detection
    for feature in feature_types['categorical']:
        chi2_stat, chi2_p, _, _ = drift_detector.chi_squared_test(original_data[feature], new_data[feature])

        report.append({
            'Feature': feature,
            'Type': 'Categorical',
            'Chi-Squared Statistic': chi2_stat,
            'Chi-Squared P-Value': chi2_p
        })

    # Textual Features Drift Detection - Can be complex and may require custom implementation
    # for feature in feature_types['textual']:
    #     Implement custom drift detection for textual features

    return pd.DataFrame(report)

if __name__ == "__main__":
    drift_det = DriftDetection()
    drift_report = drift_detection_report(data, new_data, drift_det, feature_types={'numerical': numerical_features, 'categorical': categorical_features, 'textual': text_features})

    # Printing the Table
    print("\nDrift Detection Report Table:")
    print(drift_report.to_string(index=False))
