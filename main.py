import numpy as np
import pandas as pd
from dataclasses import dataclass

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
import gensim
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance, gaussian_kde
import seaborn as sns

# Ensure NLTK punkt is downloaded
import nltk
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
    categories: list
    le: LabelEncoder = LabelEncoder()
    tfidf: TfidfVectorizer = TfidfVectorizer(max_features=10)
    word2vec_model: gensim.models.Word2Vec = None

    def fit_transform(self, data):
        self.le.fit(self.categories)
        data['category_feature_encoded'] = self.le.transform(data['category_feature'])
        tfidf_matrix = self.tfidf.fit_transform(data['verbatim_text']).toarray()
        tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
        data = pd.concat([data, tfidf_df], axis=1)
        self.train_word2vec(data)
        data = self.create_sentence_embeddings(data)
        return data

    def transform(self, data):
        data['category_feature_encoded'] = self.le.transform(data['category_feature'])
        tfidf_matrix = self.tfidf.transform(data['verbatim_text']).toarray()
        tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
        data = pd.concat([data, tfidf_df], axis=1)
        self.create_sentence_embeddings(data)
        return data

    def train_word2vec(self, data):
        data['verbatim_cleaned'] = data['verbatim_text'].str.lower().apply(word_tokenize)
        self.word2vec_model = gensim.models.Word2Vec(sentences=data['verbatim_cleaned'], vector_size=100, window=5, min_count=1, workers=4)

    def create_sentence_embeddings(self, data):
        def get_sentence_embedding(sentence):
            embeddings = [self.word2vec_model.wv[word] for word in sentence if word in self.word2vec_model.wv]
            return np.mean(embeddings, axis=0) if embeddings else np.zeros(self.word2vec_model.vector_size)

        data['verbatim_cleaned'] = data['verbatim_text'].str.lower().apply(word_tokenize)
        data['verbatim_embedding'] = data['verbatim_cleaned'].apply(get_sentence_embedding)
        embeddings_df = pd.DataFrame(data['verbatim_embedding'].tolist())
        embeddings_df.columns = [f"embedding_{i}" for i in range(embeddings_df.shape[1])]
        return pd.concat([data, embeddings_df], axis=1)


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
@dataclass
class DriftDetection:
    @staticmethod
    def ks_test(data1, data2):
        return ks_2samp(data1, data2)

    @staticmethod
    def chi_squared_test(data1, data2):
        return chi2_contingency(pd.crosstab(data1, data2))

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
categories = ["Loan", "Account", "Credit Card", "Mortgage", "Investment"]
category_type = pd.CategoricalDtype(categories=categories, ordered=True)

# Data Generation
data_gen = DataGenerator(n_rows=20, categories=categories, category_type=category_type)
data = data_gen.generate_data()

# Feature Engineering
feat_eng = FeatureEngineering(categories=categories)
data = feat_eng.fit_transform(data)

# Model Training
model_training = ModelTraining()
features = ['float_feature', 'bool_feature', 'category_feature_encoded'] + [f"tfidf_{i}" for i in range(10)] + [f"embedding_{i}" for i in range(100)]
target = 'is_systemic_risk'

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=0)
model_training.train_model(X_train, y_train)
y_pred = model_training.predict(X_test)
print("Model Accuracy:", model_training.evaluate(y_test, y_pred))

# New Data for Drift Detection
new_data = data_gen.generate_data()
new_data = feat_eng.transform(new_data)

def drift_detection_report(data, new_data, drift_detector):
    # Calculate drift metrics
    ks_stat, ks_p = drift_detector.ks_test(data['float_feature'], new_data['float_feature'])
    chi2_stat, chi2_p, _, _ = drift_detector.chi_squared_test(data['bool_feature'], new_data['bool_feature'])
    wasserstein_dist = drift_detector.wasserstein_distance(data['float_feature'], new_data['float_feature'])
    kld = drift_detector.kld(data['float_feature'], new_data['float_feature'])
    psi_float_feature = drift_detector.calculate_psi_with_smoothing(data['float_feature'], new_data['float_feature'])

    # Define thresholds for interpretation
    ks_threshold = 0.05
    chi2_threshold = 0.05
    wasserstein_threshold = 10
    kld_threshold = 0.1
    psi_threshold = 0.2

    # Prepare a DataFrame for the report
    drift_report = pd.DataFrame({
        'Metric': ['KS Test', 'Chi-Squared Test', 'Wasserstein Distance', 'KLD', 'PSI'],
        'Data Feature': ['float_feature', 'bool_feature', 'float_feature', 'float_feature', 'float_feature'],
        'Value': [ks_stat, chi2_stat, wasserstein_dist, kld, psi_float_feature],
        'P-Value': [ks_p if ks_p is not None else "N/A", chi2_p if chi2_p is not None else "N/A", "N/A", "N/A", "N/A"],
        'Threshold': [ks_threshold, chi2_threshold, wasserstein_threshold, kld_threshold, psi_threshold],
        'Drift': [
            ks_stat > ks_threshold,
            chi2_p < chi2_threshold if chi2_p is not None else False,
            wasserstein_dist > wasserstein_threshold,
            kld > kld_threshold,
            psi_float_feature > psi_threshold
        ]
    })

    return drift_report

if __name__ == "__main__":
    drift_det = DriftDetection()
    drift_report = drift_detection_report(data, new_data, drift_det)

    # Printing the Table
    print("\nDrift Detection Report Table:")
    print(drift_report.to_string(index=False))
