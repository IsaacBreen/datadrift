import pandas as pd
import numpy as np
from pandas import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
import gensim

# Set seed for reproducibility
np.random.seed(0)

# Number of rows
n_rows = 20

# Generating data
bool_col_systemic = np.random.choice([True, False], size=n_rows//2, p=[0.8, 0.2])
bool_col_no_systemic = np.random.choice([True, False], size=n_rows//2, p=[0.2, 0.8])

float_col_systemic = np.random.normal(50, 10, n_rows//2)
float_col_no_systemic = np.random.normal(20, 10, n_rows//2)

categories = ["Loan", "Account", "Credit Card", "Mortgage", "Investment"]
category_type = CategoricalDtype(categories=categories, ordered=True)
cat_col_systemic = np.random.choice(categories, size=n_rows//2, p=np.random.dirichlet(np.ones(len(categories)), size=1)[0])
cat_col_no_systemic = np.random.choice(categories, size=n_rows//2, p=np.random.dirichlet(np.ones(len(categories)), size=1)[0])

# Function to generate verbatim text
def generate_verbatim(is_systemic):
    if is_systemic:
        topics = ["security breach", "rate changes", "hidden fees", "misleading schemes", "service responsiveness"]
        issue = np.random.choice(topics)
        return f"Systemic Risk: Major issue with {issue}. Detailed investigation required."
    else:
        topics = ["delayed response", "minor errors", "application processing", "terms clarification", "platform usability"]
        issue = np.random.choice(topics)
        return f"Non-Systemic: Concerns about {issue}. Standard follow-up suggested."

# Creating DataFrame
data = pd.DataFrame({
    "float_feature": np.concatenate([float_col_systemic, float_col_no_systemic]),
    "bool_feature": np.concatenate([bool_col_systemic, bool_col_no_systemic]),
    "category_feature": pd.Categorical(np.concatenate([cat_col_systemic, cat_col_no_systemic]), categories=categories, ordered=True),
    "is_systemic_risk": np.array([True] * (n_rows // 2) + [False] * (n_rows // 2))
})

# Generating verbatim text
data['verbatim_text'] = data['is_systemic_risk'].apply(generate_verbatim)

# Encoding categorical and text features
le = LabelEncoder()
tfidf = TfidfVectorizer(max_features=10)

# Fit the LabelEncoder with all possible categories
le.fit(categories)  # Fit with all possible categories

# Encoding the categorical column
data['category_feature_encoded'] = le.transform(data['category_feature'])

# Encoding the verbatim text column using TF-IDF
tfidf_matrix = tfidf.fit_transform(data['verbatim_text']).toarray()
tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])

# Generating sentence embeddings for the verbatim text column using Word2Vec
# Step 1: Preprocess Text Data (Basic)
data['verbatim_cleaned'] = data['verbatim_text'].str.lower()
data['verbatim_cleaned'] = data['verbatim_cleaned'].apply(word_tokenize)

# Step 2: Train Word2Vec Model
word2vec_model = gensim.models.Word2Vec(sentences=data['verbatim_cleaned'], vector_size=100, window=5, min_count=1, workers=4)

# Step 3: Create Sentence Embeddings
def get_sentence_embedding(sentence, model):
    embeddings = [model.wv[word] for word in sentence if word in model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

data['verbatim_embedding'] = data['verbatim_cleaned'].apply(lambda x: get_sentence_embedding(x, word2vec_model))

# Step 4: Prepare the Dataset
# Convert the list of embeddings into a DataFrame
embeddings_df = pd.DataFrame(data['verbatim_embedding'].tolist())

# Combining the sentence embeddings with the other features
data_combined = pd.concat([data, embeddings_df], axis=1)

# Defining the feature columns and the target column
features = ['float_feature', 'bool_feature', 'category_feature_encoded'] + list(embeddings_df.columns)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_combined[features], data_combined[target], test_size=0.2, random_state=0)

# Initializing the XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Training the model
xgb_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = xgb_model.predict(X_test)

# Calculating the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Generating a new dataset with a changed distribution

# Changing the parameters for the boolean, float, and categorical features to simulate data drift
new_bool_col_systemic = np.random.choice([True, False], size=n_rows//2, p=[0.6, 0.4])
new_bool_col_no_systemic = np.random.choice([True, False], size=n_rows//2, p=[0.3, 0.7])

new_float_col_systemic = np.random.normal(60, 15, n_rows//2)  # Increased mean and standard deviation
new_float_col_no_systemic = np.random.normal(30, 15, n_rows//2)

# New probabilities for the categorical feature
new_cat_col_systemic = np.random.choice(categories, size=n_rows//2, p=np.random.dirichlet(np.ones(len(categories)), size=1)[0])
new_cat_col_no_systemic = np.random.choice(categories, size=n_rows//2, p=np.random.dirichlet(np.ones(len(categories)), size=1)[0])

# Creating the new DataFrame
new_data = pd.DataFrame({
    "float_feature": np.concatenate([new_float_col_systemic, new_float_col_no_systemic]),
    "bool_feature": np.concatenate([new_bool_col_systemic, new_bool_col_no_systemic]),
    "category_feature": np.concatenate([new_cat_col_systemic, new_cat_col_no_systemic]),
    "is_systemic_risk": np.array([True] * (n_rows // 2) + [False] * (n_rows // 2))
})

# Generating verbatim text for the new data
new_data['verbatim_text'] = new_data['is_systemic_risk'].apply(generate_verbatim)

# Preprocessing the new data
new_data['category_feature_encoded'] = le.transform(new_data['category_feature'])
new_tfidf_matrix = tfidf.transform(new_data['verbatim_text']).toarray()
new_tfidf_df = pd.DataFrame(new_tfidf_matrix, columns=[f"tfidf_{i}" for i in range(new_tfidf_matrix.shape[1])])
new_data_combined = pd.concat([new_data, new_tfidf_df], axis=1)

# Extracting features for the new data
X_new = new_data_combined[features]
y_new = new_data_combined[target]

# Running inference on the new data
y_new_pred = xgb_model.predict(X_new)

# Calculating the new accuracy
new_accuracy = accuracy_score(y_new, y_new_pred)
print("New Data Model Accuracy:", new_accuracy)


from scipy.stats import ks_2samp, chi2_contingency

# KS Test for the float_feature
ks_statistic, ks_p_value = ks_2samp(data['float_feature'], new_data['float_feature'])
print("KS Test for float_feature:")
print("Statistic:", ks_statistic, "P-Value:", ks_p_value)

# Preparing data for Chi-Squared Test for the bool_feature
bool_feature_contingency = pd.crosstab(data['bool_feature'], new_data['bool_feature'])
chi2_stat_bool, chi2_p_bool, _, _ = chi2_contingency(bool_feature_contingency)
print("\nChi-Squared Test for bool_feature:")
print("Statistic:", chi2_stat_bool, "P-Value:", chi2_p_bool)

# Preparing data for Chi-Squared Test for the category_feature
category_feature_contingency = pd.crosstab(data['category_feature'], new_data['category_feature'])
chi2_stat_cat, chi2_p_cat, _, _ = chi2_contingency(category_feature_contingency)
print("\nChi-Squared Test for category_feature:")
print("Statistic:", chi2_stat_cat, "P-Value:", chi2_p_cat)


def calculate_psi_with_smoothing(expected, actual, buckets=10, axis=0, smoothing_value=0.001):
    """Calculate the PSI for a single variable with smoothing to handle zero values.

    Args:
        expected (array-like): The original data (expected distribution).
        actual (array-like): The new data (actual distribution).
        buckets (int): The number of intervals to divide the data into.
        axis (int): Axis along which the PSI is calculated.
        smoothing_value (float): A small constant added to avoid division by zero.

    Returns:
        float: The PSI value.
    """
    # Bin the data
    breakpoints = np.linspace(
        np.min([np.min(expected), np.min(actual)]),
        np.max([np.max(expected), np.max(actual)]),
        num=buckets + 1
        )
    expected_counts = np.histogram(expected, breakpoints)[0] + smoothing_value
    actual_counts = np.histogram(actual, breakpoints)[0] + smoothing_value

    # Calculate PSI
    expected_percents = expected_counts / np.sum(expected_counts)
    actual_percents = actual_counts / np.sum(actual_counts)

    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value

# Example: Calculating PSI for float_feature with smoothing
psi_float_feature = calculate_psi_with_smoothing(data['float_feature'], new_data['float_feature'])
print("PSI for float_feature:", psi_float_feature)


from scipy.stats import wasserstein_distance

# Example: Calculating Wasserstein Distance for the float_feature
wasserstein_dist_float = wasserstein_distance(data['float_feature'], new_data['float_feature'])
print("Wasserstein Distance for float_feature:", wasserstein_dist_float)


from scipy.stats import gaussian_kde
import numpy as np

def calculate_kld(p_data, q_data):
    # Estimate the density of both distributions
    p_density = gaussian_kde(p_data)
    q_density = gaussian_kde(q_data)

    # Define the range over which to evaluate the densities
    x = np.linspace(min(min(p_data), min(q_data)), max(max(p_data), max(q_data)), num=1000)
    p_x = p_density(x)
    q_x = q_density(x)

    # Ensure that q_x is nonzero everywhere p_x is nonzero
    q_x = np.where(p_x != 0, q_x, np.min(p_x[p_x > 0]))

    # Calculate the KLD
    kld = np.sum(p_x * np.log(p_x / q_x)) * (x[1] - x[0])
    return kld

# Example: Calculating KLD for float_feature
kld_float_feature = calculate_kld(data['float_feature'], new_data['float_feature'])
print("KLD for float_feature:", kld_float_feature)
