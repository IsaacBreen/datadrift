import pandas as pd
import numpy as np
from pandas import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

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

# Combining the encoded features with the original data
data_combined = pd.concat([data, tfidf_df], axis=1)

# Defining the feature columns and the target column
features = ['float_feature', 'bool_feature', 'category_feature_encoded'] + list(tfidf_df.columns)
target = 'is_systemic_risk'

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
