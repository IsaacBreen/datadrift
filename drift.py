from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance, gaussian_kde, norm
from sklearn.model_selection import train_test_split
from termcolor import colored

from data import FeatureEngineering
from data_generator import DataGenerator
from train import Model

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

    @staticmethod
    def two_proportions_z_test(count1, nobs1, count2, nobs2):
        proportion1 = count1 / nobs1
        proportion2 = count2 / nobs2
        proportion = (count1 + count2) / (nobs1 + nobs2)
        z_statistic = (proportion1 - proportion2) / np.sqrt(proportion * (1 - proportion) * (1 / nobs1 + 1 / nobs2))
        p_value = norm.sf(abs(z_statistic)) * 2  # Two-tailed test
        return z_statistic, p_value
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

    # Boolean Features Drift Detection
    for feature in feature_types.get('boolean', []):
        chi2_stat, chi2_p, _, _ = drift_detector.chi_squared_test(original_data[feature], new_data[feature])

        count1 = original_data[feature].sum()
        nobs1 = len(original_data[feature])
        count2 = new_data[feature].sum()
        nobs2 = len(new_data[feature])
        z_stat, p_value = drift_detector.two_proportions_z_test(count1, nobs1, count2, nobs2)

        report.append({
            'Feature': feature,
            'Type': 'Boolean',
            'Chi-Squared Statistic': chi2_stat,
            'Chi-Squared P-Value': chi2_p,
            'Z-Statistic': z_stat,
            'Z P-Value': p_value
        })

    # TODO: Textual Features Drift Detection - Can be complex and may require custom implementation
    # for feature in feature_types['textual']:
    #     Implement custom drift detection for textual features

    return pd.DataFrame(report)
def humanize_column_names(drift_report):
    # Renaming columns for clarity
    column_renames = {
        'KS Statistic':          'Kolmogorov-Smirnov Test',
        'KS P-Value':            'K-S Test P-Value',
        'Chi-Squared Statistic': 'Chi-Squared Test',
        'Chi-Squared P-Value':   'Chi-Squared P-Value',
        'Wasserstein Distance':  'Wasserstein Distance',
        'KLD':                   'Kullback-Leibler Divergence',
        'PSI':                   'Population Stability Index',
        'Z-Statistic':           'Z Test Statistic',
        'Z P-Value':             'Z Test P-Value'
    }
    drift_report.rename(columns=column_renames, inplace=True)
def colorize_p_values(drift_report):
    def _colorize_p_value(value):
        if value < 0.01:
            return colored(f'{value:.3f}', 'red')
        elif value < 0.05:
            return colored(f'{value:.3f}', 'yellow')
        else:
            return colored(f'{value:.3f}', 'green')

    # Color coding P-Values
    p_value_columns = [col for col in drift_report.columns if 'P-Value' in col]
    for col in p_value_columns:
        drift_report[col] = drift_report[col].apply(_colorize_p_value)
def generate_conclusions(drift_report):
    def drift_detected(p_value):
        return 'Yes' if p_value < 0.05 else 'No'

    conclusions = pd.DataFrame()

    p_value_columns = [col for col in drift_report.columns if 'P-Value' in col]
    for col in p_value_columns:
        conclusions[col] = drift_report[col].apply(lambda x: drift_detected(float(x)))

    return conclusions
metadata = """
Kolmogorov-Smirnov Test: Measures the maximum distance between two distributions.
K-S Test P-Value: Probability of observing the data if the null hypothesis of no drift is true. Lower values suggest drift.
Chi-Squared Test: Measures the difference between expected and observed frequencies in categorical data.
Chi-Squared P-Value: Probability of observing the data under no drift in categorical features.
Wasserstein Distance: Measures the distance between two probability distributions.
Kullback-Leibler Divergence: Measures how one probability distribution diverges from a second, expected distribution.
Population Stability Index: Measures the stability of a feature's distribution over time.
Z Test Statistic & P-Value: Used for hypothesis testing in boolean features.
"""
metadata_data = {line.split(":")[0].strip(): line.split(":")[1].strip()
                 for line in metadata.split('\n') if line}
def generate_example_drift_report():
    # Data Generation (assuming this is specific and remains as provided)
    data_gen = DataGenerator(n_rows=100, categories=["A", "B", "C"], p=0.3, loc1=10, loc2=20, scale1=10, scale2=10)
    data = data_gen.generate_data()

    # Define feature types
    text_features = ['verbatim_text']
    categorical_features = ['category_feature']
    boolean_features = ['bool_feature']
    numerical_features = ['float_feature']

    # Feature Engineering
    feat_eng = FeatureEngineering(
        text_features=text_features,
        categorical_features=categorical_features,
        boolean_features=boolean_features,
        numerical_features=numerical_features
        )
    data = feat_eng.fit_transform(data)

    # Model Training
    model_training = Model()

    # Define features based on transformed data
    features = [col for col in data.columns if data[col].dtype in [np.float64, np.float32, np.int64, np.int32, np.uint8, np.uint16, np.uint32, np.uint64, np.bool_] and col != 'is_systemic_risk']
    target = 'is_systemic_risk'

    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=0, stratify=data[target])
    model_training.train(X_train, y_train)
    y_pred = model_training.predict(X_test)
    model_accuracy = model_training.evaluate(y_test, y_pred)
    print("Model Accuracy:", model_accuracy)

    # New Data for Drift Detection
    data_gen = DataGenerator(n_rows=100, categories=["A", "B", "C"], p=0.4, loc1=15, loc2=20, scale1=10, scale2=10)
    new_data = data_gen.generate_data()
    new_data = feat_eng.transform(new_data)

    drift_det = DriftDetection()
    drift_report = drift_detection_report(data, new_data, drift_det, feature_types={'numerical': numerical_features, 'categorical': categorical_features, 'boolean': boolean_features, 'textual': text_features})

    humanize_column_names(drift_report)
    colorize_p_values(drift_report)

    return drift_report


if __name__ == "__main__":
    drift_report = generate_example_drift_report()
    conclusions = generate_conclusions(drift_report)

    # Printing the Table
    print("\nDrift Detection Report Table:")
    print(drift_report.to_string(index=False))
    print("\nDrift detected?")
    print(conclusions.to_string(index=False))
    print("\nMetadata:")
    print(metadata)
