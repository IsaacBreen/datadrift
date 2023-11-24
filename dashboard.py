import dash
import numpy as np
import pandas as pd
from dash import html, dash_table, dcc
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split

from feature_engineering import FeatureEngineering
from data_generator import DataGenerator
from drift import DriftDetection, drift_detection_report, humanize_column_names, colorize_p_values, metadata_data
from train import Model

app = dash.Dash(__name__, suppress_callback_exceptions=True)

def generate_data():
    text_features = ['verbatim_text']
    categorical_features = ['category_feature']
    boolean_features = ['bool_feature']
    numerical_features = ['float_feature']
    feat_eng = FeatureEngineering(
        text_features=text_features,
        categorical_features=categorical_features,
        boolean_features=boolean_features,
        numerical_features=numerical_features
    )

    # Adjust these parameters to simulate drift
    training_gen = DataGenerator(n_rows=100, categories=["A", "B", "C"], p=0.3, loc1=10, loc2=20, scale1=5, scale2=10)
    testing_gen = DataGenerator(n_rows=100, categories=["A", "B", "C"], p=0.35, loc1=12, loc2=22, scale1=7, scale2=12)
    production_gen = DataGenerator(n_rows=100, categories=["A", "B", "C"], p=0.4, loc1=15, loc2=25, scale1=10, scale2=15)

    # Generate datasets
    train_data = training_gen.generate_data()
    test_data = testing_gen.generate_data()
    production_data = production_gen.generate_data()

    # Reset indices to ensure they are unique
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    production_data = production_data.reset_index(drop=True)

    # Combine and process data
    combined_data = pd.concat([train_data, test_data, production_data]).reset_index(drop=True)
    processed_data = feat_eng.fit_transform(combined_data)

    # Split processed data back into train, test, production sets
    # Calculate the end indices for each dataset
    train_end = len(train_data)
    test_end = train_end + len(test_data)

    train_data_processed = processed_data.iloc[:train_end]
    test_data_processed = processed_data.iloc[train_end:test_end]
    production_data_processed = processed_data.iloc[test_end:]

    # Train model
    model_training = Model()
    features = [col for col in processed_data.columns if col not in ['is_systemic_risk', 'date']]
    target = 'is_systemic_risk'
    X_train, y_train = train_data_processed[features], train_data_processed[target]
    model_training.train(X_train, y_train)

    # Evaluate the model on each set
    train_accuracy = model_training.evaluate(y_train, model_training.predict(X_train))
    test_accuracy = model_training.evaluate(test_data_processed[target], model_training.predict(test_data_processed[features]))
    production_accuracy = model_training.evaluate(production_data_processed[target], model_training.predict(production_data_processed[features]))

    # Drift detection between training and test sets, and between test and production sets
    drift_det = DriftDetection()
    drift_report_train_test = drift_detection_report(train_data_processed, test_data_processed, drift_det, feature_types={'numerical': numerical_features, 'categorical': categorical_features, 'boolean': boolean_features, 'textual': text_features})
    drift_report_test_prod = drift_detection_report(test_data_processed, production_data_processed, drift_det, feature_types={'numerical': numerical_features, 'categorical': categorical_features, 'boolean': boolean_features, 'textual': text_features})

    return train_accuracy, test_accuracy, production_accuracy, model_training, train_data_processed, test_data_processed, production_data_processed, features, drift_report_train_test, drift_report_test_prod, X_train, y_train

train_accuracy, test_accuracy, production_accuracy, model_training, train_data, test_data, production_data, features, drift_report_train_test, drift_report_test_prod, X_train, y_train = generate_data()
drift_report = drift_report_test_prod


# Define the initial layout
app.layout = html.Div([
    html.H1("Drift Detection Dashboard"),
    dcc.Tabs(id='tabs', value='tab-model', children=[
        dcc.Tab(label='Model Analysis', value='tab-model'),
        dcc.Tab(label='Drift Detection', value='tab-drift'),
    ]),
    html.Div(id='tabs-content'),
])

@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-model':
        return html.Div([
            html.H3('Model Performance'),
            dcc.Graph(id='model_performance_graph'),
            html.H3('Feature Importance'),
            dcc.Graph(id='feature_importance_graph')  # Placeholder Graph
        ])
    elif tab == 'tab-drift':
        return html.Div([
            html.H3('Drift Report'),
            dash_table.DataTable(id='drift_report_table'),
            html.H3('Metadata'),
            dash_table.DataTable(id='metadata_table')
        ])

@app.callback(
    [Output('drift_report_table', 'data'),
     Output('drift_report_table', 'columns'),
     Output('drift_report_table', 'style_data_conditional'),     Output('drift_report_table', 'tooltip_data')],    [Input('tabs', 'value')]
)
def update_drift_report_table(tab):
    if tab == 'tab-drift':
        drift_data = drift_report.to_dict('records')
        columns = [{"name": i, "id": i} for i in drift_report.columns]
        style = [{
            'if': {
                'column_id': col,
                'filter_query': f'{{{col}}} < 0.05'
            },
            'backgroundColor': '#FF4136',
            'color': 'white'
        } for col in ['K-S Test P-Value', 'Chi-Squared P-Value', 'Z Test P-Value']]
        tooltips = [{"header": col, "value": f"Tooltip for {col}"} for col in drift_report.columns]

        return drift_data, columns, style, tooltips
    return [], [], [], []

from dash.dependencies import Input, Output
import plotly.graph_objs as go

@app.callback(
    Output('model_performance_graph', 'figure'),
    [Input('tabs', 'value')]
)
def update_model_performance_graph(tab):
    if tab == 'tab-model':
        dates = ['2022-01-01 to 2022-04-30', '2022-05-01 to 2022-07-31', '2022-08-01 to 2022-12-31']
        accuracies = [train_accuracy, test_accuracy, production_accuracy]
        data = go.Scatter(x=dates, y=accuracies, mode='lines+markers')
        layout = go.Layout(title='Model Performance Over Time')
        return {'data': [data], 'layout': layout}
    return {}

@app.callback(
    Output('feature_importance_graph', 'figure'),
    [Input('tabs', 'value')]
)
def update_feature_importance(tab):
    if tab == 'tab-model':
        feature_importances = model_training.model.feature_importances_
        features = X_train.columns
        data = go.Bar(x=features, y=feature_importances)
        layout = go.Layout(title='Feature Importances')
        return {'data': [data], 'layout': layout}
    return {}

@app.callback(
    [Output('model-analysis-content', 'style'),
     Output('drift-detection-content', 'style')],
    [Input('tabs', 'value')]
)
def toggle_tab_content(tab):
    if tab == 'tab-model':
        return {'display': 'block'}, {'display': 'none'}
    elif tab == 'tab-drift':
        return {'display': 'none'}, {'display': 'block'}
    return {'display': 'none'}, {'display': 'none'}

def format_metadata_for_datatable(metadata_dict):
    formatted_data = []
    for key, value in metadata_dict.items():
        formatted_data.append({"Metric": key, "Description": value})
    return formatted_data

@app.callback(
    [Output('metadata_table', 'data'),
     Output('metadata_table', 'columns')],
    [Input('tabs', 'value')]
)
def update_metadata_table(tab):
    if tab == 'tab-drift':
        formatted_metadata_data = format_metadata_for_datatable(metadata_data)
        metadata_columns = [{"name": "Metric", "id": "Metric"}, {"name": "Description", "id": "Description"}]
        return formatted_metadata_data, metadata_columns
    return [], []


if __name__ == '__main__':
    app.run_server(debug=True)
