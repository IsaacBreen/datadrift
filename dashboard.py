import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import html, dash_table, dcc
from dash.dependencies import Input, Output

from data_generator import DataGenerator
from drift import DriftDetection, drift_detection_report, metadata_data
from feature_engineering import FeatureEngineering
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
    training_gen = DataGenerator(n_rows=100, categories=["A", "B", "C"], p=0.2, loc1=10, loc2=20, scale1=5, scale2=10, start_date='2022-01-01', end_date='2022-06-30')
    testing_gen = DataGenerator(n_rows=1000, categories=["A", "B", "C"], p=0.2, loc1=10, loc2=20, scale1=5, scale2=10, start_date='2022-01-01', end_date='2022-06-30')
    production_gen1 = DataGenerator(n_rows=1000, categories=["A", "B", "C"], p=0.2, loc1=10, loc2=20, scale1=5, scale2=10, start_date='2022-07-01', end_date='2023-12-31')
    production_gen2 = DataGenerator(n_rows=1000, categories=["A", "B", "C"], p=0.5, loc1=10, loc2=20, scale1=5, scale2=10, start_date='2024-01-01', end_date='2024-12-31')

    # Generate datasets
    train_data = training_gen.generate_data()
    test_data = testing_gen.generate_data()
    production_data1 = production_gen1.generate_data()
    production_data2 = production_gen2.generate_data()
    production_data = pd.concat([production_data1, production_data2])

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
    features = [col for col in train_data_processed.columns if train_data_processed[col].dtype in [np.float64, np.float32, np.int64, np.int32, np.uint8, np.uint16, np.uint32, np.uint64, np.bool_] and col != 'is_systemic_risk']
    target = 'is_systemic_risk'
    X_train, y_train = train_data_processed[features], train_data_processed[target]
    model_training.train(X_train, y_train)

    # Get dates for each set
    train_dates = train_data_processed['date']
    test_dates = test_data_processed['date']
    production_dates = production_data_processed['date']

    # Evaluate the model on each set with dates
    train_accuracy = model_training.evaluate_row_by_row(y_train, model_training.predict(X_train), train_dates)
    test_accuracy = model_training.evaluate_row_by_row(test_data_processed[target], model_training.predict(test_data_processed[features]), test_dates)
    production_accuracy = model_training.evaluate_row_by_row(production_data_processed[target], model_training.predict(production_data_processed[features]), production_dates)

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
     Output('drift_report_table', 'style_data_conditional'),
     Output('drift_report_table', 'tooltip_data')],
    [Input('tabs', 'value')]
)
def update_drift_report_table(tab):
    if tab == 'tab-drift':
        drift_data = drift_report.to_dict('records')
        columns = [{"name": i, "id": i} for i in drift_report.columns]
        style = []

        # Define style for P-Value columns
        p_value_columns = ['K-S Test P-Value', 'Chi-Squared P-Value', 'Z Test P-Value']
        for col in p_value_columns:
            style.append({
                'if': {
                    'column_id': col,
                    'filter_query': f'{{{col}}} < 0.05',
                    'column_type': 'numeric'
                },
                'backgroundColor': '#FF4136',
                'color': 'white'
            })

        tooltips = [{"header": col, "value": f"Tooltip for {col}"} for col in drift_report.columns]

        return drift_data, columns, style, tooltips
    return [], [], [], []

@app.callback(
    Output('model_performance_graph', 'figure'),
    [Input('tabs', 'value')]
)
def update_model_performance_graph(tab):
    if tab == 'tab-model':
        # Convert the 'Date' column to datetime if it's not already
        train_accuracy['Date'] = pd.to_datetime(train_accuracy['Date'])
        test_accuracy['Date'] = pd.to_datetime(test_accuracy['Date'])
        production_accuracy['Date'] = pd.to_datetime(production_accuracy['Date'])

        # Resample to get monthly averages and compute mean of 'Accuracy' for each month
        train_monthly = train_accuracy.resample('M', on='Date').mean()
        test_monthly = test_accuracy.resample('M', on='Date').mean()
        production_monthly = production_accuracy.resample('M', on='Date').mean()

        # Now create scatter plots for the monthly data
        train_scatter = go.Scatter(
            x=train_monthly.index,
            y=train_monthly['Accuracy'].astype(float),
            mode='lines+markers',
            name='Train'
        )
        test_scatter = go.Scatter(
            x=test_monthly.index,
            y=test_monthly['Accuracy'].astype(float),
            mode='lines+markers',
            name='Test'
        )
        production_scatter = go.Scatter(
            x=production_monthly.index,
            y=production_monthly['Accuracy'].astype(float),
            mode='lines+markers',
            name='Production'
        )

        layout = go.Layout(
            title='Model Performance Over Time',
            xaxis={
                'title': 'Date',
                'tickformat': '%Y-%m',  # Format the x-axis ticks to show only year and month
            },
            yaxis={'title': 'Average Monthly Accuracy'}
        )

        return {'data': [train_scatter, test_scatter, production_scatter], 'layout': layout}
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
