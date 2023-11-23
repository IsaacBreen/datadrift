import dash
import numpy as np
import pandas as pd
from dash import html, dash_table, dcc
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split

from main import DataGenerator, FeatureEngineering, ModelTraining, DriftDetection, drift_detection_report, colorize_p_values, humanize_column_names, metadata_data
app = dash.Dash(__name__)
data_gen = DataGenerator(
    n_rows=100,
    categories=["A", "B", "C"],
    category_type=pd.CategoricalDtype(categories=["A", "B", "C"], ordered=True),
    p=0.3,
    loc1=10,
    loc2=20,
    scale1=10,
    scale2=10
    )
data = data_gen.generate_data()
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
data = feat_eng.fit_transform(data)
model_training = ModelTraining()
features = [col for col in data.columns if
            data[col].dtype in [np.float64, np.float32, np.int64, np.int32, np.uint8, np.uint16, np.uint32, np.uint64, np.bool_] and col != 'is_systemic_risk']
target = 'is_systemic_risk'

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=0, stratify=data[target])
model_training.train_model(X_train, y_train)
y_pred = model_training.predict(X_test)
model_accuracy = model_training.evaluate(y_test, y_pred)
print("Model Accuracy:", model_accuracy)
data_gen = DataGenerator(
    n_rows=100,
    categories=["A", "B", "C"],
    category_type=pd.CategoricalDtype(categories=["A", "B", "C"], ordered=True),
    p=0.4,
    loc1=15,
    loc2=20,
    scale1=10,
    scale2=10
    )
new_data = data_gen.generate_data()
new_data = feat_eng.transform(new_data)

drift_det = DriftDetection()
drift_report = drift_detection_report(
    data,
    new_data,
    drift_det,
    feature_types={'numerical': numerical_features, 'categorical': categorical_features, 'boolean': boolean_features, 'textual': text_features}
    )

humanize_column_names(drift_report)
colorize_p_values(drift_report)
metadata_div = html.Div([
    html.H3("Metadata"),
    html.Ul([html.Li(f"{key}: {value}") for key, value in metadata_data.items()])
])

app.layout = html.Div([
    html.H1('Model Analysis Dashboard'),
    html.Div([
        html.Div([
            html.H3('Model Analysis'),
            dcc.Tabs(id='tabs', value='tab-model', children=[
                dcc.Tab(label='Model Analysis', value='tab-model'),
                dcc.Tab(label='Drift Detection', value='tab-drift'),
            ]),
            html.Div(id='tabs-content'),
        ], className='six columns', id='model-analysis-content'),
        html.Div([
            html.H3('Drift Detection'),
            dcc.Graph(id='drift_detection_graph'),
            metadata_div,
        ], className='six columns', id='drift-detection-content'),
    ], className='row'),
    
    # Placeholder for Model Performance Graph
    dcc.Graph(id='model_performance_graph', figure={}),

    # Placeholder for Feature Importance Graph
    dcc.Graph(id='feature_importance_graph', figure={}),

    # Placeholder for Drift Report Table
    dash_table.DataTable(
        id='drift_report_table',
        columns=[{"name": "", "id": ""}],  # Empty initial columns
        data=[],  # Empty initial data
    ),
], className='container')

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
            dcc.Graph(id='feature_importance_graph'),
        ])
    elif tab == 'tab-drift':
        return html.Div([
            html.H3('Drift Report'),
            dash_table.DataTable(id='drift_report_table'),
            html.H3('Metadata'),
            html.Pre(id='metadata_text'),
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
def update_model_performance(tab):
    if tab == 'tab-model':
        data = go.Bar(x=['Model Accuracy'], y=[model_accuracy])
        layout = go.Layout(title='Model Performance')
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

if __name__ == '__main__':
    app.run_server(debug=True)
