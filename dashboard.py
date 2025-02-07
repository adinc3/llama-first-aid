import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Sample data for graphs (with a new column for app_version)
df = pd.DataFrame({
    "Date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
    "Value": range(100),
    "Category": ["A", "B", "C", "D", "E"] * 20,
    "app_version": [None, "1.0", "1.1", "2.0", "2.1"] * 20  # New column for app_version
})

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Set a custom app title
app_title = "Dashboard"  # Customize this title

# Set the custom title and favicon via the index_string
app.index_string = '''
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>llama-first-aid</title>
        <!-- <link rel="icon" href="/assets/favicon.ico" type="image/x-icon"> Link to favicon -->
        {%metas%}
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        {%config%}
        {%scripts%}
        {%renderer%}
    </body>
</html>
'''

# Layout for users Page
users_page = html.Div([
    # Navbar with title, logo, and users link
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Users", href="/")),
            dbc.NavItem(dbc.NavLink("Performance", href="/performance")),  # Performance link
        ],
        brand=html.Div([
            app_title  # This will display the custom title
        ]),
        brand_href="/",
        color="primary",
        dark=True,
        className="mb-4",
    ),

    # Filters Row - Start and End Date Filters, app_version filter
    html.Div([
        dbc.Row([
            dbc.Col([
                dcc.DatePickerRange(
                    id='date-picker-range-users',  # Unique ID for users page filter
                    start_date=df['Date'].min(),
                    end_date=df['Date'].max(),
                    display_format='YYYY-MM-DD',
                    style={'width': '100%'}
                )
            ], width=4),
            dbc.Col([  # app_version filter dropdown
                dcc.Dropdown(
                    id='app-version-users',
                    options=[
                        {'label': 'No filter', 'value': ''},  # Empty string for "No filter"
                        {'label': 'Version 1.0', 'value': '1.0'},
                        {'label': 'Version 1.1', 'value': '1.1'},
                        {'label': 'Version 2.0', 'value': '2.0'},
                        {'label': 'Version 2.1', 'value': '2.1'},
                    ],
                    value='',  # Default value: empty string for "No filter"
                    style={'width': '100%'}
                )
            ], width=4),
            dbc.Col([  # Empty column for alignment
            ], width=4),
        ])
    ], style={'padding': '20px'}),

    # First Second Row - 3 graphs without filters
    html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='graph-1-no-filter-users'),
            ], width=4),
            dbc.Col([
                dcc.Graph(id='graph-2-no-filter-users'),
            ], width=4),
            dbc.Col([
                dcc.Graph(id='graph-3-no-filter-users'),
            ], width=4)
        ])
    ], style={'padding': '20px'}),

    # Second Second Row - 3 graphs with filters
    html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='graph-1-type-users',
                    options=[
                        {'label': 'Line', 'value': 'line'},
                        {'label': 'Scatter', 'value': 'scatter'},
                        {'label': 'Bar', 'value': 'bar'},
                    ],
                    value='line',  # Default value
                    style={'width': '100%'}
                ),
                dcc.Graph(id='graph-1-users'),
            ], width=4),
            dbc.Col([
                dcc.Dropdown(
                    id='graph-2-type-users',
                    options=[
                        {'label': 'Line', 'value': 'line'},
                        {'label': 'Scatter', 'value': 'scatter'},
                        {'label': 'Bar', 'value': 'bar'},
                    ],
                    value='bar',  # Default value
                    style={'width': '100%'}
                ),
                dcc.Graph(id='graph-2-users'),
            ], width=4),
            dbc.Col([
                dcc.Dropdown(
                    id='graph-3-type-users',
                    options=[
                        {'label': 'Line', 'value': 'line'},
                        {'label': 'Scatter', 'value': 'scatter'},
                        {'label': 'Bar', 'value': 'bar'},
                    ],
                    value='scatter',  # Default value
                    style={'width': '100%'}
                ),
                dcc.Graph(id='graph-3-users'),
            ], width=4)
        ])
    ], style={'padding': '20px'}),

    # Third Row - 2 graphs and no filter for the second graph
    html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='graph-4-users'),
            ], width=6),
            dbc.Col([
                dcc.Dropdown(
                id='graph-5-filter-users',
                options=[
                    {'label': 'Category X', 'value': 'X'},
                    {'label': 'Category Y', 'value': 'Y'},
                ],
                value='X',  # Default value
                style={'width': '100%'}
                ),
                dcc.Dropdown(
                    id='graph-5-type-users',
                    options=[
                        {'label': 'Line', 'value': 'line'},
                        {'label': 'Scatter', 'value': 'scatter'},
                        {'label': 'Bar', 'value': 'bar'},
                    ],
                    value='scatter',  # Default value
                    style={'width': '100%'}
                ),
                dcc.Graph(id='graph-5-users'),
            ], width=6)
        ])
    ], style={'padding': '20px'}),
])

# Layout for performance Page
performance_page = html.Div([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Users", href="/")),
            dbc.NavItem(dbc.NavLink("Performance", href="/performance")),  # Added Performance link
        ],
        brand=app_title,
        brand_href="/performance",
        color="primary",
        dark=True,
        className="mb-4",
    ),
    
    # Filters Row - Start and End Date Filters, app_version filter
    html.Div([
        dbc.Row([
            dbc.Col([
                dcc.DatePickerRange(
                    id='date-picker-range-performance',
                    start_date=df['Date'].min(),
                    end_date=df['Date'].max(),
                    display_format='YYYY-MM-DD',
                    style={'width': '100%'}
                )
            ], width=4),
            dbc.Col([  # app_version filter dropdown
                dcc.Dropdown(
                    id='app-version-performance',
                    options=[
                        {'label': 'No filter', 'value': ''},  # Empty string for "No filter"
                        {'label': 'Version 1.0', 'value': '1.0'},
                        {'label': 'Version 1.1', 'value': '1.1'},
                        {'label': 'Version 2.0', 'value': '2.0'},
                        {'label': 'Version 2.1', 'value': '2.1'},
                    ],
                    value='',  # Default value: empty string for "No filter"
                    style={'width': '100%'}
                )
            ], width=4),
            dbc.Col([  # Empty column for alignment
            ], width=4),
        ])
    ], style={'padding': '20px'}),

    # Performance Graphs - 4 Columns
    html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='graph-1-performance'),
            ], width=3),
            dbc.Col([
                dcc.Graph(id='graph-2-performance'),
            ], width=3),
            dbc.Col([
                dcc.Graph(id='graph-3-performance'),
            ], width=3),
            dbc.Col([
                dcc.Graph(id='graph-4-performance'),
            ], width=3),
        ])
    ], style={'padding': '20px'})
])

# Set up routing for pages (users and performance)
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Define page content based on URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/performance':
        return performance_page
    else:  # users page or any other invalid URL will show users
        return users_page

@app.callback(
    [
        Output('graph-1-users', 'figure'),
        Output('graph-2-users', 'figure'),
        Output('graph-3-users', 'figure'),
        Output('graph-4-users', 'figure'),
        Output('graph-5-users', 'figure'),
        Output('graph-1-no-filter-users', 'figure'),
        Output('graph-2-no-filter-users', 'figure'),
        Output('graph-3-no-filter-users', 'figure'),
    ],
    [
        Input('date-picker-range-users', 'start_date'),
        Input('date-picker-range-users', 'end_date'),
        Input('graph-1-type-users', 'value'),
        Input('graph-2-type-users', 'value'),
        Input('graph-3-type-users', 'value'),
        # Now, only use 'graph-5-filter-users' since 'graph-3-filter-users' does not exist
        Input('graph-5-filter-users', 'value'),
        Input('app-version-users', 'value'),  # New filter added
    ]
)
def update_graphs(start_date, end_date, graph_1_type, graph_2_type, graph_3_type, filter_5, app_version):
    # Filter the data based on the selected date range and app_version filter
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    if app_version != '':  # Check if app_version is not the empty string
        filtered_df = filtered_df[filtered_df['app_version'] == app_version]  # Apply the app_version filter
    
    # Create the figures (graphs)
    fig1 = px.line(filtered_df, x='Date', y='Value', title='Graph 1')
    fig2 = px.scatter(filtered_df, x='Date', y='Value', title='Graph 2')
    fig3 = px.bar(filtered_df, x='Date', y='Value', title='Graph 3')
    fig4 = px.box(filtered_df, x='Category', y='Value', title='Graph 4')
    fig5 = px.line(filtered_df, x='Date', y='Value', title='Graph 5')

    # Create no-filter versions of the graphs
    fig1_no_filter = px.line(filtered_df, x='Date', y='Value', title='Graph 1 - No Filter')
    fig2_no_filter = px.bar(filtered_df, x='Date', y='Value', title='Graph 2 - No Filter')
    fig3_no_filter = px.scatter(filtered_df, x='Date', y='Value', title='Graph 3 - No Filter')

    return fig1, fig2, fig3, fig4, fig5, fig1_no_filter, fig2_no_filter, fig3_no_filter



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
