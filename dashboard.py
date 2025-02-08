import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output


# Load dataset (must be switched to a BigQuery load)
df = pd.read_csv("data/sessions_history/sessions_history.csv")

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


# Get distinct values of 'app_version'
app_versions = [{'label': version, 'value': version} for version in df['app_version'].unique()]
# Add a 'No filter' option as well
app_versions.insert(0, {'label': 'No filter', 'value': 'All'})

# Get distinct values of 'severity'
severities = [{'label': severity, 'value': severity} for severity in df['severity'].unique()]
# Add a 'No filter' option as well
severities.insert(0, {'label': 'No filter', 'value': 'All'})

# Get distinct values of 'medical_classes'
medical_classes = [{'label': medical_class, 'value': medical_class} for medical_class in df['medical_class'].unique()]
# Add a 'No filter' option as well
medical_classes.insert(0, {'label': 'No filter', 'value': 'All'})

# Ensure 'timestamp' column is in datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])


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

    # Filters - Start and End Date Filters, app_version filter
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label('Date:', style={'margin-right': '10px', 'display': 'inline-block'}),
                    dcc.DatePickerRange(
                        id='date-picker-range-users',  # Unique ID for users page filter
                        start_date=(df['timestamp'].max() - pd.DateOffset(months=1)).strftime('%Y-%m-%d'),
                        end_date=df['timestamp'].max().strftime('%Y-%m-%d'),
                        display_format='YYYY-MM-DD',
                        style={'width': '100%'}
                    )
                ]),
            ], width=4),
            dbc.Col([  # app_version filter dropdown
                html.Div([
                    html.Label('App:', style={'margin-right': '10px', 'display': 'inline-block'}),
                    dcc.Dropdown(
                        id='app-version-users',
                        options=app_versions,
                        value='All',
                        style={'width': '100%'}
                    )
                ]),
            ], width=4),
            dbc.Col([  # Empty column for alignment
            ], width=4),
        ])
    ], style={'padding': '20px'}),

    # First Row - 3 graphs without filters
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H6("This month's sessions over last month's sessions", style={'textAlign': 'center'}),
                html.Div(id='graph-1-users', style={'fontSize': 20, 'fontWeight': 'bold', 'textAlign': 'center'})
            ], width=4),
            dbc.Col([
                html.H6("Average sessions per day", style={'textAlign': 'center'}),
                html.Div(id='graph-2-users', style={'fontSize': 20, 'fontWeight': 'bold', 'textAlign': 'center'})
            ], width=4),
            dbc.Col([
                html.H6("Sessions per day (trend)", style={'textAlign': 'center'}),
                dcc.Graph(id='graph-3-users'),
            ], width=4)
        ])
    ], style={'padding': '20px'}),

    # Second Row - 3 graphs with filters
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H6("Most frequent medical classes, by severity level", style={'textAlign': 'center'}),
                html.Div([
                    html.Label('Severity:', style={'margin-right': '10px', 'display': 'inline-block'}),
                    dcc.Dropdown(
                        id='graph-4-users-filter',
                        options=severities,
                        value='All',  # Default value
                        style={'width': '50%', 'display': 'inline-block'}
                    )
                ]),
                dcc.Graph(id='graph-4-users'),
            ], width=4),
            dbc.Col([
                html.H6("Most frequent severity levels, by weekday and medical class", style={'textAlign': 'center'}),
                html.Div([
                    html.Label('Medical Class:', style={'margin-right': '10px', 'display': 'inline-block'}),
                    dcc.Dropdown(
                        id='graph-5-users-filter',
                        options=medical_classes,
                        value='All',  # Default value
                        style={'width': '50%', 'display': 'inline-block'}
                    )
                ]),
                dcc.Graph(id='graph-5-users'),
            ], width=4),
            dbc.Col([
                html.H6("Most frequent severity levels, by time range and medical class", style={'textAlign': 'center'}),
                html.Div([
                    html.Label('Medical Class:', style={'margin-right': '10px', 'display': 'inline-block'}),
                    dcc.Dropdown(
                        id='graph-6-users-filter',
                        options=medical_classes,
                        value='All',  # Default value
                        style={'width': '50%', 'display': 'inline-block'}
                    )
                ]),
                dcc.Graph(id='graph-6-users'),
            ], width=4),
        ])
    ], style={'padding': '20px'}),

    # Third Row - 2 graphs and no filter for the second graph
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H6("Choropleth map of sessions in time", style={'textAlign': 'center'}),
                dbc.Switch(
                    id='graph-7-users-toggle-button',
                    value=False,  # default state (False is off, True is on)
                    label="Timelapse",  # Optional, you can customize the label here
                    style={'width': '50%'}  # Optional: style for the label text
                ),
                dcc.Graph(id='graph-7-users'),
            ], width=6),
            dbc.Col([
                html.H6("Heatmap of worldwide sessions", style={'textAlign': 'center'}),
                dbc.Switch(
                    id='graph-8-users-toggle-button',
                    value=False,  # default state (False is off, True is on)
                    label="Timelapse",  # Optional, you can customize the label here
                    style={'width': '50%'}  # Optional: style for the label text
                ),                  
                html.Label('Severity:', style={'margin-right': '10px', 'display': 'inline-block'}),
                dcc.Dropdown(
                    id='graph-8-users-filter',
                    options=severities,
                    value='All',  # Default value
                    style={'width': '50%'}
                ),
                html.Label('Medical Class:', style={'margin-right': '10px', 'display': 'inline-block'}),
                dcc.Dropdown(
                    id='graph-8-users-filter',
                    options=medical_classes,
                    value='All',  # Default value
                    style={'width': '50%'}
                ),
                dcc.Graph(id='graph-8-users'),
            ], width=6)
        ])
    ], style={'padding': '20px'}),
])

# Layout for performance Page
#performance_page = html.Div([
#    dbc.NavbarSimple(
#        children=[
#            dbc.NavItem(dbc.NavLink("Users", href="/")),
#            dbc.NavItem(dbc.NavLink("Performance", href="/performance")),  # Added Performance link
#        ],
#        brand=app_title,
#        brand_href="/performance",
#        color="primary",
#        dark=True,
#        className="mb-4",
#    ),
    
    # Filters Row - Start and End Date Filters, app_version filter
    #html.Div([
    #    dbc.Row([
    #        dbc.Col([
    #            dcc.DatePickerRange(
    #                id='date-picker-range-users',  # Unique ID for users page filter
    #                start_date=(df['timestamp'].max() - pd.DateOffset(months=1)).strftime('%Y-%m-%d'),
    #                end_date=df['timestamp'].max().strftime('%Y-%m-%d'),
    #                display_format='YYYY-MM-DD',
    #                style={'width': '100%'}
    #            )
    #        ], width=4),
    #        dbc.Col([  # app_version filter dropdown
    #            dcc.Dropdown(
    #                id='app-version-users',
    #                options=app_versions,
    #                value='',  # Default value: empty string for "No filter"
    #                style={'width': '100%'}
    #            )
    #        ], width=4),
    #        dbc.Col([  # Empty column for alignment
    #        ], width=4),
    #    ])
    #], style={'padding': '20px'}),

    # Performance Graphs - 4 Columns
    #html.Div([
    #    dbc.Row([
    #        dbc.Col([
    #            dcc.Graph(id='graph-1-performance'),
    #        ], width=3),
    #        dbc.Col([
    #            dcc.Graph(id='graph-2-performance'),
    #        ], width=3),
    #        dbc.Col([
    #            dcc.Graph(id='graph-3-performance'),
    #        ], width=3),
    #        dbc.Col([
    #            dcc.Graph(id='graph-4-performance'),
    #        ], width=3),
    #    ])
    #], style={'padding': '20px'})
#])

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
        Output('graph-1-users', 'children'),
        Output('graph-1-users', 'style'),
        Output('graph-2-users', 'children'),
        Output('graph-3-users', 'figure'),
        Output('graph-4-users', 'figure'),
        Output('graph-5-users', 'figure'),
        Output('graph-6-users', 'figure'),
        Output('graph-7-users', 'figure'),
        Output('graph-8-users', 'figure'),
    ],
    [
        Input('date-picker-range-users', 'start_date'),
        Input('date-picker-range-users', 'end_date'),
        Input('app-version-users', 'value'), 
        Input('graph-4-users-filter', 'value'),
        Input('graph-5-users-filter', 'value'),
        Input('graph-6-users-filter', 'value'),
        Input('graph-7-users-toggle-button', 'value')
    ]
)
def update_graphs(start_date, end_date, app_version, graph_4_severity_level, graph_5_medical_class, graph_6_medical_class, graph_7_toggle_button):
    # Filter the data based on the selected date range and app_version filter
    start_date = pd.to_datetime(start_date).date()  # Convert to date
    end_date = pd.to_datetime(end_date).date()  # Convert to date

    # Filter the dataframe based on the date range
    filtered_df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
    
    if app_version != 'All':  # Check if app_version is not the empty string
        filtered_df = filtered_df[filtered_df['app_version'] == app_version]  # Apply the app_version filter


    ## FIRST PLOT
    # Calculate session count for current and last month
    current_month_sessions = filtered_df[filtered_df['timestamp'].dt.month == pd.to_datetime(end_date).month]['session_id'].nunique()
    last_month_sessions = filtered_df[filtered_df['timestamp'].dt.month == (pd.to_datetime(end_date).month - 1)]['session_id'].nunique()

    # Calculate the percentage change (if last_month_sessions > 0)
    if last_month_sessions > 0:
        percentage_change = ((current_month_sessions - last_month_sessions) / last_month_sessions) * 100
    else:
        percentage_change = '-'  # If no sessions last month, assume no change

    # Determine color based on percentage change
    if percentage_change != '-':
        if percentage_change > 100:
            color = 'green'
        elif percentage_change > 0 & percentage_change < 100:
            color = 'red'
    else:
        color = 'black'
    
    # Style for the percentage text (color)
    text_style = {
        'fontSize': 24,
        'fontWeight': 'bold',
        'textAlign': 'center',
        'color': color  # Apply dynamic color
    }


    ## SECOND PLOT
    average_sessions_per_day = '-' if len(filtered_df) == 0 else filtered_df['session_id'].nunique() / (((end_date - start_date).days) + 1)


    ## THIRD PLOT
    # Generate a list of all dates between start_date and end_date
    date_range = pd.date_range(start=start_date, end=end_date)

    # Group the sessions by date and count the number of sessions per day
    sessions_per_day = filtered_df.groupby(filtered_df['timestamp'].dt.date).size().reindex(date_range.date, fill_value=0)

    # Create a DataFrame for the sessions per day to plot with plotly.express
    sessions_df = pd.DataFrame({
        'date': sessions_per_day.index,
        'sessions': sessions_per_day.values
    })

    # Create the initial line graph using plotly express
    sessions_line_graph = px.line(
        sessions_df, 
        x='date', 
        y='sessions'
    )

    # Add scatter trace for bullet points where sessions > 0
    non_zero_sessions = sessions_df[sessions_df['sessions'] > 0]

    # Overlay bullet points on the line graph
    sessions_line_graph.add_traces(
        go.Scatter(
            x=non_zero_sessions['date'],
            y=non_zero_sessions['sessions'],
            mode='markers',
            marker=dict(size=6, symbol='circle', color='rgb(50, 50, 50)'),  # Small bullet points
            name=None
        )
    )

    # Customize the layout for transparent background
    sessions_line_graph.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the entire figure
        title_font=dict(size=14, family="Arial, sans-serif", color='rgb(50, 50, 50)', weight='bold'),
        xaxis=dict(
            title='Date',
            title_font=dict(size=12, family="Arial, sans-serif", color='rgb(50, 50, 50)'),
            tickfont=dict(size=10, family="Arial, sans-serif", color='rgb(100, 100, 100)'),
            showgrid=False,  # Hide gridlines
            gridcolor='rgb(200, 200, 200)',  # Light gridlines
            gridwidth=0.5,
            zeroline=True,  # Line at y=0
            zerolinecolor='rgb(200, 200, 200)',
            zerolinewidth=1
        ),
        yaxis=dict(
            title='Number of Sessions',
            title_font=dict(size=12, family="Arial, sans-serif", color='rgb(50, 50, 50)'),
            tickfont=dict(size=10, family="Arial, sans-serif", color='rgb(100, 100, 100)'),
            showgrid=True,
            gridcolor='rgb(200, 200, 200)',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='rgb(200, 200, 200)',
            zerolinewidth=1
        ),
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False
    )


    ## FOURTH PLOT
    # Group the data by 'medical_class' and count the sessions per class
    # Filter data based on severity level
    if graph_4_severity_level != 'All':
        medical_class_df = filtered_df[filtered_df['severity'] == graph_4_severity_level]
    else:
        medical_class_df = filtered_df  # No severity filter applied

    medical_class_sessions = medical_class_df.groupby('medical_class')['session_id'].nunique()

    total_sessions = medical_class_df['session_id'].nunique()

    medical_class_percentage = (medical_class_sessions / total_sessions) * 100

    medical_class_df = pd.DataFrame({
        'medical_class': medical_class_percentage.index,
        'percentage': medical_class_percentage.values
    }).reset_index(drop=True)

    medical_class_df.sort_values(by='percentage', ascending=False).head(5)

    medical_class_bar_chart = px.bar(
        medical_class_df,
        x='medical_class',
        y='percentage',
    )

    medical_class_bar_chart.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the entire figure
        title_font=dict(size=24, family="Arial, sans-serif", color='rgb(50, 50, 50)', weight='bold'),
        xaxis=dict(
            title='Medical Class',
            title_font=dict(size=12, family="Arial, sans-serif", color='rgb(50, 50, 50)'),
            tickfont=dict(size=10, family="Arial, sans-serif", color='rgb(100, 100, 100)'),
            gridcolor='rgb(200, 200, 200)',
            gridwidth=0.5,
        ),
        yaxis=dict(
            title='% Sessions',
            title_font=dict(size=12, family="Arial, sans-serif", color='rgb(50, 50, 50)'),
            tickfont=dict(size=10, family="Arial, sans-serif", color='rgb(100, 100, 100)'),
            showgrid=True,
            gridcolor='rgb(200, 200, 200)',
            gridwidth=0.5,
        ),
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False
    )


    ## FIFTH PLOT
    if graph_5_medical_class != 'All':
        composition_df = filtered_df[filtered_df['medical_class'] == graph_5_medical_class]
    else:
        composition_df = filtered_df  # No severity filter applied

    composition_df_not_na = composition_df[composition_df['severity'].notna()]  # Remove rows where severity is None

    # --- Calculate the percentage of sessions for each severity level per weekday ---
    composition_df_not_na['weekday'] = composition_df_not_na['timestamp'].dt.day_name()  # Get weekday names
    severity_weekday_counts = composition_df_not_na.groupby(['weekday', 'severity']).size().reset_index(name='session_count')

    # Calculate the total sessions per weekday
    total_sessions_per_weekday = composition_df_not_na.groupby('weekday').size().reset_index(name='total_sessions')

    # Merge the session counts with the total sessions per weekday
    severity_weekday_counts = severity_weekday_counts.merge(total_sessions_per_weekday, on='weekday')

    # Calculate the percentage of sessions for each severity level
    severity_weekday_counts['percentage'] = (severity_weekday_counts['session_count'] / severity_weekday_counts['total_sessions']) * 100

    # Sort the weekdays to maintain the order (Monday, Tuesday, etc.)
    weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    severity_weekday_counts['weekday'] = pd.Categorical(severity_weekday_counts['weekday'], categories=weekdays_order, ordered=True)
    severity_weekday_counts = severity_weekday_counts.sort_values('weekday')

    # --- Ensure that all weekdays are included, even those with no data ---
    all_weekdays = pd.DataFrame({'weekday': weekdays_order})
    all_severity_levels = composition_df_not_na['severity'].unique()

    # Create all combinations of weekdays and severity levels
    all_combinations = pd.MultiIndex.from_product([all_weekdays['weekday'], all_severity_levels], names=['weekday', 'severity']).to_frame(index=False)

    # Merge with the original data to include weekdays with no sessions
    severity_weekday_counts_full = all_combinations.merge(severity_weekday_counts, on=['weekday', 'severity'], how='left').fillna(0)

    # Calculate the percentage for the missing data (if any)
    total_sessions_per_weekday_full = severity_weekday_counts_full.groupby('weekday')['session_count'].transform('sum')
    severity_weekday_counts_full['percentage'] = (severity_weekday_counts_full['session_count'] / total_sessions_per_weekday_full) * 100

    # --- Define the custom color scale ---
    color_scale = {
        '1': 'white',
        '2': 'green',
        '3': 'orange',
        '4': 'darkorange',
        '5': 'red'
    }

    # Ensure the 'severity' column is treated as a string for correct mapping
    severity_weekday_counts_full['severity'] = severity_weekday_counts_full['severity'].astype(str)

    # --- Create the stacked bar chart ---
    composition_graph = px.bar(
        severity_weekday_counts_full,
        x='weekday',
        y='percentage',
        color='severity',
        color_discrete_map=color_scale,  # Custom color scale
        category_orders={'weekday': weekdays_order}  # Order weekdays properly
    )

    composition_graph.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the entire figure
        title_font=dict(size=24, family="Arial, sans-serif", color='rgb(50, 50, 50)', weight='bold'),
        xaxis=dict(
            title='Weekday',
            title_font=dict(size=12, family="Arial, sans-serif", color='rgb(50, 50, 50)'),
            tickfont=dict(size=10, family="Arial, sans-serif", color='rgb(100, 100, 100)'),
            gridcolor='rgb(200, 200, 200)',
            gridwidth=0.5,
        ),
        yaxis=dict(
            title='% Sessions',
            title_font=dict(size=12, family="Arial, sans-serif", color='rgb(50, 50, 50)'),
            tickfont=dict(size=10, family="Arial, sans-serif", color='rgb(100, 100, 100)'),
            showgrid=True,
            gridcolor='rgb(200, 200, 200)',
            gridwidth=0.5,
        ),
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=True
    )



    ## SIXTH PLOT
    if graph_6_medical_class != 'All':
        composition_df_2 = filtered_df[filtered_df['medical_class'] == graph_6_medical_class]
    else:
        composition_df_2 = filtered_df  # No medical class filter applied

    composition_df_2_not_na = composition_df_2[composition_df_2['severity'].notna()]  # Remove rows where severity is None

    # --- Create a new column for hour ranges ---
    def get_hour_range(hour):
        if 0 <= hour <= 4:
            return '0-4'
        elif 5 <= hour <= 9:
            return '5-9'
        elif 10 <= hour <= 14:
            return '10-14'
        elif 15 <= hour <= 19:
            return '15-19'
        elif 20 <= hour <= 23:
            return '20-23'
        return 'Unknown'

    # Apply the function to the 'timestamp' column to create the 'hour_range' column
    composition_df_2_not_na['hour_range'] = composition_df_2_not_na['timestamp'].dt.hour.apply(get_hour_range)

    # --- Calculate the percentage of sessions for each severity level per hour range ---
    severity_hour_range_counts = composition_df_2_not_na.groupby(['hour_range', 'severity']).size().reset_index(name='session_count')

    # Calculate the total sessions per hour range
    total_sessions_per_hour_range = composition_df_2_not_na.groupby('hour_range').size().reset_index(name='total_sessions')

    # Merge the session counts with the total sessions per hour range
    severity_hour_range_counts = severity_hour_range_counts.merge(total_sessions_per_hour_range, on='hour_range')

    # Calculate the percentage of sessions for each severity level
    severity_hour_range_counts['percentage'] = (severity_hour_range_counts['session_count'] / severity_hour_range_counts['total_sessions']) * 100

    # Sort the hour ranges to maintain the order (0-4, 5-9, etc.)
    hour_range_order = ['0-4', '5-9', '10-14', '15-19', '20-23']
    severity_hour_range_counts['hour_range'] = pd.Categorical(severity_hour_range_counts['hour_range'], categories=hour_range_order, ordered=True)

    # Ensure all hour ranges are present, even those with 0% sessions
    # Create a DataFrame with all possible hour ranges and severity levels
    all_hour_ranges = pd.DataFrame({'hour_range': hour_range_order})
    all_severity_levels = composition_df_2_not_na['severity'].unique()

    # Create a DataFrame for all combinations of hour_range and severity (even if no sessions)
    all_combinations = pd.MultiIndex.from_product([all_hour_ranges['hour_range'], all_severity_levels], names=['hour_range', 'severity']).to_frame(index=False)

    # Merge with the original data to ensure all combinations are represented
    severity_hour_range_counts_full = all_combinations.merge(severity_hour_range_counts, on=['hour_range', 'severity'], how='left').fillna(0)

    # Calculate percentage for missing data (if any)
    total_sessions_per_hour_range_full = severity_hour_range_counts_full.groupby('hour_range')['session_count'].transform('sum')
    severity_hour_range_counts_full['percentage'] = (severity_hour_range_counts_full['session_count'] / total_sessions_per_hour_range_full) * 100

    # Ensure the 'severity' column is treated as a string for correct mapping
    severity_hour_range_counts_full['severity'] = severity_hour_range_counts_full['severity'].astype(str)

    # --- Force the severity to follow a consistent order across all hour ranges ---
    severity_order = ['5', '4', '3', '2', '1']  # Severity 5 at the top, severity 1 at the bottom
    severity_hour_range_counts_full['severity'] = pd.Categorical(severity_hour_range_counts_full['severity'], categories=severity_order, ordered=True)

    # --- Create the stacked bar chart ---
    composition_graph_2 = px.bar(
        severity_hour_range_counts_full,
        x='hour_range',
        y='percentage',
        color='severity',
        color_discrete_map=color_scale,  # Apply the custom color scale
        category_orders={'hour_range': hour_range_order}  # Order hour ranges properly
    )

    composition_graph_2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the entire figure
        title_font=dict(size=24, family="Arial, sans-serif", color='rgb(50, 50, 50)', weight='bold'),
        xaxis=dict(
            title='Time Range',
            title_font=dict(size=12, family="Arial, sans-serif", color='rgb(50, 50, 50)'),
            tickfont=dict(size=10, family="Arial, sans-serif", color='rgb(100, 100, 100)'),
            gridcolor='rgb(200, 200, 200)',
            gridwidth=0.5,
        ),
        yaxis=dict(
            title='% Sessions',
            title_font=dict(size=12, family="Arial, sans-serif", color='rgb(50, 50, 50)'),
            tickfont=dict(size=10, family="Arial, sans-serif", color='rgb(100, 100, 100)'),
            showgrid=True,
            gridcolor='rgb(200, 200, 200)',
            gridwidth=0.5,
        ),
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=True
    )


    # Sixth Plot
    # coordinates_df = filtered_df[['country', 'sessions']]  # Adjust as necessary
# 
    # # Group by country and sum the sessions
    # country_sessions = coordinates_df.groupby('country')['sessions'].sum().reset_index()
# 
    # # Create a choropleth map
    # choropleth_map = px.choropleth(country_sessions,
    #                                 locations='country',
    #                                 locationmode='country names',
    #                                 color='sessions',
    #                                 hover_name='country',
    #                                 color_continuous_scale="Viridis",
    #                                 labels={'sessions': 'Total Sessions'},
    #                                 title="Total Sessions by Country")

    # --- Return the figures for all graphs ---
    return (
        ('No sessions in last month' if percentage_change == '-' else [f"{percentage_change:.2f}%"]), 
        text_style, 
        ('Not enough days' if average_sessions_per_day == '-' else [f"{average_sessions_per_day:.2f}"]),
        sessions_line_graph,  # Line graph for sessions per day
        medical_class_bar_chart,  # Bar chart for medical class percentages
        composition_graph,  # Bar chart for weekday and severity composition
        composition_graph_2,  # Bar chart for hour range and severity composition
        composition_graph,  # Choropleth map for sessions by country
        composition_graph
    )


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
