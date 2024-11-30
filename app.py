import os
import pathlib
import dash
import re
import numpy as np
import pandas as pd
import json
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ClientsideFunction, callback_context
from dash.exceptions import PreventUpdate
from user_layout import left_column, main_page, user_keys, score_list, INDUSTRY
from score_similarity import Similarity


# Get relative data folder
PATH = pathlib.Path(os.getcwd())
DATA_PATH = PATH.joinpath("data").resolve()

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Job Recommender System"
server = app.server

# Load data
df = pd.read_csv("data_sample.csv")
# results = pd.read_csv("map-vizz/results.csv")  # Data for maps

# Load geojson data for maps
with open('cities.geojson', encoding = 'utf-8') as f:
    city_geojson = json.load(f)

# with open('world_countries.json', encoding = 'utf-8') as f:
#     world_geojson = json.load(f)

cities = [
    {
        "label": f"{feature['properties']['NAME']}, {feature['properties']['ADM1NAME']}, {feature['properties']['ADM0NAME']}",
        # longitude, latitude
        "value": f"{feature['geometry']['coordinates'][0]},{feature['geometry']['coordinates'][1]}"
    }
    for feature in city_geojson["features"]
]

# Instantiate Similarity class
similarity = Similarity(df)

# App layout
app.layout = html.Div(
    [
        dcc.Store(id='search_results', data=[]),
        html.Div([
            html.Div([
                # sidebar_header
                html.Button(
                    "Show/hide", id="toggle-sidebar-button", n_clicks=0),
                dbc.Collapse(
                    dbc.Nav([left_column], id="left-column"),
                    id="sidebar-collapse",
                    is_open=True
                ),
            ], id='sidebar', className="sidebar"),
            html.H1("Top 5 Job Similarity Scores", style={"textAlign": "center", 'display': 'block'}),
            main_page,
        ], id="main-page", className="main-page"),
        # Add choropleth map at the bottom of the stacked bar charts
        html.Div([
            html.H2("Global Counts Choropleth Map", style={"textAlign": "center", 'marginTop': '20px'}),
            dcc.Graph(id='choropleth-map')
        ], id="heatmap-container", style={"display": "none"}),
    ]
)
# ---------------------------------------------------
# -------------- Helper functions -------------------
# ---------------------------------------------------


def top5_indices(results: pd.DataFrame):
    top5 = list(results.index[:5])
    print(f"Top 5 indices: {top5}")
    return top5


# Function to generate a stacked bar chart for a given job
def generate_stacked_bar(job_index, rdf: pd.DataFrame):
    # Get the list of columns in the DataFrame that match those in score_columns
    score_columns = [col for col in rdf.columns if col in score_list]

    job_data = rdf.loc[[job_index], score_columns].melt(var_name="Score Type", value_name="Score Value")
    # Retrieve Job Title and Company to use in the chart title
    job_title = rdf.loc[job_index, 'Job Title']
    company = rdf.loc[job_index, 'Company']
    similarity_score = rdf.loc[job_index, 'final_similarity_score']
    title = f"<b>{job_title}</b>  - {company} - Similarity Score: {similarity_score:.4f}"
    job_data["Category"] = f"Job ID: {rdf.loc[job_index, 'Job Id']}"
    job_data["Category"] = ""
    fig = px.bar(job_data, x="Score Value", y="Category", color="Score Type", orientation='h', title=title)
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20), yaxis_title="", xaxis_title="Score Value", showlegend=False)
    return fig


# ---------------------------------------------------
# ------------------ CALLBACKS ----------------------
# ---------------------------------------------------

# Callback to toggle sidebar
@app.callback(
    Output("sidebar-collapse", "is_open"),
    Output("sidebar", "style"),
    [Input("toggle-sidebar-button", "n_clicks")],
    [State("sidebar-collapse", "is_open")]
)
def toggle_sidebar(n_clicks, is_open):
    if n_clicks > 0:
        return not is_open, {'height': 'none'}
    return is_open, {}


# Suggestions Callback: Updates dropdown options based on user input
@app.callback(
    Output("skills_selector", "options"),
    Input("skills_selector", "search_value"),
    Input("skills_selector", "value")
)
def update_options(search_value, current_values):
    if not search_value:
        return [{"label": skill, "value": skill} for skill in (current_values or [])]

    options = [{"label": skill, "value": skill}
               for skill in (current_values or [])]

    if search_value not in (current_values or []):
        options.append({"label": search_value, "value": search_value})

    return options


# Location callback: Updates location options based on user input
@app.callback(
    Output("location_selector", "options"),
    [Input("location_selector", "search_value")],
)
def location_auto(search_value):
    # Prevent update if the search value is too short
    if not search_value or len(search_value) < 4:
        raise PreventUpdate

    # Filter cities based on the search term (case-insensitive)
    filtered_cities = [
        city for city in cities if search_value.lower() in city["label"].lower()]

    # Return the filtered results
    return filtered_cities

# Sector callback: Updates sector options based on user input


@app.callback(
    [
        Output("sector1_selector", "options"),
        Output("sector2_selector", "options"),
        Output("sector3_selector", "options"),
    ],
    [
        Input("sector1_selector", "value"),
        Input("sector2_selector", "value"),
        Input("sector3_selector", "value"),
    ]
)
def update_dropdown_options(sector1, sector2, sector3):
    # Get the list of all selected values
    selected = [sector1, sector2, sector3]
    selected = [s for s in selected if s]  # Filter out None values

    # Define a function to filter options based on current selections
    def filter_options(current_value):
        options = [{"label": industry, "value": industry}
                   for industry in INDUSTRY if industry not in selected or industry == current_value]
        return options

    return filter_options(sector1), filter_options(sector2), filter_options(sector3)


# collect user input on search button click
@app.callback(
    Output("search_results", "data"),
    Output("main-container", "style"),
    Output("heatmap-container", "style"),
    Input("search_button", "n_clicks"),
    [
        State("job_title_container", "children"),
        State("skills_container", "children"),
        State("experience_container", "children"),
        State("qualification_container", "children"),
        State("salary_container", "children"),
        State("location_container", "children"),
        State("work_type_container", "children"),
        State("company_size_container", "children"),
        State("sector_container", "children")
    ],
)
def collect_user_input(n_clicks, *args):
    if n_clicks == 0:
        raise PreventUpdate
    else:
        user_data = {}
        # Define regex patterns
        input_pattern = r"(\w+)_selector"
        weight_pattern = r"(\w+)_weight"

        # Process each container to extract inputs and weights
        for i, arg in enumerate(args):
            # Extract input data (e.g., 'job_title', 'location')
            inputs = [child for child in arg if 'id' in child['props']
                      and re.search(input_pattern, child['props']['id'])]

            # Extract weight data (e.g., 'job_title_weight', 'location_weight')
            weights = [child for child in arg if 'id' in child['props']
                       and re.search(weight_pattern, child['props']['id'])]

            # Process inputs
            for input_item in inputs:
                match = re.search(input_pattern, input_item['props']['id'])
                if match:
                    key = match.group(1)  # e.g., 'job_title', 'location'
                    value = input_item['props'].get('value', None)
                    if key == 'salary':
                        user_data['salary_min'] = value
                    elif key == 'skills':
                        if isinstance(value, list):
                            user_data[key] = ", ".join(value)
                        else:
                            user_data[key] = value
                    elif key == 'location':
                        if value is None:
                            user_data['longitude'], user_data['latitude'] = None, None
                        # Assume 'longitude, latitude' for location
                        else:
                            lon, lat = value.split(",")
                            user_data['longitude'] = float(lon)
                            user_data['latitude'] = float(lat)
                    elif key in user_keys:
                        user_data[key] = value

            # Process weights
            for weight_item in weights:
                match = re.search(weight_pattern, weight_item['props']['id'])
                if match:
                    key = match.group(1)  # e.g., 'job_title', 'location'
                    # Assuming 'value' contains the weight
                    weight = float(weight_item['props'].get('value', 0))

                    if f"{key}_weight" in user_keys:
                        user_data[f"{key}_weight"] = weight

        # Run the scoring algorithm and return top 100 results
        results = similarity.calculate_similarity_scores(user_data)
        # print(f"Results: {results.head(5)}")

        return results.to_dict('records'), {"display": "flex"}, {"display": "block"}

# ----------------------------------------------------------------------------


def generate_charts(indices, search_results):
    charts = []
    for i, job_index in enumerate(indices):
        if job_index is not None:
            fig = generate_stacked_bar(job_index, search_results)
            charts.append(
                html.Div([
                    dcc.Graph(id=f"graph-{i}", figure=fig),
                    html.Div([
                        html.Button("Remove", id=f"remove-{i}", n_clicks=0),
                        html.Button("Details", id=f"details-{i}", n_clicks=0)
                    ], style={"display": "flex", "justify-content": "space-between"})
                ], style={"padding": "10px", "border": "1px solid #ddd", "margin-bottom": "10px", 'border-radius': '5px'})
            )
        else:
            charts.append(html.Div("No job available",style={"color": "gray"}))
    return charts


def generate_details(selected_data):
    if selected_data.empty:
        return "No details found for the selected job."

    # Dynamically generate details for a job
    details = [
        html.H5(f"Details for Job ID: {selected_data['Job Id'].iloc[0]}"),
        html.P(f"Job Title: {selected_data['Job Title'].iloc[0]}"),
        html.P(f"Role: {selected_data['Role'].iloc[0]}"),
        html.P(f"Work Type: {selected_data['Work Type'].iloc[0]}"),
        html.P(f"Job Description: {selected_data['Job Description'].iloc[0]}"),
        html.P(f"Skills: {selected_data['skills'].iloc[0]}"),
        html.P(f"Responsibilities: {selected_data['Responsibilities'].iloc[0]}"),
        html.P(f"Company: {selected_data['Company'].iloc[0]}"),
        html.P(f"Sector: {selected_data['Sector'].iloc[0]}"),
        html.P(f"Industry: {selected_data['Industry'].iloc[0]}"),
        html.P(f"Benefits: {selected_data['Benefits'].iloc[0]}"),
        html.P(f"Min Salary: ${selected_data['Min Salary in $'].iloc[0]}"),
        html.P(f"Max Salary: ${selected_data['Max Salary in $'].iloc[0]}"),
        html.P(f"Location: {selected_data['location'].iloc[0]}"),
        html.P(f"Country: {selected_data['Country'].iloc[0]}")
    ]
    return details


def handle_remove_action(i, next_job_index, top_5_indices, search_results):
    if next_job_index < len(search_results):
        top_5_indices.pop(i)
        top_5_indices.insert(i, search_results.index[next_job_index])
        next_job_index += 1
    else:
        top_5_indices[i] = None
    return top_5_indices, next_job_index


def handle_details_action(i, top_5_indices, search_results):
    job_index = top_5_indices[i]
    job_id = search_results.loc[job_index, 'Job Id']
    selected_data = search_results[search_results['Job Id'] == job_id]
    return generate_details(selected_data)

##################################################################################################################################


# Combined callback to handle both "Remove" and "Details" actions
# Global state for top 5 indices and next job index
app_state = {
    "top_5_indices": [],
    "next_job_index": 0
}


@app.callback(
    [Output(f"chart-{i}", "children") for i in range(5)] +
    [Output('info-display', 'children')],
    [Input("search_results", "data")] +  # Correctly append as part of the list
    [Input(f"remove-{i}", "n_clicks") for i in range(5)] +
    [Input(f"details-{i}", "n_clicks") for i in range(5)]
)
def combined_callback(search_results, *args):
    if not search_results:
        # Do not update outputs if search_results is missing or empty
        return [dash.no_update] * 5 + [dash.no_update]
    if search_results is None:
        raise PreventUpdate

    global app_state

    search_results = pd.DataFrame(search_results)

    # Access global state
    top_5_indices = app_state["top_5_indices"]
    next_job_index = app_state["next_job_index"]

    # If the state is not initialized or search_results is updated, set it up
    if not top_5_indices or dash.callback_context.triggered[0]["prop_id"] == "search_results.data":
        top_5_indices = list(search_results.index[:5])
        app_state["top_5_indices"] = top_5_indices
        app_state["next_job_index"] = len(top_5_indices)

        # Generate updated charts for the initial display
        chart_outputs = generate_charts(top_5_indices, search_results)
        info_display = "Click on a job for more details."
        return chart_outputs + [info_display]

    # Separate inputs
    remove_buttons = args[:5]
    details_buttons = args[5:10]

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Initialize outputs
    chart_outputs = [dash.no_update] * 5
    info_display = "Click on a job for more details."

    # Handle Remove Button Click
    for i, clicks in enumerate(remove_buttons):
        if clicks > 0 and triggered_id == f"remove-{i}":
            if next_job_index < len(search_results):
                # Replace the removed job with the next one
                top_5_indices.pop(i)
                top_5_indices.append(search_results.index[next_job_index])
                next_job_index += 1
            else:
                # No more jobs available
                top_5_indices[i] = None

            # Update the global state
            app_state["top_5_indices"] = top_5_indices
            app_state["next_job_index"] = next_job_index

            # Generate updated charts
            chart_outputs = generate_charts(top_5_indices, search_results)

            return chart_outputs + [info_display]

    # Handle Details Button Click
    for i, clicks in enumerate(details_buttons):
        if clicks > 0 and triggered_id == f"details-{i}":
            if top_5_indices[i] is None:
                info_display = "No details available."
            else:
                # job_index = top_5_indices[i]
                info_display = handle_details_action(i, top_5_indices, search_results)

            # Keep charts unchanged
            chart_outputs = generate_charts(top_5_indices, search_results)

            return chart_outputs + [info_display]

    # Default case: No updates
    return chart_outputs + [info_display]


# Callback to update the choropleth map based on search results
@app.callback(
    Output("choropleth-map", "figure"),
    Input("search_results", "data")
)
def update_choropleth_map(search_results):
    if search_results == []:
        raise PreventUpdate

    results = pd.DataFrame(search_results)

    # Preprocess results for choropleth map
    results['Job match count'] = results.groupby('Country')['Country'].transform('count')
    results = results.sort_values(by='Job match count', ascending=False)
    # print(results.loc[:, ['Country', 'location', 'Job match count']])

    fig = px.choropleth(
        results,
        geojson=city_geojson,
        locations='Country',
        locationmode='country names',
        featureidkey='properties.ADM0NAME',
        color='Job match count',
        color_continuous_scale='Oranges',
        hover_name='location',
        hover_data={'Job match count': True, 'Job Title': True},
    )
    fig.update_geos(fitbounds='locations')
    return fig


@app.callback(
    Output("static-map", "figure"),
    Output("static-map-container", "style"),
    State("search_results", "data"),
    [Input(f"details-{i}", "n_clicks") for i in range(5)]
)
def update_static_map(search_results, *clicks):
    # Validate search_results
    if not search_results or len(search_results) == 0:
        # print("No search results available. Preventing update.")
        raise PreventUpdate

    # Convert to DataFrame
    results = pd.DataFrame(search_results)

    # Get triggered context
    ctx = dash.callback_context
    if not ctx.triggered:
        # print("No input triggered the callback. Preventing update.")
        raise PreventUpdate

    # Extract the triggered ID
    triggered_id = ctx.triggered[0]["prop_id"]
    print(f"Triggered ID: {triggered_id}")

    if "details-" in triggered_id:
        # Parse button index from triggered ID (e.g., "details-2.n_clicks")
        button_index = int(triggered_id.split("-")[1].split(".")[0])
        # print(f"Details button {button_index} clicked.")

        # Ensure the button has been clicked
        if clicks[button_index] <= 0:
            raise PreventUpdate

        # Get the corresponding job index from app_state
        idx = app_state["top_5_indices"][button_index]
        # print(f"Selected job index: {idx}")

        # Ensure index is valid
        if idx < 0 or idx >= len(results):
            raise PreventUpdate

        # Extract coordinates from the selected row
        selected_row = results.iloc[idx]
        latitude = float(selected_row["latitude"])
        longitude = float(selected_row["longitude"])

        # Create the map
        fig = px.scatter_mapbox(
            lat=[latitude],
            lon=[longitude],
            zoom=7,
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r": 10, "t": 10, "l": 10, "b": 10})
        return fig, {"display": "block"}
    # Debugging: If no valid button was clicked, prevent update
    else:
        raise PreventUpdate


# Main
if __name__ == "__main__":
    app.run_server(debug=True)
