import numpy as np
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc

# Define constants
QUALS = ["Bachelor", "Master", "MBA", "PHD"]
WORK_TYPES = ["Full-Time", "Part-Time", "Contract", "Intern", "Temporary"]
COMPANY_SIZES = ["Small", "Medium", "Large"]
INDUSTRY = ['Energy/Utilites', 'Retail', 'Healthcare/Pharmaceuticals',
            'Financial Services', 'Appliances/Equipment', 'Other',
            'Technology', 'Manufacturing/Industrial', 'Transportation/Logistics',
            'Media/Entertainment', 'Food/Beverage', 'Automotive', 'Engineering',
            'Construction/Materials', 'Mining/Chemicals', 'Aerospace/Defense',
            'Telecommunications', 'Travel/Hospitality']

user_keys = {
    'salary_min', 'salary_weight',
    'experience', 'experience_weight',
    'skills', 'skills_weight',
    'qualification', 'qualification_weight',
    'longitude', 'latitude', 'location_weight',
    'work_type', 'work_type_weight',
    'company_size', 'company_size_weight',
    'job_title', 'job_title_weight',
    'skills', 'skills_weight',
    'sector1', 'sector2', 'sector3', 'sector_weight'
}

score_list = [
    "salary_score", "experience_score", "qualification_score",
    "location_score", "work_type_score", "company_size_score",
    "sector_score", "title_score", "skills_score"
]


with open('cities.geojson', encoding = 'utf-8') as f:
    geojson = json.load(f)

# Left Column Layout
left_column = html.Div(
    [
        # Job Title Section
        html.Div(
            className="input_container",
            id="job_title_container",
            children=[
                dcc.Input(
                    id="job_title_weight",
                    type="number",
                    min=1,
                    max=5,
                    step=1,
                    value=3,  # Default rank value
                    className="weights"
                ),
                html.P("Job Title:", className="control_label"),
                dcc.Input(
                    id="job_title_selector",
                    placeholder="Enter Job Title",
                    type="text",
                    className="dcc_control"
                ),
            ]
        ),

        # Skills Section
        html.Div(
            className="input_container",
            id="skills_container",
            children=[
                dcc.Input(
                    id="skills_weight",
                    type="number",
                    min=1,
                    max=5,
                    step=1,
                    value=3,  # Default rank value
                    className="weights"
                ),
                html.P("Skills:", className="control_label"),
                dcc.Dropdown(
                    id="skills_selector",
                    placeholder="Enter Skills",
                    multi=True,
                    options=[],
                    searchable=True,
                    clearable=True,
                    className="dcc_control"
                ),
            ]
        ),

        # Qualifications Section
        html.Div(
            className="input_container",
            id="qualification_container",
            children=[
                dcc.Input(
                    id="qualification_weight",
                    type="number",
                    min=1,
                    max=5,
                    step=1,
                    value=3,  # Default rank value
                    className="weights"
                ),
                html.P("Qualifications:", className="control_label"),
                dcc.RadioItems(
                    id="qualification_selector",
                    options=[{"label": qual, "value": qual} for qual in QUALS],
                    value=None,
                    labelStyle={"display": "inline-block",
                                "marginRight": "10px"},
                    className="dcc_control"
                ),
            ]
        ),

        # Experience Section
        html.Div(
            className="input_container",
            id="experience_container",
            children=[
                dcc.Input(
                    id="experience_weight",
                    type="number",
                    min=1,
                    max=5,
                    step=1,
                    value=3,  # Default rank value
                    className="weights"
                ),
                html.P("Experience (in years):", className="control_label"),
                dcc.Input(
                    id="experience_selector",
                    type="number",
                    min=0,
                    step=1,
                    value=None,
                    className="dcc_control"
                ),
            ]
        ),

        # Location Section
        html.Div(
            className="input_container",
            id="location_container",
            children=[
                dcc.Input(
                    id="location_weight",
                    type="number",
                    min=1,
                    max=5,
                    step=1,
                    value=3,  # Default rank value
                    className="weights"
                ),
                html.P("Location:", className="control_label"),
                dcc.Dropdown(
                    id='location_selector',
                    placeholder="Enter Location",
                    options=[],  # Initially empty, will be updated dynamically
                    multi=False,
                    searchable=True,
                    clearable=True,
                    className="dcc_control"
                ),
            ]
        ),

        # Work Type Section
        html.Div(
            className="input_container",
            id="work_type_container",
            children=[
                dcc.Input(
                    id="work_type_weight",
                    type="number",
                    min=1,
                    max=5,
                    step=1,
                    value=3,  # Default rank value
                    className="weights"
                ),
                html.P("Work Type:", className="control_label"),
                dcc.Dropdown(
                    id="work_type_selector",
                    options=[{"label": work, "value": work}
                             for work in WORK_TYPES],
                    value=None,
                    multi=False,
                    className="dcc_control"
                ),
            ]
        ),

        # Company Size Section
        html.Div(
            className="input_container",
            id="company_size_container",
            children=[
                dcc.Input(
                    id="company_size_weight",
                    type="number",
                    min=1,
                    max=5,
                    step=1,
                    value=3,  # Default rank value
                    className="weights"
                ),
                html.P("Company Size:", className="control_label"),
                dcc.Dropdown(
                    id="company_size_selector",
                    options=[{"label": size, "value": size}
                             for size in COMPANY_SIZES],
                    value=None,
                    multi=False,
                    className="dcc_control"
                ),
            ]
        ),

        # Sector Section
        html.Div(
            className="input_container",
            id="sector_container",
            children=[
                dcc.Input(
                    id="sector_weight",
                    type="number",
                    min=1,
                    max=5,
                    step=1,
                    value=3,  # Default rank value
                    className="weights"
                ),
                html.P("Sector:", className="control_label"),
                dcc.Dropdown(
                    id="sector1_selector",
                    options=[{"label": industry, "value": industry}
                             for industry in INDUSTRY],
                    placeholder="Select Sector 1",
                    className="dcc_control"
                ),
                dcc.Dropdown(
                    id="sector2_selector",
                    options=[{"label": industry, "value": industry}
                             for industry in INDUSTRY],
                    placeholder="Select Sector 2",
                    className="dcc_control"
                ),
                dcc.Dropdown(
                    id="sector3_selector",
                    options=[{"label": industry, "value": industry}
                             for industry in INDUSTRY],
                    placeholder="Select Sector 3",
                    className="dcc_control"
                ),
            ]
        ),

        # Salary Section
        html.Div(
            className="input_container",
            id="salary_container",
            children=[
                dcc.Input(
                    id="salary_weight",
                    type="number",
                    min=1,
                    max=5,
                    step=1,
                    value=3,  # Default rank value
                    className="weights"
                ),
                html.P("Salary:", className="control_label"),
                dcc.Slider(
                    id="salary_selector",
                    min=0,
                    max=150000,
                    marks={
                        i: f"${int(i/1000)}k" for i in range(0, 160000, 25000)},
                    step=25000,
                    className="dcc_control"
                ),
            ]
        ),
        # Search Button
        html.Button("Search", id="search_button", n_clicks=0),
        html.P(),
    ], id="user-input-panel",
)

main_page = html.Div(
    [
        html.Div(
            [
                html.Div(id=f"chart-{i}", children=[
                    dcc.Graph(id=f"graph-{i}"),
                    html.Div([
                        html.Button("Remove", id=f"remove-{i}", n_clicks=0),
                        html.Button("Details", id=f"details-{i}", n_clicks=0)
                    ])
                ]) for i in range(5)
            ], id="chart-container"),
        html.Div([
            html.Div(id="info-display"),
            # Add static city map at the bottom of job description page
            html.Div([
                html.H2("Static City Map"),
                dcc.Graph(id='static-map')
            ], id="static-map-container", style={"display": "none"})
        ], id="info-container")
    ],
    id="main-container", className="main-container", style={"display": "none"}
)
