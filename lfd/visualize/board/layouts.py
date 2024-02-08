import pandas as pd
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash import dash_table as dt


select_experiment = dbc.Card([
    dbc.CardHeader(html.H4(children="Experiments")),
    dbc.CardBody([
        dt.DataTable(
            id='experiment_overview', row_selectable='single',
        ),
    ], style={'height': '350px', 'overflow': 'auto'}, id='experiment_overviewcard')
], className='p-3 mb-5')

select_model = dbc.Card([
    dbc.CardHeader(html.H4(children="Models")),
    dbc.CardBody([
        dt.DataTable(
            id='model_overview', row_selectable='single',
        ),
    ], style={'height': '350px', 'overflow': 'auto'}, id='model_overviewcard')
], className='p-3 mb-5')

params = dbc.Card([
    dbc.CardHeader(html.H4(children="Parameters")),
    dbc.CardBody(
        dbc.Row([
            dbc.Col([html.H5('Data'), html.Pre(id='dataparams')], width=4),
            dbc.Col([html.H5('Transform'), html.Pre(id='transformparams')], width=4),
            dbc.Col([html.H5('Model'), html.Pre(id='modelparams')], width=4),
        ]), style={'height': '350px', 'overflow': 'auto'}
    )
], className='p-3 mb-5')

modellogs = dbc.Card([
    dbc.CardHeader(html.H4(children="Experiment Logs")),
    dbc.CardBody([
        html.Pre(html.P("", id='model_logs')),
    ], style={'height': '350px', 'overflow': 'auto'})
], className='p-3 mb-5')

metrics = dbc.Card([
    dbc.CardHeader(html.H4(children="Metrics")),
    dbc.CardBody(
        dbc.Row([
            dbc.Col(
                id='metrics_table',
                children=dbc.Table.from_dataframe(
                    pd.DataFrame(), striped=True, bordered=True, hover=True), 
                width=5
            ),
        ])
    )
], className='p-3 mb-5')

confusions = dbc.Card([
    dbc.CardHeader(html.H4(children="Confusion Matrices")),
    dbc.CardBody(dbc.Row([dcc.Graph(id="confusion")]))
], className='p-3 mb-5')

liftcurve = dbc.Card([
    dbc.CardHeader(html.H4(children="Lift Curve", id="liftscores")),
    dbc.CardBody(
        dbc.Row([
            dbc.Col([
                html.P(className="label", children="Split"),                
                dbc.RadioItems(
                    id="dataset_selector",
                    options=[{"label": v, "value": v} for v in []],
                ),
                html.P(className="label", children="Scores"),
                dbc.Checklist(
                    options=[{"label": "Calibrate", "value": True, "disabled": True}],
                    id="calibrated_check", switch=True,
                ),
                html.P(className="label", children="Bins"),                
                dbc.RadioItems(
                    id="bin_selector",
                    options=[{"label": v, "value": v} for v in [10, 20, 50]],
                    value=50
                ),
            ], width=1),
            dbc.Col([dcc.Graph(id="liftcurve_plot")], width=11)
        ])
    )
], className='p-3 mb-4')

pred_histogram = dbc.Card([
    dbc.CardHeader([
        dbc.Row([
            dbc.Col(html.H4("Histogram"), width=5),
            dbc.Col([
                html.P(className="label", children="Split"),                
                dbc.RadioItems(
                    options=[], value='', id='pred_split', inline=True,
                )], width=4
            ),
            dbc.Col([
                html.P(className="label", children="Scores"),                
                dbc.Checklist(
                    options=[{"label": "Calibrated", "value": True}],
                    value=[True], id="pred_calibrate", switch=True,
                )], width=3
            )]
        )]
    ),
    dbc.CardBody(
        dbc.Row([dbc.Col([dcc.Graph(id="predhist_plot")], width=12)])
    )
], className='p-3 mb-5')

calibration_plot = dbc.Card([
    dbc.CardHeader(html.H4("Calibration")),
    dbc.CardBody(
        dbc.Row([dbc.Col([dcc.Graph(id="calibration_plot")], width=12)])
    )
], className='p-3 mb-5')

distr_actual_plot = dbc.Card([
    dbc.CardHeader(html.H4("Per Actual")),
    dbc.CardBody(
        dbc.Row([
            dbc.Col([dcc.Graph(id="distr_actual_plot")], width=12)
        ])
    )
], className='p-3 mb-5')

feature_imp = dbc.Card([
    dbc.CardHeader([
        dbc.Row([
            dbc.Col(html.H4("Feature Importance"), width=8),
            dbc.Col(
                dbc.Checklist(
                    options=[{"label": "Shap Values", "value": True}],
                    value=[True], id="feature_imp_selector",
                    switch=True, style={'margin-top':'15px'}
                ), width=4
            )]
        )]
    ),
    dbc.CardBody([
        dbc.Row([dbc.Col([dcc.Graph(id="feature_imp_plot")], width=12)]),
    ])
], className='p-3')

shapscatter = dbc.Card([
    dbc.CardHeader(html.H4(children="Impact per Variable")),
    dbc.CardBody(
        dbc.Row([
            dbc.Col([
                html.P(className="label", children="Variable"),                
                dbc.RadioItems(
                    options=[{"label": v[:25], "value": v} for v in []],
                    id="shapvar_selector",
                ),
                html.Br(),
                html.P(className="label", children="Break By"),                
                dcc.Dropdown(
                    options=[{"label": v[:25], "value": v} for v in []],
                    id="shapbreak_selector",
                ),
            ], width=3),
            dbc.Col([dcc.Graph(id="shapscatter_plot")], width=9)
        ])
    )
], className='p-3')

shap_observation = dbc.Card([
    dbc.CardHeader(html.H4("Observation", id="obs_title")),
    dbc.CardBody([dbc.Row([dbc.Col([dcc.Graph(id="shap_obs_plot")], width=12)])])
], className='p-3')


select_data = dbc.Card([
    dbc.CardBody(
        dbc.Row([
            dbc.Col(html.H6("Load Data"), width=3),
            dbc.Col(
                dcc.Dropdown(
                options=[{"label": d, "value": d} for d in []],
                id="data_dropdown"
            ), width=9),
        ]),
    style={'height': '69px'})
], className='p-3')

filter_data = dbc.Card([
    dbc.CardBody(
        dbc.Row([
            dbc.Col(html.H6("Filter Data"), width=3),
            dbc.Col(dcc.Textarea(
                id="eda_filter", spellCheck=False, draggable=False,
                style={'width': '100%', 'height': 54, 'border-color': '#ccc', 'border-radius': '.5rem'}
            ), width=9),
        ]),
    style={'height': '69px'})
], className='p-3')

dataviewer = dbc.Card([
    dbc.CardHeader([html.H4(children="Table")]),
    dbc.CardBody(        
        dt.DataTable(
            id='dataviewer', row_selectable='single', selected_rows=[1],
        ), style={'height': '720px', 'overflow': 'auto'}, id='dataviewercard')
], className='p-3 mb-5')

metacontviewer = dbc.Card([
    dbc.CardHeader([html.H4(children="Continuous Variables")]),
    dbc.CardBody(style={'height': '720px', 'overflow': 'auto'}, id='metacontviewercard')
], className='p-3 mb-5')

metacatviewer = dbc.Card([
    dbc.CardHeader([html.H4(children="Categorical Variables")]),
    dbc.CardBody(style={'height': '720px', 'overflow': 'auto'}, id='metacatviewercard')
], className='p-3 mb-5')

analysis = dbc.Card([
    dbc.CardHeader(html.H4("Analysis")),
    dbc.CardBody(
        dbc.Row([
            dbc.Col([dcc.Graph(id="eda_plot")], width=10),
            dbc.Col([
                html.Br(),
                html.P(className="label", children="Chart"),
                dbc.RadioItems(
                    id="eda_chart", value='histogram',
                    options=[{"label": v, "value": v.lower()} for v in ['Histogram']],
                ),
                dbc.Row([
                    dbc.Col([html.Br(), html.P(className="label", children="Mode")], width=12),
                    dbc.Col([
                        dbc.Checklist(
                            options=[{"label": "Relative", "value": True}],
                            id="eda_relative", switch=True),
                        dbc.Checklist(
                            options=[{"label": "Cumulative", "value": True}],
                            id="eda_cumul", switch=True),
                        dbc.Checklist(
                            options=[{"label": "Switch Axes", "value": True}],
                            id="eda_axes", switch=True)],
                        width=12)], id='data_mode', style={'display': 'block'}),
                dbc.Row([
                    dbc.Col([html.Br(), html.P(className="label", children="Bins")], width=12),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col(
                                dcc.Dropdown(
                                    options=[{"label": v, "value": v} for v in [2, 3, 4, 5, 10, 20, 50, 100, 200]],
                                    value=50, clearable=False, id="eda_bins"), width=6),
                            dbc.Col(
                                dcc.Dropdown(
                                    options=[{"label": v, "value": v} for v in [2, 3, 4, 5, 10, 20, 50, 100, 200]],
                                    value=20, clearable=False, id="eda_bins2", style={'display': 'none'}), width=6),
                        ]),
                        html.Br(),
                        dcc.Dropdown(multi=True, id='eda_bincats', clearable=False, placeholder='Select categories', style={'display': 'block'}),
                        html.Br(),
                        dcc.Dropdown(multi=True, id='eda_bincats2', clearable=False, placeholder='Select categories', style={'display': 'block'}),
                        # dbc.DropdownMenu(
                        #     dbc.DropdownMenuItem(
                        #         dbc.Checklist(id='eda_bincats', style={'display': 'block'})
                        #     ),
                        # ),

                        html.Br(),
                        dbc.RadioItems(
                            id="eda_binmode", value='quantile', style={'display': 'none'},
                            options=[{"label": v.title(), "value": v} for v in ['quantile', 'range']],
                        ),
                        ], width=12)], id='eda_binoptions', style={'display': 'block'}),
                dbc.Row([
                    dbc.Col([
                        html.Br(), 
                        html.P(className="label", children="Sort"),
                        dbc.RadioItems(
                            id="eda_sort", value='median',
                            options=[{"label": v, "value": v.lower()} for v in ['Median', 'Name', 'Count']],
                        ),
                        html.Br(),
                        dbc.Checklist(
                            options=[{"label": "Reverse", "value": True}],
                            id="eda_sort_reverse",
                        )
                    ], width=12),
                ], id='eda_sortoptions', style={'display': 'none'}),
            ], width=2)
        ])
    )
], className='p-3 mb-4')

statistics = dbc.Card([
    dbc.CardHeader([html.H4(children="Statistics")]),
    dbc.CardBody(style={'height': '123px', 'overflow': 'auto'}, id='statisticscard')
], className='p-3 mb-5')



def header(title, content=None):
    return dbc.Card(
        container([
            html.H2(title),
            content
        ], className='header'))

def subheader(title):
    return dbc.Card(
        dbc.Container([
            html.H3(title, className='subheadertitle'),
        ], fluid=True, className="p-3 subheader"), className="mb-5")

def container(content, className='p-5 plain', identifier=''):
    return dbc.Container(content, fluid=True, className=className, id=identifier)


navbar = dbc.NavbarSimple([
    dbc.NavItem(dbc.NavLink(html.H5("Model"), href="/model")),
    dbc.NavItem(dbc.NavLink(html.H5("Data"), href="/data")),
    dbc.NavItem(html.H5(
        dbc.Checklist(
            options=[{"label": "Dark", "value": False}],
            value=[],
            id="theme_selector",
            switch=True,
            style={'margin-top':'9px'}
        ))
    ),
], id='navbar', sticky="top", dark=False, color='#444', fluid=True, links_left=True)

page = html.Div([
    container([
        subheader("Overview"),
        dbc.Row([
            dbc.Col([select_experiment], width=4),
            dbc.Col([select_model], width=4),
            dbc.Col([params], width=4),
            
        ]),
        subheader("Performance"),
        dbc.Row([
            dbc.Col([confusions], width=3),
            dbc.Col([liftcurve, metrics], width=9),
        ]),
        subheader("Predictions"),
        dbc.Row([
            dbc.Col([pred_histogram], width=4),
            dbc.Col([distr_actual_plot], width=4),
            dbc.Col([calibration_plot], width=4),
        ]),
        subheader("Explainability"),
        dbc.Row([
            dbc.Col([feature_imp], width=3),
            dbc.Col([shapscatter], width=5),
            dbc.Col([shap_observation], width=4),
        ]),
    ], identifier='modelpage'),
    container([
        dbc.Row([
            dbc.Col([select_data], width=3),
            dbc.Col([filter_data], width=3),
            dbc.Col([subheader("Exploratory Data Analysis")], width=6),
        ]),
        dbc.Row([
            dbc.Col([metacontviewer], width=3),
            dbc.Col([metacatviewer], width=3),
            dbc.Col([analysis, statistics], width=6),
        ]),
        subheader("Data"),
        dbc.Row([
            dbc.Col([dataviewer], width=12),
        ]),
    ], identifier='datapage'),
])
