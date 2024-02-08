from locale import normalize
import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dash.dependencies import Input, Output, State
from dash import no_update
import plotly.graph_objs as go

from .app import app
from .layouts import page
from lfd import PlotterModel, Pipeline, Data

# Global Variables
EXPERIMENT_PATH = getattr(app, 'DIRECTORY', 'experiments') 
PLOTTER = PlotterModel(colors='belfius', theme='light')
PIPE, MODEL, DATA = None, None, None
EXPERIMENT_OVERVIEW, MODEL_OVERVIEW = None, None

# EXPERIMENT OVERVIEW CARD
@app.callback([Output('experiment_overviewcard', 'children'), Output('data_dropdown', 'options'), 
    Output('data_dropdown', 'value')], [Input('navbar', 'children')], prevent_initial_call=False)
def update_overview(_):
    logging.info('Creating experiment table')
    global EXPERIMENT_OVERVIEW
    # Loop through all experiments
    
    experiments = [e for e in os.listdir(EXPERIMENT_PATH) if (os.path.isdir(os.path.join(EXPERIMENT_PATH, e)) \
                   and not e.startswith('.') and 'bootstrap' not in e.lower()) or e.endswith('.pkl')]
    datasets = [d for d in np.sort(os.listdir(EXPERIMENT_PATH)) if d.endswith('.parquet') or d.endswith('.csv')]
    datasets_options = [{'label': d, 'value': d} for d in datasets]
    dataset_value = datasets[0] if datasets else None
    if experiments:
        EXPERIMENT_OVERVIEW = pd.DataFrame()
        for experiment in experiments:
            path = os.path.join(EXPERIMENT_PATH, experiment)
            EXPERIMENT_OVERVIEW.loc[experiment, 'Experiment'] = experiment
            EXPERIMENT_OVERVIEW.loc[experiment, 'Timestamp'] = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
            EXPERIMENT_OVERVIEW.loc[experiment, 'Size'] = f'{round(os.path.getsize(path)/1024/1024, 2)} Mb'
            EXPERIMENT_OVERVIEW.loc[experiment, 'Format'] = 'Pickle' if experiment.endswith('.pkl') else 'Directory'
            EXPERIMENT_OVERVIEW.sort_values('Timestamp', inplace=True, ascending=False)
        experiment_overview = PLOTTER.make_datatable(
            'experiment_overview', EXPERIMENT_OVERVIEW, left_align=['Experiment'], 
            **dict(row_selectable='single', selected_rows=[0]))
    else: experiment_overview = no_update
    return experiment_overview, datasets_options, dataset_value

# MODEL OVERVIEW CARD
@app.callback([Output('model_overviewcard', 'children'), Output("dataparams", "children"), 
    Output("transformparams", "children"), Output("modelparams", "children"), Output("dataviewercard", "children"), 
    Output("metacontviewercard", "children"), Output("metacatviewercard", "children")], 
    [Input('experiment_overview', 'selected_rows'), Input("data_dropdown", "value"), 
    Input("eda_filter", "n_blur")], [State("eda_filter", "value")])
def get_experiment(overviewid, dataset, _, filter_string):
    logging.info('Creating model table')
    global DATA
    if overviewid:
        global MODEL_OVERVIEW, EXPERIMENT_OVERVIEW, PIPE
        experiment = EXPERIMENT_OVERVIEW.iloc[overviewid[0]]['Experiment']
        PIPE = Pipeline().load(os.path.join(EXPERIMENT_PATH, experiment), slim=False)
        PIPE.data = PIPE.data.sample(2000, replace=False) if PIPE.data.shape[0] > 2000 else PIPE.data
        PIPE.data.df = PIPE.data.df.droplevel('dataset')
        DATA = PIPE.data

        # Model overview table
        MODEL_OVERVIEW = pd.DataFrame()
        for name, model in PIPE.models.items():
            PIPE.models[name].shapvalues = PIPE.models[name].shapvalues.droplevel('dataset').loc[PIPE.data.index]
            MODEL_OVERVIEW.loc[name, 'Model'] = name
            MODEL_OVERVIEW.loc[name, 'Algorithm'] = model.__class__.__name__
            MODEL_OVERVIEW.loc[name, 'Use Case'] = model.mode
            MODEL_OVERVIEW.loc[name, 'Target'] = model.target
            MODEL_OVERVIEW.loc[name, '# Features'] = int(len(model.features))
            MODEL_OVERVIEW.loc[name, 'Calibrated'] = "Yes" if PIPE.cal_models and PIPE.cal_models.get(name) else "No"
        model_overview = PLOTTER.make_datatable(
            'model_overview', MODEL_OVERVIEW, left_align=['Model'], **dict(row_selectable='single', selected_rows=[0]))
        # Parameters
        dataparams = json.dumps(PIPE.params['data'], indent=4)
        transformparams = json.dumps(PIPE.params['transform'], indent=4)
        modelparams = json.dumps(PIPE.params['model'], indent=4)

    elif dataset:
        DATA = Data(os.path.join(EXPERIMENT_PATH, dataset))
        model_overview = dataparams = transformparams = modelparams = no_update

    # Filtering
    try: 
        df = DATA.df
        df = df[eval(filter_string)] if filter_string else df
        DATA.df = df
    except: return (no_update,) * 7

    # Data & description tables
    datatable = PLOTTER.make_datatable('dataviewer', DATA.df.reset_index().head(100), page_size=20, height='650px')
    num_vars = DATA.df[DATA.cont_columns]
    cat_vars = DATA.df[DATA.cat_columns]
    if not num_vars.empty: 
        meta_cont = num_vars.describe().T.round(2).reset_index()
        meta_cont = PLOTTER.make_datatable('metacontviewer', meta_cont, height='660px', left_align=['index'], 
            page_size=20, **dict(row_selectable='multi', selected_rows=[0]))
    else: meta_cont = PLOTTER.make_datatable('metacontviewer', pd.DataFrame(), height='660px')
    if not cat_vars.empty: 
        meta_cat = cat_vars.describe().T.round(2).reset_index()
        meta_cat = PLOTTER.make_datatable('metacatviewer', meta_cat, page_size=20,
            height='660px', left_align=['index'], **dict(row_selectable='multi', selected_rows=[]))
    else: meta_cat = PLOTTER.make_datatable('metacatviewer', pd.DataFrame(), height='660px')
    return model_overview, dataparams, transformparams, modelparams, datatable, meta_cont, meta_cat

# PIPE UPDATE
@app.callback(
    [Output("dataset_selector", "options"),     Output("dataset_selector", "value"),
     Output("pred_split", "options"),           Output("pred_split", "value"),
     Output("metrics_table", "children"),       Output("confusion", "figure"),
     Output("shapvar_selector", "options"),     Output("shapvar_selector", "value"),
     Output("shapbreak_selector", "options"),   Output("shapbreak_selector", "value"),
     Output("calibrated_check", "options"),     Output("calibrated_check", "value"),
     Output("pred_calibrate", "options"),       Output("pred_calibrate", "value")], 
     [Input('model_overview', 'selected_rows')])
def get_model(overviewid):
    logging.info('Update model')
    global MODEL_OVERVIEW, PIPE, MODEL
    # Load pipeline
    MODEL = MODEL_OVERVIEW.iloc[overviewid[0]]['Model']
    model = PIPE.cal_models[MODEL] if PIPE.cal_models else PIPE.models[MODEL]

    # Set dataset options, variable options, metrics, confusion
    datasplits = list(model.confusion.index.levels[0])
    datasplit_options = [{'label': d, 'value': d} for d in datasplits]
    metrics = model.metrics[model.metrics.columns.levels[0][0]] # Get first predictions
    metrics = PLOTTER.make_datatable('metrics', metrics.reset_index().round(3), height='100px')
    confusion = PLOTTER.confusion_heatmaps(model.confusion)
    variables = PIPE.models[MODEL].shapvalues.abs().mean().sort_values(ascending=False).index[:20] \
        if PIPE.models[MODEL].shapvalues is not None else PIPE.data.columns[:20]
    var_options = [{'label': i[:20], 'value': i} for i in variables]
    var_first = variables[0] if len(variables) > 0 else ''
    if PIPE.cal_models: caloptions, value = [{"label": "Calibrated", "value": True, "disabled": False}], [True]
    else: caloptions, value = [{"label": "Calibrated", "value": False, "disabled": True}], []
    return (datasplit_options, datasplits[0], datasplit_options, datasplits[0], 
        metrics, confusion, var_options, var_first, var_options, None, caloptions, value, caloptions, value)

# LIFTCURVE CARD
@app.callback(Output("liftcurve_plot", "figure"), [Input("dataset_selector", "value"),
              Input("calibrated_check", "value"), Input("bin_selector", "value")])
def get_liftcurve(dataset, calibrated, bins):
    model = PIPE.cal_models[MODEL] if len(calibrated)>0 else PIPE.models[MODEL]
    fig = PLOTTER.lift_curve(model.predictions.df, dataset=dataset, bins=bins)
    return fig

# CALIBRATION CARD
@app.callback(Output("calibration_plot", "figure"), [Input("pred_calibrate", "value")])
def get_calibration(calibrated):
    model = PIPE.cal_models[MODEL] if len(calibrated)>0 else PIPE.models[MODEL]
    fig = PLOTTER.line_chart(model.predictions.df)
    return fig

# HISTOGRAM CARD
@app.callback(Output("predhist_plot", "figure"),
              [Input("pred_calibrate", "value"), Input("pred_split", "value")])
def get_histogram(calibrated, dataset):
    if len(calibrated)>0: preds = PIPE.cal_models[MODEL].predictions.df
    else: preds = PIPE.models[MODEL].predictions.df
    fig = PLOTTER.histogram(preds.loc[dataset, 'scores'], preds.loc[dataset, 'target'])
    return fig

# BOXPLOT CARD
@app.callback(Output("distr_actual_plot", "figure"),
              [Input("pred_calibrate", "value"), Input("pred_split", "value")])
def get_boxplots(calibrated, dataset):
    if len(calibrated)>0: preds = PIPE.cal_models[MODEL].predictions.df
    else: preds = PIPE.models[MODEL].predictions.df
    targetbins = pd.qcut(preds.target, 5, duplicates='drop') if PIPE.models[MODEL].mode=='linear' else preds.target
    fig = PLOTTER.boxplots(targetbins.loc[dataset].rename('Actual'), preds.scores.loc[dataset].rename('Score'))
    return fig

# FEATURE IMPORTANCE CARD
@app.callback(Output("feature_imp_plot", "figure"),
              [Input("distr_actual_plot", "figure"), Input("feature_imp_selector", "value")])
def get_feature_imp(_, original=False):
    global PIPE, MODEL
    shapvalues = PIPE.models[MODEL].shapvalues
    if len(original)>0 and shapvalues is not None:
        feature_imp = shapvalues.abs().mean().sort_values(ascending=False).head(20)
        title = 'Shap Feature Importance'
    elif PIPE.models[MODEL].feature_imp is not None:
        feature_imp = PIPE.models[MODEL].feature_imp.head(20)
        title = 'Relative Feature Importance'
    else: return PLOTTER.spaceholder()
    fig = PLOTTER.plot_barchart(feature_imp, title)
    return fig

# SHAPSCATTER CARD
@app.callback(Output("shapscatter_plot", "figure"), [Input("shapvar_selector", "value"), 
              Input("shapbreak_selector", "value"), Input('shapscatter_plot', 'clickData')])
def get_shapscatter(variable, break_var, identifier=None):
    logging.info('Creating shap scatterplot')
    global PIPE, MODEL
    shapsample = PIPE.models[MODEL].shapvalues
    if shapsample is None: return PLOTTER.spaceholder()
    identifier = identifier['points'][0]['hovertext'] if identifier is not None else identifier
    color = None if break_var is None else PIPE.data.df[break_var].copy()
    fig = PLOTTER.scatter(x=PIPE.data.df[variable], y=shapsample[variable].rename('SHAP'), z=color,
        identifier=identifier, height=559)
    return fig

# SHAPOBSERVATION CARD
@app.callback([Output("shap_obs_plot", "figure"), Output("obs_title", "children")],
              [Input('shapscatter_plot', 'figure'), Input('shapscatter_plot', 'clickData')])
def get_shapobservation(_, identifier=None):
    shapsample = PIPE.models[MODEL].shapvalues
    if shapsample is None: return PLOTTER.spaceholder(), ''
    fig = go.Figure()
    if identifier is not None: identifier = identifier['points'][0]['hovertext']
    if identifier not in shapsample.index: identifier = None
    if identifier is None: identifier = shapsample.iloc[0].name
    shapsample = shapsample.loc[identifier].iloc[:20]
    shapsample = shapsample.reindex(shapsample.abs().sort_values(ascending=False).index)
    
    target_variable = PIPE.models[MODEL].target
    target_value = PIPE.data.df.loc[identifier, target_variable]
    datasample = PIPE.data.df.loc[shapsample.name, shapsample.index]

    obs_title = f'Example {shapsample.name} - {target_variable}: {target_value}' 
    fig = PLOTTER.shap_observation(shapsample, datasample)
    return fig, obs_title

# DATA OPTIONS
@app.callback([Output("eda_chart", "options"), 
    Output("eda_chart", "value"), Output("eda_bincats", "options"), Output("eda_bincats2", "options")], 
    [Input("metacontviewer", "selected_rows"), Input("metacatviewer", "selected_rows")])
def data_options(cont_vars, cat_vars):
    logging.info('Gathering data')
    global CONT_VARS, CAT_VARS, DATA
    CONT_VARS = cont_vars if cont_vars else []
    CAT_VARS = cat_vars if cat_vars else []
    for i, v in enumerate(CONT_VARS): CONT_VARS[i] = DATA.df[DATA.cont_columns[v]]
    for i, v in enumerate(CAT_VARS): CAT_VARS[i] = DATA.df[DATA.cat_columns[v]]
    options, value = [{"label": v.title(), "value": v} for v in ['histogram']], 'histogram'
    bin_cats = [{"label": str(v), "value": str(v)} for v in CAT_VARS[0].value_counts().index] if CAT_VARS else []
    bin_cats2 = [{"label": str(v), "value": str(v)} for v in CAT_VARS[1].value_counts().index] if CAT_VARS and len(CAT_VARS)==2 else []
    if len(CONT_VARS) == 2:
        options, value = [{"label": v.title(), "value": v} for v in ['heatmap', 'boxplot', 'scatter']], 'heatmap'
    elif len(CONT_VARS) == 1:
        if len(CAT_VARS) == 1: 
            options, value = [{"label": v.title(), "value": v} for v in ['boxplot', 'heatmap']], 'boxplot'
        elif len(CAT_VARS) == 0: 
            options, value = [{"label": v.title(), "value": v} for v in ['histogram']], 'histogram'
    elif len(CONT_VARS) == 0:
        if len(CAT_VARS) == 1: 
            options, value = [{"label": v.title(), "value": v} for v in ['histogram']], 'histogram'
        elif len(CAT_VARS) == 2: 
            options, value = [{"label": v.title(), "value": v} for v in ['heatmap']], 'heatmap'
    return options, value, bin_cats, bin_cats2

# DATA ANALYSIS
@app.callback([Output("eda_bins2", "style"), Output("eda_sortoptions", "style"), Output("data_mode", "style"), 
    Output("eda_binoptions", "style"), Output("eda_binmode", "style"), Output("eda_relative", "style"), 
    Output("eda_cumul", "style"), Output("eda_bincats", "style"), Output("eda_bincats2", "style"), Output("eda_plot", "figure")], 
    [Input("eda_relative", "value"), Input("eda_cumul", "value"), Input("eda_axes", 'value'), 
    Input("eda_chart", 'value'), Input("eda_bins", "value"), Input("eda_bins2", "value"), 
    Input("eda_binmode", "value"), Input("eda_bincats", "value"), Input("eda_bincats2", "value"), 
    Input("eda_sort", "value"), Input("eda_sort_reverse", "value")])
def data_analysis(normalize, cumulative, switch, chart, bins1, bins2, bin_mode, bin_cats, bin_cats2, sort, reverse):
    logging.info('Plotting data')
    # Default option styles
    d0, d1 = {'display': 'none'}, {'display': 'block'}
    bins2_style, sort_style, binmode_style = d0, d0, d0
    bincats_style, bincats2_style = d0, d0
    mode_style, allbins_style = d1, d1
    relative_style, cumul_style = d1, d1
    
    # Triage
    box_kwargs = dict(bins=bins1, bin_mode=bin_mode, bin_cats=bin_cats, sort=sort, reverse=reverse, switch=switch)
    heat_kwargs = dict(bins=bins1, bins2=bins2, bin_mode=bin_mode, bin_cats=bin_cats, bin_cats2=bin_cats2, normalize=normalize, switch=switch, cumulative=cumulative)
    hist_kwargs = dict(bins=bins1, bin_cats=bin_cats, normalize=normalize, switch=switch, cumulative=cumulative)
    fig = PLOTTER.spaceholder()
    if len(CONT_VARS) == 2:
        if chart=='heatmap': 
            fig = PLOTTER.heatmap(CONT_VARS[1], CONT_VARS[0], **heat_kwargs)
            bins2_style, binmode_style = d1, d1
        elif chart=='scatter': 
            fig = PLOTTER.scatter(CONT_VARS[0], CONT_VARS[1], switch=switch)
            allbins_style, relative_style, cumul_style =  d0, d0, d0
        else: 
            fig = PLOTTER.boxplots(CONT_VARS[0], CONT_VARS[1], **box_kwargs)
            binmode_style, relative_style, cumul_style = d1, d0, d0
    elif len(CONT_VARS) == 1:
        if len(CAT_VARS) == 1:
            if chart=='boxplot': 
                fig = PLOTTER.boxplots(CONT_VARS[0], CAT_VARS[0], **box_kwargs)
                sort_style, mode_style, bincats_style = d1, d0, d1
            elif chart=='heatmap': 
                fig = PLOTTER.heatmap(CAT_VARS[0], CONT_VARS[0], **heat_kwargs)
                bins2_style, binmode_style, bincats_style = d1, d1, d1
        elif len(CAT_VARS) == 0: 
            fig = PLOTTER.histogram(CONT_VARS[0], **hist_kwargs)
    elif len(CONT_VARS) == 0:
        if len(CAT_VARS) == 1: 
            fig = PLOTTER.histogram(CAT_VARS[0], **hist_kwargs)
            bincats_style = d1
        elif len(CAT_VARS) == 2: 
            fig = PLOTTER.heatmap(CAT_VARS[1], CAT_VARS[0], **heat_kwargs)
            bins2_style, bincats_style, bincats2_style = d1, d1, d1
    return bins2_style, sort_style, mode_style, allbins_style, binmode_style, relative_style, cumul_style, bincats_style, bincats2_style, fig

# PAGE UPDATE
@app.callback([Output('modelpage', 'style'), Output('datapage', 'style')], [Input('url', 'pathname')])
def display_page(pathname):
    if pathname in ('/', '/model'):
        model_viz, data_viz = {'display': 'block'}, {'display': 'none'}
    elif pathname in ('/data', ):
        model_viz, data_viz = {'display': 'none'}, {'display': 'block'}
    else:
        model_viz, data_viz = {'display': 'none'}, {'display': 'none'}
    return model_viz, data_viz

# THEM UPDATE
@app.callback([Output('link', 'href'), Output('page-content', 'children')], [Input('theme_selector', 'value')])
def switch_theme(theme):
    theme = 'light' if len(theme)==0 else 'dark'
    global PLOTTER
    PLOTTER = PlotterModel('belfius', theme)
    style = '/assets/styles_light.css' if theme=='light' else '/assets/styles_dark.css'
    return style, page