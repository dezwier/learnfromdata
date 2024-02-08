import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dash_table as dt
from plotly.subplots import make_subplots


class Plotter:
    '''
    This class collects methods for visualization.

    Arguments
    ---------
    colors : String, default 'belfius'
            Defines which color scheme to use accross al charts. Should be
            one of ('belfius', 'google', 'amazon', 'facebook')
    theme : String, default 'light'
            Whether to run all visualizations in light or dark mode. Should
            be in ('light', 'dark').

    Attributes
    ----------
    Plotter.colors : List[String]
            List of color hexacodes to use throughout the visualizations.
    Plotter.theme : String
            Whether all visualizations run in light or dark mode. Either 'light' or 'dark'.
    '''
    def __init__(self, colors=None, theme='light'):
        color_dict = {
            "google": ["#4285f4", "#34a853", "#fbbc05", "#ea4335"]*10,
            "amazon": ["#ff9900", "#146eb4", "#232f3e"]*10,
            "facebook": ["#3b5998", "#8b9dc3", "#dfe3ee"]*10,
            "ngdata": ["#3E9539", "#3244C9", "#252827"]*10,
            "belfius": ['#CD5C5C', '#8e8e8e', '#ffaa00', '#00a4f7', '#029900']*10
        }
        self.colors = color_dict['belfius'] if colors is None else color_dict[colors]
        self.theme = 'plotly_dark' if theme=='dark' else 'plotly_white'
        self._plotcolor = "#fff" if self.theme=='plotly_white' else "#333"
        self._kwargs = dict(template=self.theme, paper_bgcolor=self._plotcolor, plot_bgcolor=self._plotcolor)

    def heatmap(self, x=None, y=None, z=None, data=None, aggfunc=None, bins=5, bins2=None, normalize=False, 
                cumulative=False, switch=False, bin_mode='quantile', bin_cats=None, bin_cats2=None):
        '''
        Plots a heatmap. Either 3 pandas Series are given (x, y, and z=color) and the function 
        aggregates them, or the data to heatmap is given straightaway.
        '''
        if data is not None: conf = data
        else:
            x_numeric = x.dtype.kind in 'iufc'
            y_numeric = y.dtype.kind in 'iufc'

            x, y = x.reset_index(drop=True), y.reset_index(drop=True)
            if not x_numeric and not y_numeric and (bin_cats2 or bin_cats): 
                bin_cats, bin_cats2 = bin_cats2, bin_cats
            if bin_cats and not x_numeric:
                mask = x.astype(str).isin(bin_cats)
                x, y = (x[mask], y[mask])
            if bin_cats2 and not y_numeric:
                mask = y.astype(str).isin(bin_cats2)
                x, y = (x[mask], y[mask])
            if switch: (x, y) = (y, x)
            bins2 = bins2 if bins2 else bins
            cut = pd.qcut if bin_mode=='quantile' else pd.cut
            bins, bins2 = min(bins, 20), min(bins2, 20)

            # Group x and y
            assert (x is not None and y is not None), 'Either give a Dataframe, or x - y Series to heatmap.'
            x_togroup = bins < x.nunique()
            if x_numeric and x_togroup: x = cut(x, bins, duplicates='drop')
            elif not x_numeric and x_togroup:
                x = x.astype(str)
                x[x.isin(x.value_counts().index[bins-1:])] = '<Other>'
            y_togroup = bins2 < y.nunique()
            if y_numeric and y_togroup: y = cut(y, bins2, duplicates='drop')
            elif not y_numeric and y_togroup:
                y = y.astype(str)
                y[y.isin(y.value_counts().index[bins2-1:])] = '<Other>'

            # Build crosstab
            conf = pd.crosstab(x, y, z, aggfunc=aggfunc, normalize=0 if normalize else False).round(3)
            if not x_numeric: conf = conf.loc[x.value_counts().index[:bins][::-1]]
            if not y_numeric: conf = conf.loc[:, y.value_counts().index[:bins2]]
            if cumulative: conf = conf.cumsum(axis=1)
            if x_numeric and x_togroup:
                conf.index = conf.index.to_series().apply(lambda x: f'{x.left.round(3)}+')
            else: conf.index = conf.index.astype(str)+'_'
            if y_numeric and y_togroup:
                conf.columns = conf.columns.to_series().apply(lambda x: f'{x.left.round(3)}+')
            else: conf.columns = conf.columns.astype(str)+'_'
            conf['All'] = conf.sum(axis=1).round(2) if not cumulative else conf.iloc[:, -1]
            conf.index.name, conf.columns.name = x.name, y.name

        # Build figure
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Heatmap(x=conf.columns, y=conf.index, z=conf.assign(All=0).values, 
            hoverinfo='skip', showscale=False, xaxis="x", yaxis="y", zmin=conf.iloc[:, :-1].min().min(),
            zmax=conf.iloc[:, :-1].max().max(), colorscale=[[0, self._plotcolor], [1, self.colors[0]]]
        ), col=1, row=1)

        # Add value annotations on heatmaps
        if normalize: conf = conf.round(2)
        annotations = [dict(text=str(conf.loc[n, m]), showarrow=False, x=m, y=n,
                            xref="x", yref="y") for m in conf.columns for n in conf.index]
        # Update layout
        height = min(490, conf.shape[0]*40+180)
        fig.update_layout(
            annotations=annotations, yaxis=dict(title=conf.index.name), xaxis=dict(title=conf.columns.name), 
            autosize=True, height=height, margin=dict(l=80, r=20, b=20, t=40), **self._kwargs)
        return fig

    def line_chart(self, preds):
        '''
        Plots a linechart.
        '''
        binscores = get_binscores(preds, bins=50)
        x = binscores.target
        y = binscores.scores.drop('Total')
        counts = binscores['count'].drop('Total')

        variables = y.columns
        lin_max = max(y.max().max(), x.max().max())
        lin_min = min(y.min().min(), x.min().min())
        linspace = np.linspace(lin_max, lin_min, num=500)

        fig = go.Figure()
        for v, variable in enumerate(variables):
            fig.add_trace(go.Scatter(
                x=x[variable], y=y[variable], marker=dict(color=self.colors[v], size=4), name=variable, 
                opacity=0.8, text=counts[variable], textposition='middle center', mode='lines+markers', line_shape='spline'))
        fig.add_trace(go.Scatter(
            x=linspace, y=linspace, mode='lines',name='Ideal Prevalence', hoverinfo='skip',
            marker=dict(color='rgba(200,200,200,.5)', size=1), showlegend=False))

        fig.update_xaxes(title_text="Score")
        fig.update_yaxes(title_text="Prevalence", ticks="")
        fig.update_layout(legend=dict(x=.75, y=0.1), autosize=True, height=490, 
            margin=dict(l=40, r=10, b=50, t=20, pad=4), **self._kwargs)
        return fig

    def boxplots(self, x, y, bins=10, bin_mode='quantile', sort='value', reverse=False, switch=False, bin_cats=None):
        '''
        Plots boxplots. Needs one categorical and one numerical variable.
        '''
        # Assert at least 1 continuous variable, put categorical one on x-axis if present
        assert x.dtype.kind in 'iufc' or y.dtype.kind in 'iufc', "At least x or y must be continuous."
        if x.dtype.kind in 'iufc' and y.dtype.kind not in 'iufc': x, y = (y, x)
        elif x.dtype.kind in 'iufc' and y.dtype.kind in 'iufc' and switch: x, y = (y, x)
        if bin_cats: 
            mask = x.isin(bin_cats)
            x, y = x[mask], y[mask]
        reverse = True if reverse else False

        # Group x variable, be it continuous or categorical
        cut = pd.qcut if bin_mode=='quantile' else pd.cut
        x_numeric = x.dtype.kind in 'iufc'
        x_togroup = bins < x.nunique()
        if x_numeric and x_togroup: x = cut(x, bins, duplicates='drop')
        elif not x_numeric and x_togroup:
            x = x.astype(str)
            to_show = x.value_counts().index if sort=='count' \
                else pd.Series(x.unique()).sort_values() if sort=='name' \
                else y.groupby(x).median().sort_values().dropna().index
            to_show = to_show[-bins+1:] if reverse else to_show[:bins-1]
            x[~x.isin(to_show)] = '<Other>'

        # Create boxplot
        boxplot = pd.concat((
            y.groupby(x).quantile([0.01, 0.25, 0.5, 0.75, 0.99]), 
            y.groupby(x).agg(['count', 'mean']).stack())).unstack(-1)
        boxplot.columns = boxplot.columns.astype(str)
        if x_numeric: boxplot = boxplot.sort_index()
        else: boxplot = boxplot.sort_index(ascending=not reverse) if sort=='name' else \
            boxplot.sort_values('count', ascending=reverse) if sort=='count' else \
            boxplot.sort_values('0.5', ascending=not reverse)
        if x_numeric and x_togroup: boxplot.index = boxplot.index.to_series().apply(
            lambda x: f'{x.left.round(3)}+' if type(x)!=float else x).astype(str)

        # Build figure
        fig = go.Figure()
        for i, cat in enumerate(boxplot.index):
            d = boxplot.loc[cat]
            c = 0 if np.isnan(d["count"]) else int(d["count"])
            fig.add_trace(
                go.Box(x=[f"{cat} (n={c})"], q1=[d['0.25']], q3=[d['0.75']], median=[d['0.5']], 
                upperfence=[d['0.99']], lowerfence=[d['0.01']], marker_color=self.colors[i%2], line=dict(width=1.5)))
        fig.update_layout(showlegend=False, yaxis=dict(title=y.name), autosize=True, height=490, 
            margin=dict(l=40, r=10, b=50, t=20, pad=4), xaxis=dict(title=x.name), **self._kwargs)
        return fig

    def plot_barchart(self, x, title):
        '''
        Plots a barchart.
        '''
        max_letters = min(x.index.to_series().apply(len).max(), 25)
        trace = go.Bar(x=x, y=x.index.str.slice(0, 25), orientation='h',
            marker=dict(color=x, colorscale=[[0.0, "#888"], [1.0, self.colors[0]]]))
        layout = go.Layout(showlegend=False, yaxis=dict(autorange="reversed", dtick=1), 
            xaxis=dict(title=title), height=len(x)*25+60, 
            margin=dict(t=30, l=max_letters*7, r=0, b=10, pad=10), **self._kwargs)
        fig = go.Figure([trace], layout)
        return fig

    def histogram(self, x, y=None, z=None, bins=20, normalize=False, cumulative=False, switch=False, bin_cats=None):
        '''
        Plots a histogram.
        '''
        df = pd.concat([series for series in [x, y, z] if series is not None], axis=1)
        if df.columns.duplicated().any(): df.columns = [f"{c}{i}" for i, c in enumerate(df, 1)]
        if bin_cats: df = df[x.isin(bin_cats)]

        # Group variable(s)
        is_numeric = x.dtype.kind in 'iufc'
        to_group = is_numeric and bins < df.nunique().max()
        if to_group:
            bins = np.linspace(df.min().min(), df.max().max(), bins+1).round(3)
            for c in df: df[c] = pd.cut(df[c], bins=bins, include_lowest=True)
        df = pd.concat([df[c].value_counts(normalize=normalize) for c in df], axis=1, keys=df.columns)
        if is_numeric: df.sort_index(inplace=True) 
        if not is_numeric: df = pd.concat(
            (df.iloc[:bins-1], df.iloc[bins-1:].sum(axis=0).rename('Other').to_frame().T), axis=0)
        if cumulative: df = pd.concat([df[c].cumsum() for c in df], axis=1)
        if to_group: df.index = df.index.to_series().apply(
            lambda x: f'{x.left.round(3)}+' if type(x)!=float else x).astype(str)

        # Build figure
        if switch: traces = [go.Bar(y=df.index.astype(str), x=df[c], marker=dict(
            color=self.colors[i]), name=c, orientation='h') for i, c in enumerate(df)]
        else: traces = [go.Bar(x=df.index.astype(str), y=df[c], marker=dict(
            color=self.colors[i]), name=c) for i, c in enumerate(df)]
        layout = go.Layout(autosize=True, height=490, margin=dict(t=20, r=20, b=50, l=20), 
            barmode='group', yaxis=dict(autorange="reversed") if switch else dict(), **self._kwargs)
        fig = go.Figure(traces, layout)
        return fig

    def scatter(self, x, y, z=None, identifier=None, switch=False, height=490):
        '''
        Plots a scatter chart.
        '''
        if switch: x, y = (y, x)
        if len(x) > 2000:
            np.random.seed(0)
            indices = np.random.choice(np.arange(len(x)), size=2000, replace=False)
            x, y, z = x.iloc[indices], y.iloc[indices], (z.iloc[indices] if z is not None else z)

        # Add some jitter on discrete values
        if x.nunique()<20: 
            x = x.astype(float)
            minmaxrange = (x.max()-x.min())/x.nunique()*0.6
            x += (np.random.uniform(size=len(x))-0.5)*minmaxrange
        if y.nunique()<20:
            y = y.astype(float)
            minmaxrange = (y.max()-y.min())/y.nunique()*0.6
            y += (np.random.uniform(size=len(y))-0.5)*minmaxrange

        # Set color params according to z
        if z is not None:
            if z.nunique()> 5: colorscale=[[0.0, self.colors[1]], [1.0, self.colors[0]]]
            else: colorscale=[[c, self.colors[i]] for i, c in enumerate(np.linspace(0, 1, z.nunique()))]
        else: colorscale = None
        
        # Build gigure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=x, y=y, name=y.name, mode='markers', hoverinfo='text', hovertext=x.index,
                marker=dict(color=z if z is not None else self.colors[0], opacity=0.5, size=6, colorscale=colorscale))),
        if identifier is not None and identifier in x.index:
            fig.add_trace(go.Scatter(x=[x.loc[identifier]], y=[y.loc[identifier]], mode='markers', 
                hoverinfo='text', hovertext=identifier, marker=dict(color='#888', opacity=.8, size=20))),
        fig.update_layout(autosize=True, showlegend=z is not None, height=height, xaxis=dict(title=x.name), 
            yaxis=dict(title=y.name), margin=dict(t=30, r=10, b=10, l=20), **self._kwargs)
        return fig

    def make_datatable(self, identifier, data, height='200px', page_size=20, left_align=None, **kwargs):
        '''
        Plots a data table.
        '''
        if left_align:
            style_cell_conditional=[{'if': {'column_id': c}, 'textAlign': 'left', 'fontWeight': '600'} for c in left_align]
            kwargs.update(style_cell_conditional=style_cell_conditional)
        background_header = '#333' if self.theme=='plotly_dark' else '#ddd'
        if page_size is not None: kwargs.update(dict(page_size=page_size))
        color = '#eee' if self.theme=='plotly_dark' else '#555'
        table = dt.DataTable(
            id=identifier, 
            data=data.loc[:, data.columns[:100]].to_dict('records'),
            columns=[{"name": i, "id": i} for i in data.columns[:100]],
            style_header={'font-family':'sans-serif', 'backgroundColor': background_header, 'color': color, 'fontWeight': 'bold'},
            style_data={'font-family':'sans-serif', 'backgroundColor': self._plotcolor, 'color': color},
            style_table={'height': height, 'overflowX': 'auto'},
            style_as_list_view=True,
            sort_action="native",
            **kwargs
        )
        return table

    def spaceholder(self):
        '''
        Plots a spaceholder.
        '''
        fig = go.Figure()
        fig.update_layout(showlegend=False, yaxis=dict(showticklabels=False), xaxis=dict(showticklabels=False), 
            autosize=True, height=490, margin=dict(l=40, r=10, b=50, t=20, pad=4), **self._kwargs)
        return (fig)



# Helper function
def get_binscores(preds, bins=50):
    cols = ['target', 'scores']
    to_bin_scores = preds['scores'].nunique()>bins
    preds['bins'] = pd.qcut(preds['scores'], bins, duplicates='drop') if to_bin_scores else preds['scores']
    groups = preds.reset_index().groupby(['dataset', 'bins'])
    binscores = groups[cols].mean().join(groups.target.count().rename('count')).unstack(0)
    totals = preds.reset_index().groupby(['dataset'])[cols].mean().unstack()
    if to_bin_scores:
        binscores.index = binscores.index.to_series().apply(
            lambda x: f'{x.left.round(10)}+' if type(x)!=float else x).astype(str)
    binscores.loc['Total'] = totals
    return binscores
