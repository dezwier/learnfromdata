import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from .general import Plotter, get_binscores
from .board import run_app


class PlotterModel(Plotter):

    def __init__(self, colors='belfius', theme='light'):
        super().__init__(colors=colors, theme=theme)

    def run_app(self, directory='.', host="0.0.0.0", port=9063, debug=True):
        '''
        Run Dash dashboard that visualizes experiments made with lfd.
        If running from a notebook, kill the kernal to close the connection.
        If running in the background, it can be closed in a terminal by 
        running "netstat -ltnp | grep <port>" and "kill -9 <PID number>".
        
        Arguments
        ---------
        directory : String, default './'
                Path to the directory where experiments are stored.
        host : String, default '0.0.0.0'
                Host where the app is served. Defaults to loacal host.
        port : Integer, default '9063'
                Port where the app is served.
        debug : Bool, default True
                Whether to run the app in debug mode.
        '''
        run_app(directory, host, port, debug)

    def confusion_heatmaps(self, confusion):
        '''
        Displays a modelling tailored confusion heatmap.
        
        Arguments
        ---------
        confusion : pandas.DataFrame
                Confusion matrix in same format of Model.confusion. 
        '''
        pred_name = confusion.columns.levels[0][0]
        confusion = confusion.sort_index()[pred_name]
        low_bins = confusion.shape[1] < 10

        datasets = list(confusion.index.levels[0])
        datasets.reverse()
        n_data = len(datasets)
        fig = make_subplots(rows=len(datasets), cols=1)
        annotations = []
        for d, dataset in enumerate(datasets):
            conf = confusion.loc[dataset].copy()
            conf.columns = [f'P{c}' if type(c)!=str else c for c in conf.columns]
            conf.index = [f'L{i}' if type(i)!=str else i for i in conf.index]
            conf.loc['Total'] = 0
            conf['Total'] = 0

            # Heatmap traces
            fig.add_trace(go.Heatmap(
                x=conf.columns, y=conf.index, z=conf.values, hoverinfo='skip' if low_bins else None,
                showscale=False, xaxis=f"x{d+1}", yaxis=f"y{d+1}", zmin=0,
                zmax=conf.max(axis=1).max(),
                colorscale=[[0, self._plotcolor], [1, self.colors[0]]]), col=1, row=d+1)
            conf.loc['Total'] = conf.sum()
            conf['Total'] = conf.sum(axis=1)

            if low_bins:
                # Add counts as annotations on heatmaps
                for n in conf.index:
                    for m in conf.columns:
                        annotations.append(dict(text=str(conf.loc[n, m]), showarrow=False,
                                                x=m, y=n, xref=f"x{d+1}", yref=f"y{d+1}"))
            # Update layout locations
            fig.update_layout({f'xaxis{d+1}': dict(title=dataset, dtick=1 if low_bins else None, side='top')})
            fig.update_layout({f'yaxis{d+1}': dict(title='Actual', dtick=1 if low_bins else None, anchor=f"x{d+1}", 
                autorange='reversed', domain=[d * 1/n_data, (d+1)*1/n_data-0.2/n_data])})
        width = min(conf.shape[1]*40+130, 620)
        height = min(conf.shape[0]*40+120, 540)
        fig.update_layout(annotations=annotations, width=width, height=height*len(datasets),
            margin=dict(l=80, r=10, b=20, t=10), **self._kwargs)
        return fig

    def lift_curve(self, preds, dataset='Test', bins=50):
        '''
        Displays a modelling tailored lift curve.
        
        Arguments
        ---------
        preds : pandas.DataFrame
                Prediction dataframe in same format of Model.predictions.df.
                Contains targets and predictions to calculate lift curve on the fly. 
        '''
        binscores = get_binscores(preds, bins)
        targets = binscores.target
        preds = binscores.scores.drop('Total')
        counts = binscores['count'].drop('Total')
        totals = targets[[dataset]]
        targets.drop('Total', inplace=True)
        
        fig = make_subplots(rows=1, cols=1, subplot_titles=None, specs=[[{"secondary_y": True}]])
        if len(dataset)>0:
            d = dataset
            fig.add_trace(go.Scatter(
                x=targets.index, y=targets[d], marker=dict(color=self.colors[0]), name=d+' Actual'))
            fig.add_trace(go.Scatter(
                x=preds.index, y=preds[d], marker=dict(color=self.colors[1]), name=d+' Predictions'))
            fig.add_trace(go.Bar(hoverinfo='name+y',
                x=counts.index, y=counts[d], marker=dict(color="#bbb"), name=d+' Count', opacity=0.3), secondary_y=True)
            fig.update_layout(yaxis2=dict(title="Count", showgrid=False, range=[0, counts[d].max()*4]),)
                
        fig.update_layout(autosize=True, height=541, showlegend=True, 
                        margin=dict(t=30, r=40, b=50, l=20), yaxis=dict(title="Prevalence"),
                        legend=dict(x=0.05, y=0.9), xaxis=dict(title="Prediction Bins", dtick=1), **self._kwargs)
        return fig

    def shap_observation(self, shapsample, datasample):
        '''
        Displays a modelling tailored shap observation chart.
        
        Arguments
        ---------
        shapsample : pandas.DataFrame
                Shapvalues dataframe in same format of Model.shapvalues.df.
                Ideally use a sample for quick rendering. 
        datasample : pandas.DataFrame
                Input dataframe in same format of Model.data.df.
                Ideally use a sample for quick rendering. 
        '''
        max_letters = min(25, shapsample.index.to_series().apply(len).max())
        trace = go.Bar(x=shapsample, y=shapsample.index.str.slice(0, 25), orientation='h', 
                    hoverinfo='x+y+text', hovertext=datasample, text=datasample, 
                    textposition='auto', textangle=0,
                    marker=dict(color=(shapsample>0).astype(float), colorscale=[[0.0, self.colors[1]], [1.0, self.colors[0]]]))
        layout = go.Layout(autosize=True, showlegend=False, yaxis=dict(autorange="reversed", dtick=1),
                        xaxis=dict(title="Impact on Prediction"), height=len(shapsample)*25+60,
                        margin=dict(t=30, l=max_letters*7, r=30, b=10, pad=10), **self._kwargs)
        fig = go.Figure([trace], layout)
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        return fig