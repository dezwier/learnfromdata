from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


from .general import Plotter


class PlotterVisual(Plotter):

    def __init__(self, colors=None, theme='dark'):
        super().__init__(colors=colors, theme=theme)
        self.layout = dict(margin = dict(pad=0, t=50, r=0, l=0, b=0), #font = dict(family='Palatino'), 
            template=self.theme, paper_bgcolor=self._plotcolor, plot_bgcolor=self._plotcolor, autosize=True, height=550)

    def plot_images(self, visual, indices):
        '''
        Plots images for a visual dataset.
        '''
        nx, ny = 5, 2
        titles = [f"{i} - {visual.df.label[i]}" for i in indices]
        fig = make_subplots(ny, nx, horizontal_spacing=0.015, vertical_spacing=0.10, subplot_titles=titles)
        for i, img in enumerate(visual.tensor):
            fig.add_trace(go.Image(z=img), i//nx+1, i%nx+1)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(**self.layout)
        fig.update_layout(height=600)
        fig.update_annotations(font=dict(size=11))
        return fig

    def plot_spaceholder(self):
        '''
        Plots a spaceholder.
        '''
        fig = go.Figure()
        fig.update_layout(showlegend=False, yaxis=dict(showticklabels=False), 
                        xaxis=dict(showticklabels=False), **self.layout)
        return fig

