import numpy as np

import plotly.offline as py
import plotly.graph_objs as go
from .general import Plotter


class PlotterGraph(Plotter):

    def __init__(self, colors=None, theme='dark'):
        super().__init__(colors=colors, theme=theme)
        self.layout = dict(margin = dict(pad=0, t=0, r=0, l=0, b=0), #font = dict(family='Palatino'), 
            template=self.theme, paper_bgcolor=self._plotcolor, plot_bgcolor=self._plotcolor, autosize=True, height=700)

    def plot_graph(self, graph, mode='mesh', showgrid=False, opacity=1, size=1, color=None, ambient=0.2, diffuse=0.8, fresnel= .1, specular= 1, roughness= .1):
        """
        Create a 3D space object.
        This object can be plotted, saved, projected, generated to HTML/JS, etc.

        Parameters:
        data (dataframe): dataframe with coordinates x, y, z and graphs i, j, k
        showgrid (boolean): whether to add axes to the space object
        
        Returns:
        Plotly figure: 3D space object
        """
        m = np.ceil(np.abs(graph.vertices).max().max())
        color = '#fff' if color is None else color
        
        if mode == 'point':
            data = go.Scatter3d(opacity=opacity, 
                x = graph.vertices.x, y = graph.vertices.y, z = graph.vertices.z,
                #colorscale=[[0, '#C30045'], [0.5, '#51626F'], [0.7, 'silver'], [1, '#252827']],
                mode = 'markers', marker = dict(color=color, opacity=opacity, size=size)
            )
        elif mode == 'mesh':
            data = go.Mesh3d(opacity=opacity, 
                x = graph.vertices.x, y = graph.vertices.y, z = graph.vertices.z,
                i = graph.faces.i, j = graph.faces.j, k = graph.faces.k,
                #colorscale=[[0, '#C30045'], [0.5, '#51626F'], [0.7, 'silver'], [1, '#252827']],
                lighting=dict(ambient=ambient, diffuse=diffuse, fresnel=fresnel, specular=specular, roughness=roughness),
                lightposition=dict(x=100, y=200, z=150), color=color,
            )
        
        axes = dict(
            range=[-m, m], showgrid=True, zeroline=True,
            gridcolor='#888', gridwidth=2,
            zerolinecolor='#969696', zerolinewidth=4,
            linecolor='#999999', linewidth=3,
            backgroundcolor=self._plotcolor
        ) if showgrid else dict(
            range=[-m,m], showgrid=False, zeroline=False,
            showticklabels=False, title='',
            backgroundcolor=self._plotcolor
        )        
            
        fig = go.Figure(data=[data])
        fig.update_layout(scene=dict(
                bgcolor=self._plotcolor,
                aspectratio=dict(x=1, y=1, z=1),
                xaxis=axes, yaxis=axes, zaxis=axes,
                camera = dict(eye=dict(x=.4, y=.4, z=.4))
            ), **self.layout
        )
        return fig

    def plot_spaceholder(self):
        '''
        Plots a spaceholder.
        '''
        fig = go.Figure()
        fig.update_layout(showlegend=False, yaxis=dict(showticklabels=False), 
                        xaxis=dict(showticklabels=False), **self.layout)
        return fig

