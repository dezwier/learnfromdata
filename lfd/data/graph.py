import pandas as pd
import logging
import os
import shutil
import json
from datetime import datetime
from lfd.utils import get_memory


class Graph:

    def __init__(self, path=None, name=''):

        self.name = path.split('/')[-1].split('.')[0] if path is not None else name
        self.type = 'graph'
        
        self.vertices = pd.DataFrame()
        self.faces = pd.DataFrame()
        if path is not None and path.endswith('.csv'):
            self.import_csv(path)
        elif path is not None and path.endswith('.obj'):
            self.import_graph(path)


    def __repr__(self):
        print(f'This is a graph object, with {len(self.vertices)} vertices and {len(self.faces)} faces.')
        return ''
    
    def import_csv(self, path):
        """
        Create a vertices and faces from a .csv data file.

        Parameters:
        path (string): path to .csv data
        
        Returns:
        vertices: Numpy array graph vertices
        faces: Numpy array of graph faces
        """
        graph_data = pd.read_csv(path)
        self.vertices = graph_data[["x","y","z"]].dropna()
        self.faces = graph_data[["i","j","k"]].dropna()

    def import_graph(self, path):
        """
        Create a vertices and faces from an .obj data file.

        Parameters:
        path (string): file path to .obj data
        
        Returns:
        vertices: Numpy array of graph vertices
        faces: Numpy array of graph faces
        """
        obj_data=open(path,'rb').read().decode('utf-8')
        
        vertices = []
        faces = []
        lines = obj_data.splitlines()   
    
        for line in lines:
            slist = line.split()
            if slist:
                if slist[0] == 'v':
                    vertex = list(map(float, slist[1:]))
                    vertices.append(vertex)
                elif slist[0] == 'f':
                    face = []
                    for k in range(1, len(slist)):
                        face.append([int(s) for s in slist[k].replace('//','/').split('/')])
                    
                    if len(face) > 3: # triangulate the n-polyonal face, n>3
                        faces.extend([[face[0][0]-1, face[k][0]-1, face[k+1][0]-1] for k in range(1, len(face)-1)])
                    else:    
                        faces.append([face[j][0]-1 for j in range(len(face))])
                else: pass
        
        self.vertices = pd.DataFrame(vertices, columns=["x","y","z"]).dropna()
        self.faces = pd.DataFrame(faces, columns=["i","j","k"]).dropna()
        

    def save(self, directory, only_meta=False):
        '''
        Save dataframe.
        '''
        path = os.path.join(directory, self.name)
        logging.info(f'Data - Saving to {path}')

        # Remove existing and create new directory
        if not only_meta:
            if os.path.exists(path): shutil.rmtree(path)
            os.mkdir(path)

        # Store data
        if not only_meta: 
            self.vertices.to_csv(os.path.join(path, 'vertices.csv'), index=False)
            self.faces.to_csv(os.path.join(path, 'faces.csv'), index=False)
        with open(os.path.join(path, f'meta.json'), 'w') as f:
            json.dump({
                'name': self.name,
                'type': self.type,
                '#vertices': len(self.vertices),
                '#faces': len(self.faces),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'memory': get_memory(self)
            }, f, indent=4)
    
    @classmethod
    def load(cls, directory):
        '''
        Load dataframe.
        '''
        logging.info(f'Data - Loading from directory {directory}')
        data = Graph()
        with open(os.path.join(directory, 'meta.json'), 'r') as f:
            json_data = json.load(f)
            data.name, data.type = json_data['name'], json_data['type']

        data.vertices = pd.read_csv(os.path.join(directory, 'vertices.csv'))
        data.faces = pd.read_csv(os.path.join(directory, 'faces.csv'))

        logging.info(f'Data - Loaded.')
        return data
        