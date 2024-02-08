import os
import pandas as pd
import json
import logging
from tabulate import tabulate


class Transformer:
    '''
    This class collects methods for transforming data. 

    Arguments
    ---------
    name : String, default 'Transformer'
            Arbitrary name for the data object, that shows up in the LFD output,
            such as metrics, confusions, dashboards, etc.

    Attributes
    ----------
    Transformer.name : String
            Name of the transformer
    Transformer.params : Dict
            Dictionary of parameters to learn the transformer.
    '''    
    def __init__(self, name='Transformer'):
        self.params = {}
        self.name = name

    def learn(self, X):
        '''
        Learn a transformer on a data object.
        
        Arguments
        ---------
        X : lfd.Data
                Data used to learn the transformer on.
        '''
        logging.info(f'{X.df.shape} - {X.name} - learning {self.name}')

    def apply(self, X):
        '''
        Apply a transformer on a data object.
        
        Arguments
        ---------
        X : lfd.Data
                Data object to apply the transformer on.
        '''
        logging.info(f'{X.df.shape} - {X.name} - applying {self.name}')
    
    def save(self, storage):
        '''
        Save transformer.
        
        Arguments
        ---------
        storage : String
                Directory where the transformer should be saved.
        '''
        with open(os.path.join(storage, f'{self.name}.json'), 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    def load(self, storage):
        '''
        Load transformer.
        
        Arguments
        ---------
        storage : String
                Directory where the transformer should be loaded from.
        '''
        with open(os.path.join(storage, f'{self.name}.json'), 'r') as f:
            self.__dict__.update(json.load(f))
        return self

    def _summary(self):
        '''
        Returns a pandas Series with meta information.
        '''
        return pd.Series(dict(
            Name = self.name,
            Attributes = ", ".join([a for a in self.__dict__.keys() if a not in ('name', 'params')]),
            Parameters = ", ".join([f'{k}: {v}' for k, v in self.params.items()]),
        ))

    def __repr__(self):
        string = 'This is a TRANSFORM object.\n'
        string += tabulate(self._summary().to_frame(), tablefmt='simple_outline')
        return string