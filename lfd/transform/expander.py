import logging
import pandas as pd
import numpy as np
from itertools import combinations
from .general import Transformer


class Expander(Transformer):

    def __init__(self):
        super().__init__('expander')
        self.expand_behaviour = {'interactions': False, 'log': False}
        self.itx_pairs = []
    
    def learn(self, X, interactions=True, log=False, set_aside=None):
        '''
        This function learns how to add extra features to a dataset.
        
        Arguments
        ---------
        X : lfd.Data
                Data object to learn transformation on
        interactions : Bool
                Whether to include interactions, i.e. the multiplication of all pairwise features. 
                Can be helpful for regression models to include effects of variables together.
        log : Bool, NOT YET IMPLEMENTED
                Whether to include logs of continuous variables.
                Can be helpful for typically linear models to introduce non-linearity in the features.
        set_aside : List[String], optional
                List of column names that are excluded from the transformation.
        '''
        super().learn(X)
        self.params = dict(interactions=interactions, log=log)

        self.expand_behaviour['interactions'] = interactions
        self.expand_behaviour['log'] = log
        
        if self.expand_behaviour['interactions']:
            self.itx_pairs = []
            con_vars = [x for x in X.df.select_dtypes('number').columns if x not in set_aside]
            cat_vars = []
            for col in X.df.select_dtypes('category').columns:
                if col not in set_aside and np.all([x.isdigit() for x in X.df[col].cat.categories]).all():
                    cat_vars.append(col)
            columns = con_vars+cat_vars
            base_vars = pd.Series(columns).str.split('__').apply(lambda c: c[0]).unique()
            base_pairs = [x for x in combinations(base_vars, 2)]

            for base_var1, base_var2 in base_pairs:
                features1 = X.df.columns[X.df.columns.str.startswith(base_var1)][:5]
                features2 = X.df.columns[X.df.columns.str.startswith(base_var2)][:5]
                feature_pairs = [[f1, f2] for f1 in features1 for f2 in features2]
                self.itx_pairs += feature_pairs
        return self
    
    def apply(self, X, inplace=False):
        super().apply(X)
        data = X if inplace else X.copy()
        
        if self.expand_behaviour['interactions']:
            features = np.unique(np.array([list(x) for x in self.itx_pairs]).flatten())
            temp = data.df[features].astype(float)                        
            logging.info(f'Number of interactions to calculate: {len(self.itx_pairs)}')
            for i, (feature1, feature2) in enumerate(self.itx_pairs):
                itx = temp[feature1].mul(temp[feature2]).astype('category')
                itx.cat.set_categories(itx.cat.categories.astype(str))
                data.df[f'ITX_{feature1}_x_{feature2}'] = itx
                if i!=0 and (i+1)%1000==0: logging.info(f'Interactions calculated: {i+1}')
            
        return data    
