import pandas as pd
import numpy as np
import logging
from .general import Transformer


class Encoder(Transformer):

    def __init__(self):
        super().__init__('encoder')
        self.encode_values = {}

    def learn(self, X, min_occ=0.01, override=None, include_others=True, method='onehot', target=None, set_aside=None):
        '''
        This function learns which variables should be encoded to which dummies.
        
        Arguments
        ---------
        X : lfd.Data
                Data object to learn transformation on.
        min_occ : Float, default 0.01
                Should be between 0 and 1. Minimum occurrence or share each value must have to 
                receive it's own indicator.
        override : Dictionary[String: Float]
                Set a particular min_occ per variable. If this is filled in for a feature, this 
                will override the default min_occ.
        include_others : Bool, default True
                Whether to group all values occurring less than min_occ in a single indicator.
        method : String, default 'onehot'
                Should be in in ('onehot', 'target'). Whether to group all values occurring less 
                than min_occ in a single indicator.
        target : String, optional
                In case of target encoding, identify which column is used as target.
        set_aside : List[String], optional
                List of column names that are excluded from the transformation.
        '''
        super().learn(X)
        self.params = dict(min_occ=min_occ, override=override, include_others=include_others, method=method, target=target)
        
        # Assert that sep string does not occur in columns of dataset
        cols = pd.Series(X.df.columns)
        assert cols.str.contains('__').sum()==0, 'Seperator string "__" already occurs in column names!'
        assert method in ('onehot', 'target'), "Method should be in ('onehot' or 'target')"
        if method =='target': assert target is not None, "Target should be given if method is 'target'"
        
        self.encode_values = {}
        if min_occ is None: return self
        
        # Per variable we will be storing the values that occur more than min_occ
        for var in X.df.select_dtypes(['category', 'object', 'bool']):
            if set_aside is not None and var in set_aside: continue
            shares = X.df[var].value_counts(normalize=True)
            
            # In case of 2 values, choose one to one-hot-encode based on value
            if method == 'onehot' and X.df[var].nunique() == 2:
                if '1' in shares.index: shares = shares.loc[['1']]
                elif True in shares.index: shares = shares[shares.index]
                elif 1 in shares.index: shares = shares.loc[[1]]
                else: shares = shares.iloc[[1]]  # Less occuring value is 'trigger'
                
            # Override when needed, and remove low occuring values
            min_occ_to_use = override[var] if override and var in override.keys() else min_occ 
            encode_values = list(shares.index[shares>min_occ_to_use])
            
            # Add other category if requested and other values are present
            if include_others and len(shares) > len(encode_values):
                encode_values.append('OTHER')

            if method=='target':
                x = set_categories(X.df[var].copy(), encode_values)
                encode_values = X.df[target].groupby(x).mean().to_dict()
            self.encode_values[var] = encode_values

        return self

    def apply(self, X, inplace=False):
        super().apply(X)
        # Create new data object which will contain all dummies
        data = X if inplace else X.copy()
            
        # Loop through each variable to dummy
        for i, (var, values) in enumerate(self.encode_values.items()):
            
            # Set categories. Values not part of categories become missing
            data.df[var] = set_categories(data.df[var], values)

            # One hot encoding
            if isinstance(values, list):
                # Retrieve dummy variables, make them categorical and set level type as string
                dummies = pd.get_dummies(data.df[[var]], prefix_sep='__')
                dummies = dummies.astype('category') if dummies.shape[1] > 0 else dummies
                for d in dummies.columns: 
                    dummies[d].cat.set_categories(dummies[d].cat.categories.astype(str))
                    
                # Remove brackets and commas from column names (libraries like xgboost don't like them)
                dummies.columns = [c.replace('(', '').replace(')', '').replace(']', '').replace(
                    ' ', '').replace('<', '_st_').replace('>', '_gt_') for c in dummies.columns]

                # Append dummies and drop original variable
                data.df.drop(var, axis=1, inplace=True)
                data.df = pd.concat([data.df, dummies], axis=1)
                logging.debug(f'{i}: {var} encoded into {dummies.shape[1]} dummies.')

            # Target encoding
            elif isinstance(values, dict):
                data.df[var] = pd.to_numeric(data.df[var].replace(values), downcast='float')
            
        if not inplace: return data


def set_categories(x, categories):
    if x.dtype.name=='category':
        x = x.cat.set_categories(categories)
    else:  # dtype is string or bool
        x[~x.isin(categories)] = np.nan
        x = x.astype('category').cat.set_categories(categories)
    if 'OTHER' in categories: 
        x.fillna('OTHER', inplace=True)
    return x
