import logging
from .general import Transformer


class UniSelector(Transformer):

    def __init__(self):
        super().__init__('uniselector')
        self.columns = []

    def learn(self, X, min_occ=0.01, max_occ=0.99, include_missings=True, variable_select=None, set_aside=None):
        '''
        This function learns which variables should be selected.
        
        Arguments
        ---------
        X : lfd.Data
                Data object to learn transformation on
        min_occ : Float, default 0.01
                Minimum occurrence that the highest occurring value must have. 
                If it doesn't reach this, the variable is considered too variable and thrown out.
        max_occ : Float, default 0.99
                Maximum occurrence that the highest occurring value can have.
                If it exceeds this, the variable is considered too constant and thrown out.
        include_missings : Bool, default True
                Whether to include missing values as a value to consider in the variation calculation.
        variable_select : List[Strings], optional 
                This is a way to hard-select variables. If given, the other parameters are ignored.
        set_aside : List[String], optional
                List of column names that are excluded from the transformation.
        '''
        super().learn(X)
        self.params = dict(min_occ=min_occ, max_occ=max_occ, include_missings=include_missings, variable_select=variable_select)
        self.columns = []
        
        # Hard select variables if given.
        if variable_select:
            assert type(variable_select)==list, 'variable_select is not a list'
            self.columns = list(variable_select)
            if set_aside: self.columns += set_aside
            return self
        
        # If hard select is not given, perform data-wase uni selection
        constant_vars = []
        low_variation_vars = []
        high_variation_vars = []
        
        # Loop through variables and decide whether to keep
        for var in X.df.columns:
            if type(set_aside)==list and var in set_aside:
                self.columns.append(var)
                continue
            shares = X.df[var].value_counts(normalize=True, dropna=not include_missings)
            if len(shares)<=1:  # Variables is constant
                constant_vars.append(var)
            elif shares.iloc[0] > max_occ:  # Var has low variation
                low_variation_vars.append(var)
            elif X.df[var].dtype.name in ('object', 'category') and shares.iloc[0] < min_occ: # Var has high variation
                high_variation_vars.append(var)
            else: self.columns.append(var)
            
        logging.debug(f'Constant variables: {len(constant_vars)}, {constant_vars}')
        logging.debug(f'Low variation variables {len(low_variation_vars)}, {low_variation_vars}')
        logging.debug(f'High variation variables: {len(high_variation_vars)}, {high_variation_vars}')
        return self

    def apply(self, X, inplace=False):
        super().apply(X)
        data = X if inplace else X.copy()
        data.df = data.df[[c for c in self.columns if c in X.df]]
        return data
