import logging
from .general import Transformer


class Standardizer(Transformer):

    def __init__(self):
        super().__init__('standardizer')
        self.elements = {}
        
    def learn(self, X, standardize=True, set_aside=None):
        '''
        This function learns which variables to standardize and which values to use.
        
        Arguments
        ---------
        X : lfd.Data
                Data object to learn transformation on
        standardize : Bool, default True
                Whether to standardize numerical variables
        set_aside : List[String], optional
                List of column names that are excluded from the transformation.
        '''
        super().learn(X)
        self.params = dict(standardize=standardize)

        if not standardize: return self
        set_aside = [] if set_aside is None else set_aside

        # Only continuous variables will be selected to be standardized
        variables = [v for v in X.df.select_dtypes(['number']).columns if v not in set_aside]
        
        # Calculate the mean and standard deviation for each column
        self.elements = {v : {'mean': float(X.df[v].mean()), 'std': float(X.df[v].std())} for v in variables}
        return self
    
    def apply(self, X, inplace=False):
        super().apply(X)
        data = X if inplace else X.copy()
        
        for nv in self.elements:
            data.df[nv] = (data.df[nv] - self.elements[nv]['mean']) / self.elements[nv]['std'] 
        if not inplace: return data
