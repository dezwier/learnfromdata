import logging
from .general import Transformer


class Imputer(Transformer):
 
    def __init__(self):
        super().__init__('imputer')
        self.impute_values = {}
    
    def learn(self, X, default_cat='MISSING', default_cont='median', override=None, set_aside=None):
        '''
        This function learns how to impute data for each given variable, 
        and stores this information into an impute_values attribute.
        
        Arguments
        ---------
        X : lfd.Data
                Data object to learn transformation on
        default_cat : String
                Imputation value for all categorical variables.
        default_cont : String
                Integer or Float or 'mean' or 'median'. Imputation value for all continuous variables.
                If 'mean' or 'median', these will be calculated.
        override : Dictionary. 
                Set a particular imputation value per feature. If this is filled in for a feature,
                this will override the default parameters.
        set_aside : List[String], optional
                List of column names that are excluded from the transformation.
        '''
        super().learn(X)
        self.params = dict(default_cat=default_cat, default_cont=default_cont, override=override)

        self.impute_values = {}
        set_aside = set_aside if set_aside else []

        # Retrieve categorical and numerical variables, asserting there are no others
        cat_vars = [c for c in X.df.select_dtypes(['object', 'category']).columns if c not in set_aside]
        cont_vars = [c for c in X.df.select_dtypes(['number']).columns if c not in set_aside]
        logging.debug(f'Storing imputation for {len(cat_vars)} categorical and {len(cont_vars)} continuous variables.')        
        
        # Update impute_values attribute with appropriate values
        for var in cat_vars:
            if X.df[var].isnull().sum()>0: self.impute_values.update({var: default_cat})
        if default_cont is not None:
            self.impute_values.update({v: float(X.df[v].mean()) if default_cont=='mean' \
                                       else float(X.df[v].median()) if default_cont=='median' \
                                       else float(default_cont) for v in cont_vars})
        
        # Override any given imputation values
        if type(override)==dict: self.impute_values.update(override)
        return self
    
    def apply(self, X, inplace=False):
        super().apply(X)
        data = X if inplace else X.copy()

        # Add missing value to categories if not present
        for var in data.df.select_dtypes('category').columns:
            if var in self.impute_values.keys() and self.impute_values[var] not in data.df[var].cat.categories:
                data.df[var] = data.df[var].cat.add_categories(self.impute_values[var])
        
        # Actual missing value imputation
        data.df.fillna(self.impute_values, inplace=True)
        return data
