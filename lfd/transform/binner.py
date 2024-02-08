import pandas as pd
import numpy as np
import logging
from .general import Transformer


class Binner(Transformer):
    
    def __init__(self):
        super().__init__('binner')
        self.bin_intervals = {}
        self.bin_values = {}
        self.transform = False
        
    def learn(self, X, n_bins=10, min_prop=0.05, transform=True, predef_cuts=None, set_aside=None):
        '''
        This function learns which variables to binn and which splitting points to use.

        Arguments
        ---------
        X : lfd.Data
                Data object to learn transformation on.
        n_bins : Integer, default 10
                Number of equal-sized bins to bin the variables into.
                This is the same for all continuous variables.
        min_prop : Float, default 0.05
                Minimum the proportion each bin must have in order to preserve the bin.
                If not successful for n_bins, n_bins-1 will be attempted and so on.
        transform : Bool, default True
                Whether to add or transform the initial non-binned variables.
        predef_cuts : Dict[String: List[Float]], optional
                Overwrite the determined cutting points according the selected method by 
                predefined values. These must be provided in a dictionary; the key is the 
                variable name, the value is a list with pre-defined cutting points.
        set_aside : List[String], optional
                List of column names that are excluded from the transformation.
        '''
        super().learn(X)
        self.params = dict(n_bins=n_bins, min_prop=min_prop, transform=transform, predef_cuts=predef_cuts)

        # Reset bins
        self.bin_intervals = {}
        if type(predef_cuts)==dict: self.bin_intervals.update(predef_cuts)
        self.bin_values = {}
        self.transform = transform
        if n_bins is None: return self
        n_bins = int(n_bins)
                        
        # Retrieve numerical variables to bin
        vars_bin = X.df.select_dtypes(['number']).columns
        if type(set_aside)==list: vars_bin = [c for c in vars_bin if c not in set_aside]
        if type(predef_cuts)==dict: vars_bin = [c for c in vars_bin if c not in predef_cuts.keys()]
        logging.debug(f'Attempting to bin following variables: {vars_bin}')
        
        # Learn intervals or values per variable
        for binvar in vars_bin:
            # Case with fewer unique values than bins
            if X.df[binvar].nunique() <= n_bins:
                logging.debug(f'Variable {binvar} has fewer unique values than {n_bins} desired bins! \
                    Creating a bin per value.')
                self.bin_values[binvar] = X.df[binvar].dropna().unique().tolist()
            else: # Case with more unique values than bins
                tot_len = X.df[binvar].notna().sum()
                for new_bins in np.arange(n_bins, 0, step=-1):
                    bins_series, cut_points = pd.qcut(
                        X.df[binvar], q=new_bins, retbins=True, duplicates='drop')
                    min_occ = bins_series.value_counts().min()
                    if(min_occ/tot_len >= min_prop):  # At new_bins=1, this condition will always be True
                        logging.debug(f'Variable {binvar} is binned into {new_bins} out of {n_bins} desired')
                        self.bin_intervals[binvar] = cut_points.tolist()
                        break
                if new_bins==1:
                    logging.debug(f'Could not bin {binvar}. There is only one category for {binvar}')
        return self
        
    def apply(self, X, inplace=False):
        super().apply(X)
        data = X if inplace else X.copy()

        for var, intervals in self.bin_intervals.items():
            labels = [f'b{i}' for i, _ in enumerate(intervals[:-1])]
            # _l{np.round(intervals[i], 2)}_r{np.round(intervals[i+1], 2)}
            data.df[var+'__binned'] = pd.cut(data.df[var], intervals, labels=labels, include_lowest=True)
            if self.transform: data.df.drop(var, inplace=True, axis=1)
            data.df[var+'__binned'].cat.set_categories(data.df[var+'__binned'].cat.categories.astype(str))
        for var, values in self.bin_values.items():
            data.df[var+'__binned'] = data.df[var].astype('category').cat.set_categories(values)
            if self.transform: data.df.drop(var, inplace=True, axis=1)
            data.df[var+'__binned'].cat.set_categories(data.df[var+'__binned'].cat.categories.astype(str))
        if not inplace: return data    
