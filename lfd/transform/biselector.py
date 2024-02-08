import pandas as pd
import numpy as np
import logging
from .general import Transformer


class BiSelector(Transformer):

    def __init__(self):
        super().__init__('biselector')
        self.columns = []
        
    def learn(self, X, threshold=0.8, target=None, feature_select=None, set_aside=None):
        '''
        This function learns which variables should be selected.
        
        Arguments
        ---------
        X : lfd.Data
                Data object to learn transformation on.
        threshold : Float, default 0.8
                Maximum allowed correlation (Cramer's V) between 2 variables.
        target : String, optional
                Target variable. If 2 variables correlate too much according to threshold, the one with
                the highest target correlation is preserved. If not given, the first one in X is retained.
        feature_select : List[Strings], optional 
                Similarly to variable_select in the uniselector, this is a way to hard-select
                features, albeit in this stage - after potential binning, imputing and encoding. 
                If given, the other parameters are ignored.
        set_aside : List[String], optional
                List of column names that are excluded from the transformation.
        '''
        super().learn(X)
        self.params = dict(threshold=threshold, target=target, feature_select=feature_select)

        # Hard select variables if given.
        if feature_select is not None:
            assert type(feature_select)==list, 'feature_select is not a list'
            for v in feature_select:
                if v in X.df: self.columns.append(v)
                else: logging.debug(f'{v} not present in data')
            if set_aside: self.columns += set_aside
            return self

        # Shortcut of threshold is 1: all features are selected
        if threshold==1:
            self.columns = list(X.df.columns)
            return self
        self.columns = []
        set_aside = [] if set_aside is None else set_aside

        # Take sample and calculate pairwise and target correlations
        variables = [x for x in X.df if x not in set_aside or x == target]
        sample = X.df.sample(min([len(X.df), 10000]), random_state=0, replace=True)[variables]
        sample = pd.concat((
            sample.select_dtypes(('category', 'object')),
            sample.select_dtypes('number').apply(
                lambda x: pd.qcut(x, q=10, duplicates='drop') if x.nunique() > 10 else x)
        ), axis=1).apply(lambda column: np.unique(column.astype(str).fillna('<m>'), return_inverse=True)[1])
        logging.debug('Sample created for correlation matrix')
        
        # Calculate cramers V in between features and features vs target
        corr_matrix = sample.corr(method=calculate_cramersv)
        if target is not None:
            corr_matrix.drop(target, inplace=True)
            corr_target = corr_matrix.pop(target)
            features_sorted = corr_target.sort_values(ascending=False).index
        else: features_sorted = sample.columns
        
        # Loop through candidates and potentialy append or replace list of kept vars
        kept_vars = []
        for candidate in features_sorted:
            to_keep = True
            for i, kept_var in enumerate(kept_vars):
                if corr_matrix.loc[candidate, kept_var] > threshold:
                    logging.debug(f'Candidate {candidate} correlates too much with kept var {kept_var}.')
                    if target is not None and corr_target[candidate] > corr_target[kept_var]:
                        if not kept_var.startswith('in_'):
                            logging.debug(f'Candidate {candidate} correlates more with target, so replacing {kept_var}.')
                            kept_vars[i] = candidate
                    to_keep = False
                    break
            if to_keep: 
                logging.debug(f'Candidate {candidate} added to kept variables.')
                kept_vars.append(candidate)
        self.columns = kept_vars + set_aside
        return self

    def apply(self, X, inplace=False):
        super().apply(X)
        data = X if inplace else X.copy()
        data.df = data.df[[c for c in self.columns if c in X.df]]
        return data


def calculate_cramersv(var1, var2):
    '''
    Helper method to calculate Cramer's V statistic for categorical variables
    '''
    var1, var2 = var1.astype(int), var2.astype(int)
    cramers_v = 0  
    n = var1.max() + 1
    m = var2.max() + 1
    
    # Calculating cross table efficiently
    obs_count = np.bincount(n*var2+var1, minlength=m*n).reshape(m, n)
    
    s1 = obs_count.sum(axis=1).reshape(m, 1)
    s0 = obs_count.sum(axis=0).reshape(1, n)
    ss = obs_count.sum()
    exp_count = (s1 * s0 * 1.0) / ss
    if not np.all(exp_count.sum(axis=1)):
        logging.warning('Constant variable in Cramers\' V calculation.')
        return 0
    chi2 = ((obs_count - exp_count) ** 2 / exp_count).sum()
    if m > 1 and n > 1:
        cramers_v = np.sqrt(chi2 / ss / min(m - 1, n - 1))
    return cramers_v