import numpy as np
import pandas as pd
from .general import Model


class Isotonic(Model):

    def __init__(self, name='Isotonic'):
        super().__init__(name=name)
        
    def learn(self, data, target, mode=None, hyper_params={}, set_aside=None, seed=0):
        super().learn(data, target, mode, hyper_params, set_aside, seed)

        assert len(self.features)==1, f'Can only use mapping on 1 features, but {len(self.features)} were given.'
        method = hyper_params.get('method', 'quantile')
        
        if method == 'quantile':
            n_quantiles = 1000

            # Clf is a mapping table with quantiles on the scores and the target. This acts as the 'model'
            self.clf = pd.DataFrame(dict(
                scores = data.df[self.features[0]], target = data.df[self.target]
            )).quantile(np.arange(0, 1, 1/n_quantiles))
            self.clf.index = pd.Series(self.clf.index).apply(lambda x: round(x, len(str(n_quantiles))-1)) # Rounding issue

            # This line provides an upper cap so we always find a mapping
            self.clf.loc[2.00] = dict(scores = 1e10, target = self.clf.iloc[-1].target)  
            self.clf.target = self.clf.target.astype(data.df[self.target].dtype)
        
        elif method == 'regression':
            del hyper_params['method']
            from sklearn.isotonic import IsotonicRegression
            self.clf = IsotonicRegression(**hyper_params)
            self.clf.fit(data.df[self.features].astype(float), data.df[self.target])
        return self
    
    def _predict_scores(self, data):
        super()._predict_scores()
        scores = data.df[self.features[0]].astype(float)
        if isinstance(self.clf, pd.DataFrame):
            # Quantile method: look up first row in quantile table with a higher score, and take that target quentile
            new_scores = self.clf.target.iloc[pd.Series(np.searchsorted(self.clf.scores, scores))].values
        else:
            # Regression method: predict with classifier
            new_scores = self.clf.predict(scores)
        return new_scores
