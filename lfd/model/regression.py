import pandas as pd
from .general import Model


class Regression(Model):
    
    def __init__(self, name='Regression'):
        super().__init__(name=name)
        
    def learn(self, data, target, mode, hyper_params={}, set_aside=None, seed=0):
        super().learn(data, target, mode, hyper_params, set_aside, seed)

        # Initialize a classifier
        hyper_params['l1_ratio'] = hyper_params.get('hyper_params', 0)
        if self.mode in ('binaryclass', 'multiclass'):
            from sklearn.linear_model import LogisticRegression
            self.clf = LogisticRegression(penalty='elasticnet', solver='saga', 
                                          random_state=seed, **hyper_params)
        elif self.mode=='linear':   
            from sklearn.linear_model import ElasticNet 
            self.clf = ElasticNet(random_state=seed, **hyper_params)
        self.clf.fit(data.df[self.features].astype(float), data.df[self.target])

        # Show feature importance
        coef = self.clf.coef_[0] if len(self.clf.coef_.shape)==2 else self.clf.coef_
        self.feature_imp = pd.Series(coef, index=self.features).sort_values(ascending=False).rename('importance')
        return self
    
    def _predict_scores(self, data):
        super()._predict_scores()
        if self.mode=='binaryclass':
            scores = self.clf.predict_proba(data.df[self.features].astype(float))[:, 1]
        elif self.mode=='multiclass':    
            scores = self.clf.predict_proba(data.df[self.features].astype(float))
        elif self.mode=='linear':    
            scores = self.clf.predict(data.df[self.features].astype(float))
        return scores
