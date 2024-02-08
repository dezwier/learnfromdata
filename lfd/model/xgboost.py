import pandas as pd
import numpy as np
from .general import Model


class Xgboost(Model): 
    
    def __init__(self, name='Xgboost'):
        super().__init__(name=name)
        
    def learn(self, data, target, mode, hyper_params={}, set_aside=None, seed=0):
        super().learn(data, target, mode, hyper_params, set_aside, seed)

        # Make sure hyper params are integers where needed
        for h in ['n_estimators', 'gamma', 'early_stopping_rounds']:
            if hyper_params.get(h): hyper_params[h] = int(hyper_params[h])
        
        # Initialize and train a classifier
        if self.mode in ('binaryclass', 'multiclass'):
            from xgboost import XGBClassifier
            self.clf = XGBClassifier(random_state=seed, **hyper_params)
            self.clf.fit(data.df[self.features].astype(float), data.df[self.target], eval_metric='logloss')
            if self.mode=='multiclass': self.categories = self.clf.classes_
        elif self.mode=='linear':
            from xgboost import XGBRegressor
            self.clf = XGBRegressor(random_state=seed, **hyper_params) 
            self.clf.fit(data.df[self.features].astype(float), data.df[self.target])

        # Show feature importance
        self.feature_imp = pd.Series(
            self.clf.feature_importances_, index=self.features).sort_values(ascending=False).rename('importance')
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
    
    def explain(self, data):
        super().explain(data)
        import shap
        X = data.df[self.features].astype(float)
        explainer = shap.Explainer(self.clf, feature_names=self.features)
        self.shapvalues = explainer.shap_values(X, check_additivity=True)
        if len(np.array(self.shapvalues).shape)>2: self.shapvalues = self.shapvalues[0]

        self.shapvalues = pd.DataFrame(self.shapvalues, columns=self.features, index=X.index)
        #self.shap_expected = float(explainer.expected_value)

        # Sort columns of shap values based on global importance
        self.shapvalues = self.shapvalues[
            self.shapvalues.abs().sum().sort_values(ascending=False).index]
