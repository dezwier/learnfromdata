from .general import Model


class DecisionTree(Model):
    
    def __init__(self, name='Decisiontree'):
        super().__init__(name=name)

    def learn(self, data, target, mode, hyper_params={}, set_aside=None, seed=0):
        super().learn(data, target, mode, hyper_params, set_aside, seed)
        
        # Initialize a classifier
        if self.mode in ('binaryclass', 'multiclass'):
            from sklearn.tree import DecisionTreeClassifier
            self.clf = DecisionTreeClassifier(random_state=seed, **hyper_params)
        elif self.mode=='linear':
            from sklearn.tree import DecisionTreeRegressor
            self.clf = DecisionTreeRegressor(random_state=seed, **hyper_params)
        self.clf.fit(data.df[self.features].astype(float), data.df[self.target])

        # Show feature importance
        self.feature_imp = None
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
