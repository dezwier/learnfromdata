import pandas as pd
import numpy as np
#from sklearn.cluster import KMeans
from .general import Model


class GaussianMixture(Model):

    def __init__(self, name='GM'):
        super().__init__(name=name)
        
    def learn(self, data, target=None, mode='clustering', hyper_params={}, set_aside=None, seed=0):
        super().learn(data, target, mode, hyper_params, set_aside, seed)
        
        # Initialize and train a classifier
        from sklearn.mixture import GaussianMixture as GM
        self.clf = GM(**hyper_params, random_state=seed)
        scores = self.clf.fit_predict(data.df[self.features].astype(float))
        self.feature_imp = calculate_cluster_drivers(data.df[self.features], scores)
        return self
    
    def _predict_scores(self, data):
        super()._predict_scores()
        scores = self.clf.predict_proba(data.df[self.features])
        return scores


def calculate_cluster_drivers(df, predictions):
    n_clusters = pd.Series(predictions).sort_values().unique()

    # Calculate feature importance
    np.random.seed(0)
    size = min(10000, len(df))
    sample_indices = np.random.randint(len(df), size=size)
    sample = df.iloc[sample_indices].astype(float)
    sample = sample.apply(lambda x: pd.qcut(x, q=10, duplicates='drop') if x.nunique() > 10 else x).apply(
                lambda column: np.unique(column.astype(str).fillna('<m>'), return_inverse=True)[1])
    feature_imp = pd.DataFrame()
    scores = pd.Series(predictions[sample_indices], index=sample.index)
    feature_imp['cluster_all'] = sample.corrwith(scores, method=calculate_cramersv)
    for n in n_clusters:
        scores = pd.Series((predictions[sample_indices]==n).astype(np.int8), index=sample.index)
        feature_imp[f'cluster_{n}'] = sample.corrwith(scores, method=calculate_cramersv)

    # Sort columns of shap values based on global importance
    feature_imp = feature_imp.loc[
        feature_imp.cluster_all.abs().sort_values(ascending=False).index]
    return feature_imp


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
        return 0
    chi2 = ((obs_count - exp_count) ** 2 / exp_count).sum()
    if m > 1 and n > 1:
        cramers_v = np.sqrt(chi2 / ss / min(m - 1, n - 1))
    return cramers_v