import numpy as np
import pandas as pd
import os
import pickle
import json
import shutil
import logging
from tabulate import tabulate
from ..data.general import Data


class Model:
    '''
    This class collects methods for learning, applying, evaluating and explaining a model. 
    
    Arguments
    ---------
    name : String, default 'Model'
            Arbitrary name for the model object, that shows up in many of the LFD output,
            such as metrics, confusions, dashboards, etc.

    Attributes
    ----------
    Model.name : String
            Name of the model object, that shows up in many of the LFD output,
            such as metrics, confusions, dashboards, etc.
    Model.params : Dict
            Dictionary of hyperparameters used in the underlying algorithm.
    Model.clf : Any
            Underlying modelling object that is different per subclass. E.g. sklearn.linear_model.LogisticRegression
    Model.features : List[String]
            List of features used in the learning or applying of the model.
    Model.target : String
            In case of supervised learning, what variable to use as  a target to train on.
    Model.categories : np.array[String]
            In case of multiclass learning, what are the categories of the target.
    Model.mode : String
            Type of model. One of ('linear', 'binaryclass', 'multiclass', 'multilabel',
                'clustering', 'dimensionreduction', 'anomalydetection')
    Model.feature_imp : pandas.Series
            Series of model specific features with importance. Not the same as SHAP.
    Model.predictions : lfd.Data
            Data object that centralizes scores and predictions for all observations, as a result of Model.apply().
    Model.confusion : pd.DataFrame
            Confusion matrix, broken into data split, as a result from Model.evaluate().
    Model.confusion_cuts : List[Float]
            In case of linear outputs, the cutoff thresholds used to bin in favour of a confusion matrix.
    Model.metrics : pd.DataFrame
            Metrics table centralizing metrics, as a result of Model.evaluate().
    Model.shapvalues : pd.DataFrame
            Shap value table, as a result of Model.explain(). Has same dimensions as data object.
    '''    
    def __init__(self, name='Model'):
        self.name: str = name
        self.params: None
        self.clf = None
        self.features: None
        self.target: str = None
        self.categories: np.array[str] = None
        self.mode: str = None
        self.feature_imp: pd.Series = None
        self.predictions: Data = None
        self.confusion: pd.DataFrame = None
        self.metrics: pd.DataFrame = None
        self.shapvalues: pd.DataFrame = None

    def learn(self, data, target, mode, hyper_params, set_aside=None, seed=0):
        '''
        Learn a model on a data (training) object.
        
        Arguments
        ---------
        data : lfd.Data
                Data used for training the model.
        target : String
                Target variable for supervised learning.
        hyper_params : Dict
                Dictionary of hyper parameters which is passed on to the underlying algorithm.
        set_aside : List[String]
                List of column names which are not included as features in training.
        seed : Integer, default 0
                Defining the seed for reproducibility when training.
        '''
        logging.info(f'{data.df.shape} - {data.name} - learning {self.name}')
        self.params = hyper_params
        np.random.seed(seed)
        assert mode in ('linear', 'binaryclass', 'multiclass', 'multilabel', 
            'clustering', 'dimensionreduction', 'anomalydetection'), \
            "Mode should be one of ('linear', 'binaryclass', 'multiclass', \
                'multilabel', 'clustering', 'dimensionreduction', 'anomalydetection')"
        self.mode = mode

        # Store features to train on
        set_aside = [] if set_aside is None else set_aside
        self.features = [c for c in data.df.columns if c not in set_aside and c!=target]
        self.target = target

    def _predict_scores(self):
        '''
        Private method to predict scores on a pandas DataFrame.
        '''
        pass
        
    def apply(self, data, include_preds=True, store=False, cutoff_params=None):
        '''
        Apply a model on a data object, i.e. make scores and predictions for it.
        
        Arguments
        ---------
        data : lfd.Data
                Data used for apply the model on.
        include_preds : Bool, default True
                Whether to include predictions aside from the model scores.
                E.g. binary 1/0's based on scores and a cutoff.
        store : Bool, default False
                Whether to store the predictions as an attribute in the object.
        cutoff_params : Dict[String: List[Float]]
                Parameters to decide on possible multiple cutoffs used for binary classification.
                Format is dict(cutoffs=[0.3, 0.5], fix_recalls=[0.6], fix_flags=[0.1, 0.2], beta=1)
                Any of these is optional.
                1. Cutoffs is a manual list of cutoffs for which to include the predictions.
                2. Fix_recalls infers cutoffs to reach the desired recall. 3. Fix_flags infers 
                cutoffs to reach a desired number of flags. 4. Beta is a coefficient to decide a
                cutoff to optimize f1 which is the harmonic mean of precision and recall. 
                Beta = 1 favours both equally, beta > 1 favours recall, beta < 1 
                favours precision.
        '''
        logging.info(f'{data.df.shape} - {data.name} - applying {self.name}')

        # Intitialize a predictions df, add target and scores
        preds = pd.DataFrame(index=data.df.index)
        if self.target in data.df: preds['target'] = data.df[self.target]
                
        # In case of binary classification
        if self.mode=='binaryclass':
            preds['scores'] = pd.Series(self._predict_scores(data), index=data.df.index)
            if include_preds:
                # Include manual cutoffs, and cutoffs infered by fix recalls, flags and beta.
                if cutoff_params is None: cutoff_params = dict()
                cutoffs = cutoff_params.get('cutoffs')
                if cutoffs is not None and len(cutoffs)>1:
                    for cutoff in cutoffs:
                        preds[f'predictions_{cutoff}'] = (preds.scores>=cutoff).astype(int)
                beta = cutoff_params.get('beta', 1)
                if beta and self.target in data.df:
                    _, cutoff = get_best_cutoff(data.df[self.target], preds.scores, beta)
                    preds[f'predictions_best'] = (preds.scores>=cutoff).astype(int)
                fix_recalls = cutoff_params.get('fix_recalls')
                if fix_recalls and self.target in data.df:
                    for dr in fix_recalls:
                        cutoff = preds.scores[data.df[self.target]==1].quantile(1-dr)
                        preds[f'predictions_rec{dr}'] = (preds.scores>cutoff).astype(int)
                fix_flags = cutoff_params.get('fix_flags')
                if fix_flags:
                    for df in fix_flags:
                        cutoff = preds.scores.quantile(1-df)
                        preds[f'predictions_fla{df}'] = (preds.scores>cutoff).astype(int)

        # In case of linear regression
        elif self.mode=='linear':
            preds['scores'] = pd.Series(self._predict_scores(data), index=data.df.index)
            if include_preds and self.target in data.df:
                preds['target_bins'] = preds.target
                preds['predictions'] = preds.scores
                if preds.scores.nunique()>10:
                    self.confusion_cuts = pd.qcut(data.df[self.target], q=5, retbins=True, duplicates='drop')[1]
                    labels = [f'bin{c+1}' for c in np.arange(len(self.confusion_cuts[:-1]))]
                    # Apply learned cuttoffs on whole dataset
                    preds['target_bins'] = pd.cut(preds['target'], bins=self.confusion_cuts, labels=labels,
                                            duplicates='drop', include_lowest=True)
                    preds['predictions'] = pd.cut(preds.scores, bins=self.confusion_cuts, labels=labels,
                                                duplicates='drop', include_lowest=True)
                    preds['predictions'] = preds['predictions'].cat.add_categories('missing')
                    preds['predictions'].fillna('missing', inplace=True)

        # In case of multiclass classification
        elif self.mode=='multiclass':
            scores = pd.DataFrame(self._predict_scores(data), index=data.df.index)
            scores.columns = [f'scores_{c}' for c in self.categories]
            preds = preds.join(scores)
            if self.target in data.df:
                preds['scores'] = preds.apply(lambda row: row[f'scores_{row.target}'], axis=1)
            if include_preds:
                preds['predictions'] = self.categories[scores.to_numpy().argmax(axis=1)]

        elif self.mode=='clustering':
            scores = pd.DataFrame(self._predict_scores(data), index=data.df.index)
            clusters = np.arange(scores.shape[1])
            scores.columns = [f'scores_{c}' for c in clusters]
            preds = preds.join(scores)
            if include_preds:
                preds['predictions'] = clusters[scores.to_numpy().argmax(axis=1)]
            
        preds = Data(preds, 'Preds')
        if store: self.predictions = preds
        return preds
    
    def evaluate(self, predictions, broken_by='dataset'):
        '''
        Evaluate a mode on any data object. Outputs a confusion matrix and model metrics.
        
        Arguments
        ---------
        predictions : lfd.Data
                Data object as return by the apply function Contains scores and/r predictions
                to evaluate.
        broken_by : String, default 'dataset'
                Shoud be in index levels if given. The metrics and confusion matrices are broken by 
                its values. E.g. the confusion and model metrics for each data split.
        '''
        logging.info(f'{predictions.df.shape} - {predictions.name} - evaluating {self.name}')
        pred_cols = [c for c in predictions.columns if 'predictions' in c]
        if 'target' not in predictions.df or len(pred_cols) == 0: return None, None

        # 1. Build confusion table
        confusion = pd.DataFrame()
        target = 'target' if self.mode != 'linear' else 'target_bins'

        # Groupby given broken_by, targets and predictions
        if broken_by not in predictions.index.names: predictions.add_index(broken_by, 'All')
        values = predictions.index.get_level_values(broken_by)
        confusion = pd.concat([predictions.df.groupby([values, target, pred]).size() \
            for pred in pred_cols], axis=1, keys=pred_cols).unstack().fillna(0).astype(int)
        
        # These lines make sure all confusion conbinations are present
        # E.g. in the case where none of the actuals or predictions are 1
        confusion = confusion.reindex(pd.MultiIndex.from_product(
            confusion.index.levels), axis=0, fill_value=0)
        confusion = confusion.reindex(pd.MultiIndex.from_product(
            confusion.columns.levels), axis=1, fill_value=0)
    
        # 2. Create metric tables
        metrics = pd.DataFrame(
            index=pd.MultiIndex.from_product([predictions.index.levels[0]]),
            columns=pd.MultiIndex.from_tuples([(p, m) for p in pred_cols for m in ['accuracy']]))

        from itertools import product
        combinations = list(product(predictions.index.levels[0], pred_cols))
        preds = predictions.df
        # Linear metrics
        if self.mode=='linear':
            from scipy.stats import pearsonr, spearmanr
            for b, p in combinations:
                rmse = lambda p, t: np.sqrt(np.square(p-t).mean())
                pr, ta = preds.loc[b, 'scores'], preds.loc[b, 'target']
                metrics.loc[b, (p,'rmse')] = rmse(pr, ta)
                metrics.loc[b, (p,'pearsonr')] = pearsonr(pr, ta)[0]
                metrics.loc[b, (p,'spearmanr')] = spearmanr(pr, ta)[0]

        # Binary metrics
        elif self.mode=='binaryclass':
            from sklearn.metrics import roc_curve, auc
            bins = pd.qcut(preds.scores, 50, duplicates='drop')
            for b, p in combinations:
                metrics.loc[b, (p,'precision')] = confusion.loc[(b, 1),(p, 1)] / confusion.loc[b, (p,1)].sum()
                metrics.loc[b, (p,'recall')] = confusion.loc[(b, 1),(p, 1)] / confusion.loc[(b, 1),p].sum()
                metrics.loc[b, (p,'flags')] = confusion.loc[b, (p, 1)].sum() / confusion.loc[b, p].sum().sum()
                metrics.loc[b, (p,'f1')] = f1(metrics.loc[b, (p,'precision')], metrics.loc[b, (p,'recall')], 1)
                fpr, tpr, _ = roc_curve(preds.loc[b, 'target'], preds.loc[b, 'scores'])
                metrics.loc[b, (p,'auc')] = auc(fpr, tpr)
                metrics.loc[b, (p,'lift')] = preds.loc[b].groupby(bins.loc[b]).target.mean().iloc[-1] / preds.loc[b].target.mean()

        # Multiclass metrics
        elif self.mode=='multiclass': pass
        
        # Mode agnostic metrics
        for b, p in combinations:  # For any mode
            metrics.loc[b, (p,'accuracy')] = np.diag(confusion.loc[b, p]).sum() / confusion.loc[b, p].sum().sum()
            metrics.loc[b, (p,'c_index')] = c_index(preds.loc[b], 'target', 'scores')
        metrics = metrics.round(3)

        self.confusion = confusion
        self.metrics = metrics
        return self.confusion, self.metrics

    def explain(self, data):
        '''
        Explain the model with a data object, either globally (regression coefficients) or 
        locally (shap values).
        
        Arguments
        ---------
        data : lfd.Data
                Data object to use for explainability.
        '''
        logging.info(f'{data.df.shape} - {data.name} - explaining {self.name}')

    def save(self, directory=None, name=None, slim=False, as_pickle=False):
        '''
        Save model.
        
        Arguments
        ---------
        directory : String, optional
                Directory where the model should be saved.
        name : String, optional
                Name the folder or file will have in the directory. If not given, name 
                attribute will be used. If given, name attribute will be overwritten.
        slim : Bool, default False
                Whether to store also predictions, data and shapvalues. 
                Generally not needed for prediction, but useful for inspection.
        as_pickle : Bool, default False
                Whether to save to a pickle file.
        '''
        logging.info(f'Saving model {self.name}')
        self.name = name if name else self.name
        if directory is None or self.name is None:
            logging.warning('Pipeline not saved. No directory or name was provided.')
            return
            
        # Save as pickle
        if as_pickle:
            with open(os.path.join(directory, f'{self.name}.pkl'), 'wb') as f:
                pickle.dump(self, f)
            return

        # Reset path
        path = os.path.join(directory, self.name)
        if os.path.isdir(path): shutil.rmtree(path)
        os.mkdir(path)

        if not slim and self.predictions is not None:
            preds = self.predictions.df.drop('bins', axis=1) if 'bins' in self.predictions.df else self.predictions.df
            preds.to_parquet(os.path.join(path, 'predictions.parquet'))
        if not slim and self.shapvalues is not None:
            self.shapvalues.to_parquet(os.path.join(path, 'shapvalues.parquet'))
        if self.feature_imp is not None:
            self.feature_imp.to_csv(os.path.join(path, 'features.csv'))
        if self.metrics is not None:
            self.metrics.to_csv(os.path.join(path, 'metrics.csv'))
        if self.confusion is not None:
            self.confusion.to_csv(os.path.join(path, 'confusion.csv'))
        with open(os.path.join(path, 'model.json'), 'w') as f:
            json.dump({'features': self.features, 'target': self.target, 'mode': self.mode, 
                       'algorithm':self.__class__.__name__.lower(), 'params': self.params}, f, indent=4)
        with open(os.path.join(path, 'model.pickle'), 'wb') as f:
            pickle.dump(self.clf, f)

    def load(self, path=None, slim=False):
        '''
        Load model.
        
        Arguments
        ---------
        path : String, optional
                Path where the model should be loaded from. Can be a path to a 
                pipeline directory or pipeline pickle file.
        slim : Bool, default False
                If True, will not load predictions and shapvalues.
                Generally not needed for prediction, but useful for inspection.
        '''
        logging.info('Loading model')
        paths = path.split('/')
        storage = '/'.join(paths[:-1])

        # Load pickle file
        if paths[-1].endswith('.pkl'):
            name = paths[-1][:-4]
            with open(os.path.join(storage, f'{name}.pkl'), 'rb') as f:
                self = pickle.load(f)
            self.name = paths[-1][:-4]
            return self
        # Load files
        self.name = paths[-1]
        path = os.path.join(storage, self.name)

        files = os.listdir(path)
        self.feature_imp, self.metrics, self.confusion = None, None, None
        if 'features.csv' in files:
            self.feature_imp = pd.read_csv(os.path.join(path, 'features.csv'), index_col=0).squeeze('columns')
        if 'metrics.csv' in files:
            self.metrics = pd.read_csv(os.path.join(path, 'metrics.csv'), index_col=0, header=[0, 1])
        if 'confusion.csv' in files:
            self.confusion = pd.read_csv(os.path.join(path, 'confusion.csv'), index_col=[0, 1], header=[0, 1])
        if not slim and 'predictions.parquet' in files:
            self.predictions = Data(pd.read_parquet(os.path.join(path, 'predictions.parquet')), 'Preds')
        if not slim and 'shapvalues.parquet' in files:
            self.shapvalues = pd.read_parquet(os.path.join(path, 'shapvalues.parquet'))
        with open(os.path.join(path, 'model.json'), 'r') as f:
            data = json.load(f)
            self.features, self.target, self.mode, self.params = \
                data['features'], data['target'], data['mode'], data['params']
        with open(os.path.join(path, 'model.pickle'), 'rb') as f:
            self.clf = pickle.load(f)
        return self

    def _summary(self):
        '''
        Returns a pandas Series with meta information.
        '''
        return pd.Series(dict(
            Name = self.name,
            Algorithm = self.__class__.__name__.lower(),
            Type = self.clf.__class__.__name__,
            Features = len(self.features),
            Target = self.target if self.target else 'None',
            Mode = self.mode,
            Parameters = ", ".join([f'{k}: {v}' for k, v in self.params.items()]),
        ))

    def __repr__(self):
        string = 'This is a MODEL object.\n'
        string += tabulate(self._summary().to_frame(), tablefmt='simple_outline')
        string += '\n\nAttributes: clf, algorithm, mode, params, features, target, predictions, confusion, metrics, shapvalues.\n'
        string += 'Methods: learn, apply, evaluate, explain, save, load.\n'
        return string

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     for a in ['predictions', 'shapvalues']:
    #         state[a] = None
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
#         for a in ['predictions', 'shapvalues']:
#             self.__dict__[a] = None


# Helper functions
def c_index(df, actuals, scores, n=10000):
    higher = df[[actuals, scores]].sample(n, replace=True, random_state=0)
    lower = df[[actuals, scores]].sample(n, replace=True, random_state=1)
    mask = higher[actuals].values > lower[actuals].values
    c_index = (higher.loc[mask, scores].values > lower.loc[mask, scores].values).mean()
    return c_index

def recall(t, p):
    return ((t==1) & (p==1)).sum() / (t==1).sum()

def precision(t, p):
    return ((t==1) & (p==1)).sum() / (p==1).sum()

def f1(p, r, beta):
    return (1+beta**2)*p*r/(beta**2*p+r)

def get_best_cutoff(target, probs, beta):
    cutoffs_cand = np.arange(0, 1, 0.005)
    recalls, precisions, f1s, flags = [], [], [], []
    for c in cutoffs_cand:
        temp_preds = (probs>=c).astype(int)
        flags.append(temp_preds.sum())
        recalls.append(recall(target, temp_preds))
        precisions.append(precision(target, temp_preds))
        f1s.append(f1(precisions[-1], recalls[-1], beta))
    cutoff_table = pd.DataFrame({
        'recalls': recalls, 'precisions':precisions, 'f1s':f1s, 'flags':flags},
        index=cutoffs_cand).sort_values('f1s', ascending=False)
    best_cutoff = cutoff_table.index[0]
    logging.debug(f'Best cutoff for beta {beta} is {best_cutoff}')
    logging.debug('\n'+tabulate(cutoff_table.head(), headers='keys', tablefmt='psql'))
    return cutoff_table, best_cutoff
