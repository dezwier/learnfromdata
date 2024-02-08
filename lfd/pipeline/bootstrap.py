import os
import re
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
from tqdm.notebook import tqdm
import collections

from . import Pipeline
from ..data import Data
from ..transform import UniSelector
from ..config import set_logging


class Bootstrap:
    '''
    This class collects methods for bootstrapping pipelines. 

    Arguments
    ---------
    storage : String, optional
            Path to where the bootstrap should be saved to or loaded from. 
    logparams : Dict, optional
            Dictionary of parameters that are given to the set_logging function when 
            learning a pipeline.

    Attributes
    ----------
    Bootstrap.pipe : lfd.Pipeline
            Pipeline object which is used multiple times over to iterate.
    Bootstrap.storage : String
            Path to directory where the bootstrap can be saved or loaded from.
    Bootstrap.log_params : Dict
            Dictionary of parameters that set logging, and is given to config.set_params().
    '''
    def __init__(self, storage=None, logparams=None):
        self.pipe = None
        self.storage = storage
        if not os.path.exists(storage): os.mkdir(storage)
        self._experiments = [c for c in os.listdir(self.storage) if not '.' in c]
        self.logparams = logparams if logparams else dict(stdout=False, log_dir=None)
        set_logging(**self.logparams)

    def learn_pipelines(self, data, params, data_iters=1, model_iters=1, 
                         cutoff_params=None, method=None):
        '''
        Learns a number of pipelines with the goal of finding an optimum in parameter space.
        Reasons to learn multiple pipelines: Get insight in 1. number of rows to train on, 
        2. number of features to train on, 3. what hyper parameters to use for the modelling 
        algorithm and 4. to have a robust evaluation with random cross-validation.
        Total number of pipeline iterations is data_iters * model_iters.
        
        Arguments
        ---------
        data : lfd.Data
                Data object used for each of the pipeline iterations.
        params : Dict[]
                Parameters in dictionary used in underlying data, transform and modelling 
                functions. See Config.get_params() on how to format. At least one of the 
                parameters should be a numpy array to make bootstrap useful.
        data_iters : Int, default 1
                Number of times the data is split, sampled, balanced an transformed.
        model_iters : Int, default 1
                Number of times - per data iteration - that the models are trained.
        cutoff_params : Dict[String: List[Float]]
                Dictionary of cutoff parameters for binary classification. See Model.apply. 
                A list for fixed recalls, precisions, flags and betas can be given.
        method : Function, optional
                Arbitrary function that is applied to each lfd.Pipe object before saving.
        '''
        for _ in tqdm(np.arange(data_iters), 'Data Iterations'):
            self.pipe = Pipeline(logparams=self.logparams)
            self.pipe.set_aside = params['set_aside']
            
            self.pipe._set_data(dict_sample(params['data']), data=data.copy())
            self.pipe._learn_transform(dict_sample(params['transform']))
            
            for __ in tqdm(np.arange(model_iters), 'Model Iterations'):
                experiment = datetime.strftime(datetime.now(), '%y%m%d_%H%M%S_%f')
                self._experiments.append(experiment)
                self.pipe._learn_model(dict_sample(params['model']), cutoff_params)
                self.pipe.save(self.storage, name=experiment, as_pickle=False, slim=True)
                if method: self.pipe = method(self.pipe)
                np.random.seed(int(experiment[-6:]))  # Reset seed after every experiment

        return self

    def _get_features(self, aggregate=False, model='Xgboost'):
        '''
        After pipelines have been iterated, this function reads the feature importance of 
        each experiment and aggregates them into a single pandas DataFrame.
        '''
        features_df = pd.DataFrame()
        for i, experiment in enumerate(self._experiments):
            path_e = os.path.join(self.storage, f'{experiment}')
            features = pd.read_csv(os.path.join(path_e, model, 'features.csv'), index_col=0, squeeze=True)
            features_df = features_df.append(features.rename(experiment))
        if aggregate: features_df = features_df.mean().abs().sort_values(ascending=False)
        return features_df

    def _get_metrics(self):
        '''
        After pipelines have been iterated, this function reads the metrics of 
        each experiment and aggregates them into a single pandas DataFrame.
        '''
        metrics_df = []
        for i, experiment in enumerate(self._experiments):
            path_e = os.path.join(self.storage, f'{experiment}')
            models = [m for m in os.listdir(path_e) if \
                      os.path.isdir(os.path.join(path_e, m)) and not re.match(r'^(\.)', m) and m != 'transform']
            metrics = pd.concat(([pd.read_csv(os.path.join(path_e, m, 'metrics.csv'), index_col=0, header=[0, 1]
                                             ) for m in models]), keys=models, names=['model'])
            metrics_df.append(metrics)
            if (i+1)%1000==0: logging.info(f'{i+1} metric experiments read.')
        logging.info(f'All {i+1} metric experiments read.')
        metrics_df = pd.concat((metrics_df), keys=self._experiments, names=['experiment'])
        return metrics_df

    def _get_params(self):
        '''
        After pipelines have been iterated, this function reads the used parameters of 
        each experiment and aggregates them into a single pandas DataFrame.
        '''
        params_df = pd.DataFrame()
        for i, experiment in enumerate(self._experiments):
            with open(os.path.join(self.storage, experiment, 'params.json'), 'r') as f:
                params = pd.Series(dict_flatten(dict_map(json.load(f))), name=experiment)
            params_df = params_df._append(params)
            if (i+1)%1000==0: logging.info(f'{i+1} parameter experiments read.')
        logging.info(f'All {i+1} parameter experiments read.')
        return params_df

    def get_meta(self, model, metrics=None, dataset='Test', predictions=None, include_seeds=False):
        '''
        This function assumes a iterate_pipeline has run (or is still running).
        It combines all parameters and resulting metrics of those stored pipelines into 
        a single pandas DataFrame. This could be used for hyperparameter selection.
        
        Arguments
        ---------
        model : String
                Which model for which the metrics and params should be read.
        metrics : String, optional
                Which metrics should be used, e.g. 'accuracy'. If not given, all metrics
                available will be part of the output table.
        dataset : String, default 'Test'
                For which dataset, e.g. 'Test' or 'Train' the table should be made.
        predictions : String, optional
                For which predictions the table should be made, e.g. 'predictions_rec0.16'
                If not given, the first available prediction set is used.
        include_seeds : Bool, default False
                Whether to include varying seeds over the experiments in the output table.
                Can be usefull to assess cross validation or reproducibility.
        '''        
        # Gather params and metrics
        params = self._get_params()
        metrics_df = self._get_metrics().xs((model, dataset), level=['model', 'dataset'])
        
        # Collect predictions and metrics if not given
        predictions = metrics_df.columns.levels[0][0] if predictions is None else predictions
        metrics = metrics_df.columns.levels[1] if metrics is None else metrics
        
        # Join two datasets, 1 line per experiment
        meta = metrics_df[predictions][metrics].join(params)
        if not include_seeds: meta = meta.iloc[:, ~meta.columns.str.contains('seed|SEED')]
        meta = Data(meta, name='Meta')
        #meta.df = meta.df.sort_values(metrics_df.columns, ascending=False)
        meta = UniSelector().learn(meta, min_occ=0.01, max_occ=0.99, include_missings=False).apply(meta)
        return meta
    

def dict_sample(d):
    if type(d)==list:
        return [dict_sample(v) for v in d]
    elif type(d)==dict:
        new_d = d.copy()
        for k, v in d.items():
            if type(v) in (np.ndarray, np.array):
                new_d[k] = np.random.choice(v)
                new_d[k] = int(new_d[k]) if type(new_d[k])==np.int64 else new_d[k]
                if type(new_d[k])==dict:
                    new_d[k] = dict_sample(new_d[k])
            elif type(v) in (dict, list):
                new_d[k] = dict_sample(v)
        return new_d
    else: return d

def dict_flatten(d, prefix=''):
    new_d = {}
    for k, v in d.items():
        k = f'{prefix}|{k}' if prefix!='' else str(k)
        if type(v)==dict:
            new_d.update(dict_flatten(v, k))
        elif type(v)==list:
            if type(v[0])==dict:
                for e in v:
                    new_d.update(dict_flatten(e, k))
            else: new_d[k] = v
        else: new_d[k] = v
    return new_d

def dict_map(d):
    new_d = d.copy()
    for k, v in d.items():
        if type(k)==str and k.isdigit():
            new_d.update({int(k): v})
            del new_d[k]
        elif type(v)==dict:
            new_d[k] = dict_map(v)
    return new_d

def dict_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def dict_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = dict_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

def get_basevars(features):
    return list(pd.Series(features).str.split('__', expand=True)[0].unique())
