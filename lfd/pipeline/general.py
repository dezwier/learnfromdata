import pandas as pd
import os
import logging
import json
from memory_profiler import profile
import shutil
from tabulate import tabulate
from datetime import datetime
import pickle

from ..data import *
from ..transform import TransformEnum
from ..model import ModelEnum
from ..config import set_logging


class Pipeline:
    '''
    This class collects methods for learning applying, saving and loading a pipeline, which
    is composed a number of transformers (imputer, encoder...) and (un)supervised model(s).
    Configuration on how to learn happens with a dictionary as provided by config.get_params().
    This configuration can be adapted, and makes learning and applying of a pipeline reproducible.

    Arguments
    ---------
    logparams : Dict, optional
            Dictionary of parameters that are given to the get_params function when 
            learning a pipeline.
    name : String, optional
            Arbitrary name for the pipeline object. If not given, it will be the current 
            timestamp in format 'yymmdd_hhmmss_ms'.
            
    Attributes
    ----------
    Pipeline.name : String
            Name of the pipeline.
    Pipeline.data : lfd.Data
            Data object to use in a pipeline. Used to split, predict, evaluate, explain, etc.
    Pipeline.train : lfd.Data
            Data object to train on. Either given or split from self.data when learning.

    Pipeline.transformers : Dict[String: lfd.Transformer]
            Dictionary of transformer objects.
    Pipeline.models : Dict[String: lfd.Model]
            Dictionary of model objects.
    Pipeline.cal_models : Dict[String: lfd.Model]
            Dictionary of model objects, that calibrate the models in self.models.
    Pipeline.stack_model : lfd.Model
            Model object that is built on top of the models in self.models.

    Pipeline.params : Dict
            Dictionary as given or provided by config.get_params().
    Pipeline.set_aside : List
            List of variables that is not transformed or used as features in modelling.
    Pipeline.log_params : Dict
            Dictionary of parameters that set logging, and is given to config.set_params().
    '''
    def __init__(self, logparams=None, name=None):
        set_logging(**dict() if logparams is None else logparams)
        # Data objects
        self.data = None
        self.train = None
        
        # Transform & model objects
        self.transformers = {}
        self.models = {}
        self.cal_models = {}
        self.stack_model = None
        
        # Settings attributes
        self.params = {}
        self.set_aside = None
        self.logparams = logparams

        self.name = name if name else datetime.strftime(datetime.now(), '%y%m%d_%H%M%S_%f')
        
    def _set_data(self, params, data=None, train=None):
        '''
        Private helper function for learning a pipeline. Set data objects, optionally split into 
        train - test - validation, optionally adding random noise, optional sampling, 
        optional balancing of training.
        '''
        self.params['data'] = params
        self.data = data
        self.train = train

        if train is not None: self.train = train

        # Split in training, test and validation, add noise
        if params.get('test_split'): 
            self.data.split(subset='All', names=('Train', 'Test'), **params['test_split'])
        if params.get('valid_split'):
            self.data.split(subset='Train', names=('Train', 'Valid'), **params['valid_split'])
        if params.get('add_noise'): 
            self.data.add_noise(**params['add_noise'])
        if params.get('test_split') is None and params.get('valid_split') is None and 'dataset' not in self.data.df.index.names:
            self.data.add_index('dataset', 'Train')

        # Create samples for learning and evaluation
        if self.train is None:
            self.train = self.data.select(['Train'], drop=False, name='Train')
        if params.get('sample'):
            self.train = self.train.sample(params['sample'])
        if params.get('train_balance'):
            self.train = self.train.balance(**params['train_balance'], name='Balanced')
        if params.get('actual_balance'):
            self.data = self.data.balance(**params['actual_balance'], name='All')

    def _learn_transform(self, params):
        '''
        Private helper function for learning a pipeline. Learns any number of transformers given
        by a parameter dictionary.
        '''    
        # Create transformers and learn on train
        self.params['transform'] = params
        self.transformers = {}
        for name, parameters in params.items():
            if parameters is None: continue
            self.transformers[name] = TransformEnum[name].value()
            self.train = self.transformers[name].learn(
                self.train, set_aside=self.set_aside, **parameters).apply(self.train)

        # Apply the transform objects on validation and testing
        self.data = self._apply_transform(self.data)
        self.params['transform'] = params
      
    def _apply_transform(self, data):
        '''
        Private helper function for applying a pipeline. Applies any number of transformers which
        were previously learned.
        '''
        for name in self.transformers:
            self.transformers[name].apply(data, inplace=True)
        return data

    def _learn_model(self, params, cutoff_params=None, evaluate=True, explain=False):
        '''
        Private helper function for learning a pipeline. Learns any number of models given
        by a parameter dictionary.
        '''    
        self.params['model'] = params
        seed = params.get('seed_train', 0)
        target = params.get('target')
        mode = params['mode']
        self.models, self.cal_models, self.stack_model = {}, {}, None
                
        # Train models
        base_params = [v for k, v in params.items() if k.startswith('base') and v is not None]
        for param_set in base_params:
            if param_set is None: continue
            # Make an object of a class according to 'algorithm' params, with 'name' attribute
            model = ModelEnum[param_set['algorithm']].value(param_set['name'])
            model.learn(self.train, target, mode, set_aside=self.set_aside, seed=seed, 
                        hyper_params=param_set['hyper_params'])
            if explain: model.explain(self.data)
            self.models[param_set['name']] = model

        # Train calibration models
        if params.get('calibrate') is not None:
            for model in self.models.values():
                train_unbalanced = self.data.select(['Train'], drop=False, name='Train')
                scores = model.apply(train_unbalanced, include_preds=False)
                cal_model = ModelEnum[params['calibrate']['algorithm']].value(name='C_'+model.name)
                cal_model.learn(scores, 'target', mode, set_aside=['target'], seed=seed,
                                hyper_params=params['calibrate'].get('hyper_params'))
                self.cal_models[model.name] = cal_model

        # Train stack model
        if params.get('stack'):
            scores = pd.DataFrame({name:model.predict_scores(self.train.df) for name, model in self.models.items()})            
            scores[target] = self.train.df[target].values
            self.stack_model = ModelEnum[params['stack']['algorithm']].value(name='Stack')
            self.stack_model.learn(
                Data(scores, name='scores'), target, mode, set_aside=self.set_aside,
                hyper_params=params['stack'].get('hyper_params'), seed=seed)

        # Evaluate all models
        self._apply_model(self.data, cutoff_params, evaluate=evaluate)
                 
    def _apply_model(self, data, cutoff_params=None, evaluate=True, explain=False):
        '''
        Private helper function for applying a pipeline. Applies any number of models which
        were previously learned.
        '''
        for model in self.models.values(): # apply the model to get the predictions
            model.apply(data, store=True, cutoff_params=cutoff_params)
            if evaluate: model.evaluate(model.predictions)
            if explain: model.explain(data)
        
        # Apply platt-calibration to the predictions of the model to get calibrated scores
        for modelname, cal_model in self.cal_models.items(): 
            cal_model.apply(self.models[modelname].predictions, store=True, cutoff_params=cutoff_params)
            if evaluate: cal_model.evaluate(cal_model.predictions)

        if self.stack_model:
            scores = pd.DataFrame({name: model.predictions.df.scores for name, model in self.models.items()})
            scores[self.stack_model.target] = data.df[self.stack_model.target].values
            self.stack_model.apply(Data(scores, name='scores'), store=True, cutoff_params=cutoff_params)
            if evaluate: self.stack_model.evaluate(self.stack_model.predictions)

    @profile
    def learn(self, params, data=None, train=None, cutoff_params=None, evaluate=True, explain=True):
        '''
        Learns a modelling pipeline. From data setting, to learning transformers, to learning and 
        evaluating models.
        
        Arguments
        ---------
        params : Dict[]
                Parameters in dictionary used in underlying data, transform and modelling 
                functions. See Config.get_params() on how to format. 
        data : lfd.Data, optional
                Data object used as input for splitting, adding noise, sampling and balancing.
                If not given, self.data attribute is used.
        train : lfd.Data, optional
                Data object used for training. Typically not given, but can be to overrule 
                splitting data. If not given, self.train attribute is used. If this is None, 
                self.data will be split.
        cutoff_params : Dict[String: List[Float]]
                Dictionary of cutoff parameters for binary classification. See Model.apply. 
                A list for fixed recalls, precisions, flags and betas can be given.
        evaluate : Bool, default True
                Whether the pipeline should also be evaluated by means of metrics and confusion matrices.
        explain : Bool, default True
                Whether the pipeline should also be explained by means of SHAP values.
        '''
        logging.info('Learning pipeline')
        self.set_aside = params['set_aside']
        self._set_data(params['data'], data, train)
        self._learn_transform(params['transform'])
        self._learn_model(params['model'], cutoff_params, evaluate, explain)
        return self
    
    def apply(self, data=None, cutoff_params=None, evaluate=False, explain=False):
        '''
        Applies a modelling pipeline on any data object. From applying transformers, to applying and 
        evaluating models.
        
        Arguments
        ---------
        data : lfd.Data, pandas.DataFrame, String, optional
                Data object to apply the pipeline on. Can be lfd.Data, pandas.DataFrame or a pathfile 
                to Parquet or csv. Pipeline must first be learned (on same or other data object).
        cutoff_params : Dict[String: List[Float]]
                Dictionary of cutoff parameters for binary classification. See Model.apply. 
                A list for fixed recalls, predictions, flags and betas can be given.
        evaluate : Bool, default True
                Whether the pipeline should also be evaluated by means of metrics and confusion matrices.
        explain : Bool, default False
                Whether the pipeline should also be explained by means of SHAP values.
        '''
        logging.info('Applying pipeline')
        self.data = Data(data, 'Data')
        self.data = self._apply_transform(self.data)
        self._apply_model(self.data, cutoff_params, evaluate, explain)         
        return self

    def save(self, directory=None, name=None, slim=False, as_pickle=False):
        '''
        Save pipeline.
        
        Arguments
        ---------
        directory : String, optional
                Directory where the pipeline should be saved.
        name : String, optional
                Name the folder or file will have in the directory. If not given, name 
                attribute will be used. If given, name attribute will be overwritten.
        slim : Bool, default False
                Whether to store also predictions, data and shapvalues. 
                Generally not needed for prediction, but useful for inspection.
        as_pickle : Bool, default False
                Whether to save to a pickle file.
        '''
        logging.info('Saving pipeline')
        # Set name and check presence
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

        # Save parameters
        if self.params: 
            with open(os.path.join(path, 'params.json'), 'w') as f:
                json.dump(self.params, f, indent=4)

        # Save transformers
        transform_path = os.path.join(path, 'transform')
        if not os.path.isdir(transform_path): os.mkdir(transform_path)
        for name in self.transformers:
            self.transformers[name].save(transform_path)
            
        # Save models
        kwargs = dict(directory=path, slim=slim, as_pickle=False)
        for model in self.models.values(): 
            model.save(**kwargs)
            if model.name in self.cal_models.keys():
                self.cal_models[model.name].save(**kwargs)
        if self.stack_model: 
            self.stack_model.save(**kwargs)
            
        if not slim and self.data is not None:
            self.data.save(os.path.join(path, 'data.parquet'))
            
    def load(self, path=None, slim=False):
        '''
        Load pipeline.
        
        Arguments
        ---------
        path : String, optional
                Path where the pipeline should be loaded from. Can be a path to a 
                pipeline directory or pipeline pickle file.
        slim : Bool, default False
                If True, will not load predictions, data and shapvalues. Generally not 
                needed for prediction, but useful for inspection or visualization.
        '''
        logging.info('Loading pipeline')
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

        with open(os.path.join(path, 'params.json'), 'r') as f:
            self.params = dict_map(json.load(f))

        # Load transformers
        self.transformers = {}
        transform_path = os.path.join(path, 'transform')
        for name in self.params['transform']:
            self.transformers[name] = TransformEnum[name].value()
            self.transformers[name].load(transform_path)

        # Load models
        self.models, self.cal_models = {}, {}
        model_dirs = [f for f in os.listdir(path) if os.path.isdir(f'{path}/{f}') and f != 'transform']
        for model_dir in model_dirs:
            with open(os.path.join(path, model_dir, 'model.json'), 'r') as f:
                algorithm = json.load(f)['algorithm']
            model = ModelEnum[algorithm].value(model_dir).load(os.path.join(path, model_dir), slim=slim)
            if model_dir=='stack': self.stack_model = model
            elif model_dir.startswith('C_'): self.cal_models[model.name[2:]] = model
            else: self.models[model.name] = model
        if not slim and 'data.parquet' in os.listdir(path):
            self.data = Data(os.path.join(path, 'data.parquet'), 'Data')
        return self

    def _summary(self):
        '''
        Returns 5 pandas objects with meta information.
        '''
        datas = [d for d in [self.data, self.train] if d is not None]
        data = [data._summary() for data in datas]
        data = pd.concat(data, axis=1).T if len(data)>0 else pd.DataFrame()
        models = [model._summary()._append(pd.Series(dict(Role='Base'))) for model in self.models.values()]
        models += [model._summary()._append(pd.Series(dict(Role='Calibration'))) for model in self.cal_models.values()]
        models = pd.concat(models, axis=1).T if len(models)>0 else pd.DataFrame()
        transformers = [transformer._summary() for transformer in self.transformers.values()]
        transformers = pd.concat(transformers, axis=1).T if len(transformers)>0 else pd.DataFrame()
        return data, transformers, models, self.metrics, self.confusion

    def __repr__(self):
        string = 'This is a PIPELINE object.'
        data, transformers, models, metrics, confusion = self._summary()
        string += f"\n\nDATA\n"
        string += tabulate(data, tablefmt='simple_outline', showindex=False, headers='keys')
        string += f"\n\nTRANSFORMERS\n"
        string += tabulate(transformers, tablefmt='simple_outline', showindex=False, headers='keys')
        string += f"\n\nMODELS\n"
        string += tabulate(models, tablefmt='simple_outline', showindex=False, headers='keys')
        string += '\n\nAttributes: data, train, transformers, models, cal_models, stack_model, metrics, confusion, params, storage.\n'
        string += 'Methods: learn, apply, save, load.'
        return string

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     state['data'] = None
    #     state['train'] = None
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     if self.slim:
    #         for a in ['data', 'train']:
    #             self.__dict__[a] = None

    @property
    def metrics(self):
        metrics = [model.metrics for model in self.models.values() if model.metrics is not None]
        return pd.concat(metrics, keys=self.models.keys()).round(3) if metrics else None

    @property
    def confusion(self):
        confusion = [model.confusion for model in self.models.values() if model.confusion is not None]
        return pd.concat(confusion, keys=self.models.keys()).round(3) if confusion else None


# Helper functions
def dict_map(d):
    new_d = d.copy()
    for k, v in d.items():
        if type(k)==str and k.isdigit():
            new_d.update({int(k): v})
            del new_d[k]
        elif type(v)==dict:
            new_d[k] = dict_map(v)
    return new_d


def reset_storage(folder):
    """ If folder exists, clean. Else, create. """
    if os.path.isdir(folder): 
        try: 
            shutil.rmtree(folder)
            logging.info('Deleted previous run')
        except:
            logging.info('Overwriting previous run')
    else: 
        os.mkdir(folder)
        logging.info('Creating new storage folder')
    #set_logging(**dict(log_dir=folder) if logparams is None else logparams)
