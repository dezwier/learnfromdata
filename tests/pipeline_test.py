import unittest
import numpy as np
from lfd import Data, Pipeline, set_logging, get_params, Bootstrap
import shutil
import os


class PipelineTests(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.quick = False
        self.storage = 'test'
        os.mkdir(self.storage)
        self.logparams = dict(stdout=False, stdout_level='INFO', log_dir=None, log_level='DEBUG')

    @classmethod
    def tearDown(self):
        shutil.rmtree(self.storage, ignore_errors=True)

    def test_pipeline_titanic(self):
        # Get data, params and learn pipeline
        data = Data('datasets/titanic.csv', name='titanic')
        params = get_params(target='Survived', set_aside=['Survived'], mode='binaryclass')
        pipe = Pipeline(logparams=self.logparams)
        pipe.learn(params, data=data, cutoff_params=dict(fix_recall=[0.3], beta=1), evaluate=True, explain=True)
        pipe.save(self.storage, name='testrun', as_pickle=True)
        self.assertAlmostEqual(pipe.metrics.loc['Xgboost'].predictions_best.lift.Train, 2.608)

    def test_pipeline_houses(self):
        # Get data, params and learn pipeline
        data = Data('datasets/houses.csv', name='houses')
        params = get_params(target='SalePrice', set_aside=['SalePrice'], mode='linear')
        pipe = Pipeline(logparams=self.logparams)
        pipe.learn(params, data=data, evaluate=True, explain=True)
        pipe.save(self.storage, name='testrun', as_pickle=True)
        self.assertAlmostEqual(pipe.metrics.loc['Xgboost'].predictions.accuracy.Train, 0.978595890410959 )

    def test_bootstrap_houses(self):
        # Get data, params and run bootstrap
        data = Data('datasets/houses.csv', name='houses')
        params = get_params(target='SalePrice', set_aside=['SalePrice'], mode='linear')
        params['data']['test_split']['seed'] = np.array([1, 2, 3, 4, 5])
        params['model']['base0']['hyper_params']['n_estimators'] = np.array([10, 20, 30, 40, 50])
        boot = Bootstrap(os.path.join(self.storage, 'bootstrap'), self.logparams)
        boot.learn_pipelines(
            data.copy(), params, data_iters=2, model_iters=3)
        meta = boot.get_meta(model='C_Xgboost', dataset='Test', metrics=None, predictions=None)
        self.assertEqual(meta.shape[0], 6)
