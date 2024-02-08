import numpy as np
import pandas as pd
import logging
import os
import json

from .general import Model


class NeuralNet(Model):
    '''
    This class collects methods for training and utilising an Xgboost model. 
    '''    
    
    def __init__(self, name='neuralnet'):
        super().__init__(name=name)
        self._name = name

    def learn(self, data, target, mode, hyper_params={}, set_aside=None, seed=0):
        '''
        This function trains a model.
        '''
        logging.info(f'{data.df.shape} - {data.name} - training {self.name} with parameters {hyper_params}')
        super().learn(data, target, mode, hyper_params, set_aside, seed)
        from tensorflow.keras.regularizers import l1_l2
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import Model as KerasModel
        from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, Concatenate, BatchNormalization, Activation

        np.random.seed(seed)
        assert mode in ('linear', 'binaryclass', 'multiclass', 'multilabel'), \
            "Mode should be one of ('linear', 'binaryclass', 'multiclass', 'multilabel')"
        set_aside = [] if set_aside is None else set_aside
        self.features = [c for c in data.df.columns if c not in set_aside and c!=target]
        self.target = target
        self.mode = mode
        if 'activation' not in hyper_params.keys(): hyper_params['activation'] = 'tanh'
        if 'regularization' not in hyper_params.keys(): hyper_params['regularization'] = None
        if 'epochs' not in hyper_params.keys(): hyper_params['epochs'] = 10
        if 'batch_size' not in hyper_params.keys(): hyper_params['batch_size'] = 16

        # Store features to train on
        act = hyper_params['activation']
        reg = l1_l2(hyper_params['regularization'], hyper_params['regularization'])
        outputs_n = 1 if self.mode in ('linear', 'binaryclass') else data.df[target].nunique()
        output_act = 'sigmoid' if self.mode=='binaryclass' \
            else 'linear' if self.mode=='linear' \
            else 'softmax' if self.mode=='multiclass' \
            else 'NOT_IMPLEMENTED'

        if data.type=='visual' and hyper_params['transfer_learning']:

            # Retrieve and set pretrained model
            input_layer = Input(shape=data.dimensions)
            import tensorflow_hub as hub
            x = hub.KerasLayer(hyper_params['transfer_learning'], output_shape=None, trainable=False)(input_layer)
            x = Dropout(0.5)(x)
            x = Dense(512, activation=act)(x)
            x = Dropout(0.5)(x)
            if len(self.features) > 0:
                input_layer2 = Input(shape=len(self.features))
                input_layer = [input_layer, input_layer2]
                x = Concatenate()([x, input_layer2])

        # Build network architecture
        elif data.type=='visual' and not hyper_params['transfer_learning']:
            input_layer = x = Input(shape=data.dimensions)
            counter = 1
            while x.shape[1] > 2:
                x = Conv2D(data.dimensions[-1]*2**counter, kernel_size=(4, 4), strides=(2, 2), padding='same', activation=act)(x)
                x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
                counter += 1
            if len(self.features) > 0:
                input_layer2 = Input(shape=len(self.features))
                input_layer = [input_layer, input_layer2]
                x = Concatenate()([x, input_layer2])
            x = Dropout(0.5)(x)
            x = Flatten()(x)
            layers = calculate_layers(x.shape[1], outputs_n)

            for layer in layers:
                x = Dense(layer, act, kernel_regularizer=reg)(x)
            x = Dropout(0.5)(x)

        elif data.type == 'tabular':
            input_layer = x = Input(shape=len(self.features))
            layers = calculate_layers(x.shape[1], outputs_n)
            for layer in layers:
                x = Dense(layer, act, kernel_regularizer=reg)(x)
            x = Dropout(0.5)(x)
            
        output_layer = Dense(outputs_n, activation=output_act)(x)
        self.clf = KerasModel(inputs=input_layer, outputs=output_layer)
        self.clf.summary()

        # Compile, with mean_squared_error or binary_crossentropy loss
        loss = 'mean_squared_error' if self.mode=='linear' \
            else 'binary_crossentropy' if self.mode=='binaryclass'\
            else 'categorical_crossentropy' if self.mode=='multiclass'\
            else 'NOT_IMPLEMENTED'
        metrics = ['accuracy'] if self.mode in ('binaryclass', 'multiclass') else []
        self.clf.compile(Adam(lr=hyper_params['learning_rate']), loss=loss, metrics=metrics)
        
        # Train
        from tensorflow.keras.callbacks import EarlyStopping
        callback = EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True, verbose=1)
        if data.type=='visual':
            history = self.clf.fit(data.get_generator(self.mode, train=True, batch_size=hyper_params['batch_size']), 
                epochs=hyper_params['epochs'], use_multiprocessing=True, workers=5, callbacks=[callback])

        elif data.type=='tabular':
            if self.mode=='binaryclass': target = data.df[self.target].cat.codes
            elif self.mode=='multiclass': 
                target = pd.get_dummies(data.df[self.target])
                self.categories = target.columns.values
            elif self.mode=='linear': target = data.df[self.target]
            history = self.clf.fit(data.df[self.features].astype(float), target, callbacks=[callback],
                                epochs=hyper_params['epochs'], use_multiprocessing=True,
                                batch_size=hyper_params['batch_size'])#, verbose=0)
        # Show feature importance
        self.feature_imp = pd.Series()  # Todo: implement
        return self

    def _predict_scores(self, data):
        super()._predict_scores()
        if data.type == 'visual':
            X = data.get_generator(self.mode, train=False)
        elif data.type=='tabular':
            X = data.df[self.features].astype(float)
        if self.mode in ('binaryclass', 'linear'):
            scores = self.clf.predict(X).flatten()
        elif self.mode=='multiclass':
            scores = self.clf.predict(X)
        return scores
    
    def explain(self, data, split=2):
        super().explain(data)
        assert(split in data.df.index.levels[0], "Split is not present in data")
        # TODO: implement
        pass


def calculate_layers(n_input, n_output):
    summ = n_input + n_output
    layers = np.geomspace(n_input, n_output, endpoint=False,
        num=int(np.power(summ, 0.2)*1.2)).round().astype(int)
    return layers[1:]
