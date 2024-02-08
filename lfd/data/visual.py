import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
from lfd.utils import get_memory

from .general import Data

import cv2


class Visual(Data):

    def __init__(self, path=None, name=''):


        self.type = 'visual'
        self.dimensions = (0, 0, 0)
        self.tensor = np.empty((0, ) + self.dimensions, dtype=np.int16)
        self.color_sequence = 'rgb'

        transformations = dict(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=(1, 1.2),
            horizontal_flip=False
        )
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        self.train_transformator = ImageDataGenerator(**transformations)
        self.data_transformator = ImageDataGenerator()
        
    def __repr__(self):
        print(f'This is a visual object, with {len(self.df)} images with dimension {self.dimensions}.')
        return ''

    def save(self, directory, only_meta=False):
        '''
        Save dataframe.
        '''
        path = os.path.join(directory, self.name)
        logging.info(f'Data - Saving to {path}')

        # Create new directory if needed
        if not os.path.exists(path): os.mkdir(path)
        
        # Store data
        if not only_meta: self.df.parquet(os.path.join(path, 'data.csv'))

        with open(os.path.join(path, f'meta.json'), 'w') as f:
            json.dump({
                'name': self.name,
                'type': self.type,
                'dimensions': list(self.dimensions),
                'color': self.color_sequence,
                'categories': self.categories, 
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'memory': get_memory(self) + ' and ' + get_memory(path=os.path.join(path, 'images'))
            }, f, indent=4)
    
    def load(self, directory, load_images=False, name='Visual'):
        '''
        Load dataframe.
        '''
        logging.info(f'Data - Loading from directory {directory}')
        super().load(directory, name)
        with open(os.path.join(directory, 'meta.json'), 'r') as f:
            json_data = json.load(f)
        self.name, self.type, self.dimensions, self.color_sequence, self.categories = \
            json_data['name'], json_data['type'], tuple(json_data['dimensions']), json_data['color'], json_data['categories']
        if load_images: self.load_images(self.df.index, directory)
        logging.info(f'Data - Loaded.')
        return self

    def load_images(self, indices, directory):
        self.tensor = np.empty((len(indices),) + self.dimensions, dtype=np.int16)
        for i, img_name in enumerate(indices):
            img = cv2.imread(os.path.join(directory, 'images', f'{img_name}.jpg'))
            if self.color_sequence == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Draw boxes if present
            if 'xmin' in self.df.columns:
                bboxes = self.df.loc[[img_name], ['xmin', 'ymin', 'xmax', 'ymax']]
                for _, (xmin, ymin, xmax, ymax) in bboxes.iterrows():
                    color = (1, 0, 0)
                    thickness = 2
                    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

            # Reshape towards dimensions
            if img.shape != self.dimensions:
                ratio=1
                y, x, _ = img.shape
                if x/y < ratio:
                    pad = int((ratio*y - x)/2)
                    img = np.pad(img, ((0,0), (pad, pad), (0,0)), 'constant')
                else:
                    pad = int((1/ratio*x - y)/2)
                    img = np.pad(img, ((pad, pad), (0,0), (0,0)), 'constant')
                img = cv2.resize(img, self.dimensions[:2], interpolation=cv2.INTER_LINEAR)

            # STACK
            #img = self.transformator.random_transform(img)
            img = np.expand_dims(img, axis=0)
            self.tensor[i] = img

    def get_generator(self, mode, train=False, batch_size=32):

        from tensorflow.keras.utils import Sequence

        class DataGenerator(Sequence):
            'Generates data for Keras'
            def __init__(self, data, directory, batch_size=32, dimensions=(28,28,3), shuffle=True, 
                color_sequence='rgb', transformator=None, mode=None):
                'Initialization'
                self.dimensions = dimensions
                # Set data
                self.data = data
                self.data.df.index = self.data.df.index.get_level_values('image')
                self.label = self.data.df['label']
                if mode=='binaryclass': self.label = self.label.cat.codes
                elif mode=='multiclass': self.label = pd.get_dummies(self.label)
                elif mode=='linear': self.label = self.label

                self.batch_size = batch_size
                self.shuffle = shuffle
                self.on_epoch_end()
                self.directory = directory
                self.color_sequence = color_sequence
                self.transformator = transformator
                self.mode = mode

            def __len__(self):
                'Denotes the number of batches per epoch'
                return int(np.ceil(len(self.data.df) / self.batch_size))

            def __getitem__(self, index):
                'Generate one batch of data'
                indices = np.arange(len(self.data.df))[index*self.batch_size:(index+1)*self.batch_size]
                X, y = self.__data_generation(indices)
                return X, y

            def on_epoch_end(self):
                'Updates indexes after each epoch'
                if self.shuffle == True:
                    self.data.df.sample(frac=1)

            def __data_generation(self, indices):
                'Generates data containing batch_size samples'
                # Initialization
                X_features = self.data.df.iloc[indices].drop('label', axis=1)
                y_label = self.label.iloc[indices].values
                X_images = np.zeros((len(indices), *self.dimensions))

                for i, img_name in enumerate(X_features.index):
                    try:
                        img = cv2.imread(os.path.join(self.directory, 'images', f'{img_name}.jpg'))
                        if self.color_sequence == 'rgb':
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # RESIZE
                        if img.shape != self.dimensions:
                            ratio=1
                            y, x, _ = img.shape
                            if x/y < ratio:
                                pad = int((ratio*y - x)/2)
                                img = np.pad(img, ((0,0), (pad, pad), (0,0)), 'constant')
                            else:
                                pad = int((1/ratio*x - y)/2)
                                img = np.pad(img, ((pad, pad), (0,0), (0,0)), 'constant')
                            img = cv2.resize(img, self.dimensions[:2], interpolation=cv2.INTER_LINEAR)

                        img = self.transformator.random_transform(img)
                        img = np.expand_dims(img, axis=0) / 256
                        X_images[i] = img
                    except: print(logging.info(f"Could not read {img_name}"))
                X = [X_images, X_features.values] if not X_features.empty else X_images
                return X, y_label


        return DataGenerator(self, f'datasets/{self.type}/{self.name}', mode=mode, dimensions=self.dimensions, batch_size=batch_size,
            color_sequence=self.color_sequence, transformator=self.train_transformator if train else self.data_transformator)
