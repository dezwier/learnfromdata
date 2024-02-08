import pandas as pd
import numpy as np
import os
import json
from copy import deepcopy
import logging
from tabulate import tabulate
from IPython.core.display import display

import warnings  # Todo: handle warnings
warnings.filterwarnings('ignore')

from ..utils import set_dtypes, get_memory


class Data:
    '''
    This class collects methods for data operations. It is provides extra functionality 
    on top of pandas DataFrames, and has its data stored in the self.df attribute.

    Arguments
    ---------
    df : pandas.DataFrame, lfd.Data, or String, optional
            Source for the data, type will be inferred.
    name : String, default 'Data'
            Arbitrary name for the data object that shows up in the LFD output,
            such as metrics, confusions, dashboards, etc.


    Attributes
    ---------
    Data.df : pandas.DataFrame
            Stores the data.
    Data.name : String, default 'Data'
            Name of the data object, that shows up in the LFD output,
            such as metrics, confusions, dashboards, etc.
    Data.shape : Tuble[Int, Int]
            Dimensions of the data, rows and columns.
    Data.type : String, default 'tabular'
            Type of dataset, can be tabular, visual or graph
    Data.index : pandas.Index
            Index of the data.
    Data.columns : pandas.Index
            Columns of the data.
    '''
    def __init__(self, df=None, name='Data', type='tabular'):
        self.df = None
        self.name = None
        self.type = type
        self.load(df, name)

    @property
    def shape(self): return self.df.shape

    @property
    def index(self): return self.df.index

    @property
    def columns(self): return self.df.columns

    @property
    def cont_columns(self): return [c for c in self.df.columns if self.df[c].dtype.kind in 'iufc']

    @property
    def cat_columns(self): return [c for c in self.df.columns if self.df[c].dtype.kind not in 'iufc']

    @classmethod
    def merge(cls, data_objects, merge_types, prefixes=None, name='Merged'):
        '''
        Class method. Merging data objects together.
        
        Arguments
        ---------
        data_objects : List[lfd.Data]
                List of data objects to merge.
        merge_types : List[String] 
                Elements should be in in ('left', 'inner', 'outer'). List of length same as data_objects 
                minus 1, defining how to merge each table.
        prefixes : List[String], optional
                List of length same as data_objects, defining a prefix for column names per table.
        name : String, default 'Merged'
                Name for the merged dataframe
        '''
        prefixes = ['' for _ in data_objects] if prefixes is None else prefixes
        df = data_objects.pop(0).df.add_prefix(prefixes.pop(0))
        logging.info(f'Starting from shape {df.shape}')
        for i, new_df in enumerate(data_objects):
            df = pd.merge(
                df, new_df.df.add_prefix(prefixes[i]), how=merge_types[i],
                left_index=True, right_index=True, validate='one_to_one')
            logging.info(f'Data for {prefixes[i]} is {merge_types[i]} merged, shape is {df.shape}')

        # Remove categories that might be lost because of lost rows due to merging
        for var in df.select_dtypes('category').columns:
            df[var] = df[var].cat.remove_unused_categories()
            df[var].cat.set_categories(df[var].cat.categories.astype(str))
        return Data(df, name, self.type)

    def split(self, subset='Train', mask=None, test_size=None, stratify_col=None, 
              names=('Train', 'Test'), seed=0):
        '''
        Split data object into 2 subsets, by creating or adding a dataset index. Additional 
        memory usage is very limited this way.
        
        Arguments
        ---------
        subset : String, default 'Train'
                Which index to split in case a dataset index is present. If not present, it will
                use the whole dataset.
        mask : String or pandas.Series[Bool], optional
                Identifying how the subset/dataset should be split. Can be a string too which 
                will be evaluated, e.g. "self.df.year > 2021". If given, test_size is ignored.
        test_size : Float, optional
                Only used if mask is None. Share of subset/dataset that is split randomly.
        stratify_col : String, optional
                Variable that is optionally used to stratify the split, meaning the distribution 
                of this variable will remain the same in each split. Often used for a model target.
        names : Tuple[String, String], default ('Train', 'Test')
                Two strings that define the names in the dataset index of both splits.
        seed : Integer, default 0
                Defining the seed for reproducibility when making the random split. Only used when
                splitting with test_size.
        '''
        logging.info(f'{self.df.shape} - {self.name} - splitting into {names[0]} and {names[1]}')
        from sklearn.model_selection import train_test_split

        # If data has no dataset index yet, add one
        if not isinstance(self.df.index, pd.MultiIndex): 
            self.add_index('dataset', subset)
        
        if mask is not None:
            mask = eval(mask) if type(mask)==str else mask
        elif test_size is not None:
            all_indices = np.arange(len(self.df.loc[subset]))
            indices = train_test_split(all_indices,
                test_size=test_size, random_state=seed,
                stratify=self.df.loc[subset, stratify_col] if stratify_col else None)[0]
            mask = pd.Series(all_indices).isin(indices)
            
        # Take initial dataset index and overwrite with new dataset values
        self.df['dataset'] = self.df.index.get_level_values('dataset').astype(str)
        self.df.loc[subset, 'dataset'] = mask.replace(
            {True: names[0], False: names[1]}).values
        self.df['dataset'] = self.df['dataset'].astype('category')
        
        # Remove initial dataset index and replace with new one
        self.df.index = self.df.index.droplevel('dataset')
        self.df.set_index('dataset', append=True, inplace=True)
        self.df.index = self.df.index.swaplevel()
        
        # Print info on splits
        if stratify_col:
            logging.debug('\n'+tabulate(
                self.df.groupby([self.df.index.get_level_values('dataset'),
                                 self.df[stratify_col]]).size().unstack(),
                headers='keys', tablefmt='psql'))
        logging.debug('\n'+tabulate(
            self.df.index.get_level_values('dataset').value_counts().to_frame(),
            headers='keys', tablefmt='psql'))
        return self
            
    def balance(self, target, counts, stratified=None, name='Balanced', seed=0):
        '''
        Balance data by combining over- and undersampling according to a target variable. Often
        used to oversample low occurring target values and undersamply high occurring target values.
        
        Arguments
        ---------
        target : String
                Column to use for balancing the data.
        count : Dict[Any: Integer or Float]
                Defining the desired number or share of each value of the target column 
                in the resulting data.
        stratified : String, optional
                Define another column for which you want to retain its distribution.
        name : String, default 'Balanced'
                Arbitrary name for the resulting balanced data object.
        seed : Integer, default 0
                Defining the seed for reproducibility when over- & undersampling.
        '''
        logging.info(f'{self.df.shape} - {self.name} - balancing data: {counts}')
        assert target != stratified, "Stratified column cannot be the target"
        data = self.copy()

        samples = []
        if stratified:
            stratified_values = self.df[stratified].unique()
            n_values = len(stratified_values)
            for strat in stratified_values:
                temp = self.df[self.df[stratified]==strat]
                for value, count in counts.items():
                    newtemp = temp[temp[target]==value]
                    count /= n_values
                    sample = newtemp.sample(random_state=seed, n=int(count), replace=(count > len(newtemp)))
                    samples.append(sample)
        else:
            for value, count in counts.items():
                temp = self.df[self.df[target]==value]
                count = len(temp)*count if isinstance(count, float) else count
                if count<0.5: logging.warning(f'Trying to sample 0 rows for value {value}')
                sample = temp.sample(random_state=seed, n=int(count), replace=(count > len(temp)))
                samples.append(sample)
        data.df = pd.concat(samples, axis=0)
        if stratified: 
            logging.debug('\n' + tabulate(pd.crosstab(balanced[target], balanced[stratified]),
                                          headers='keys', tablefmt='psql'))
        return data
     
    def filter(self, mask, inplace=False):
        '''
        Filter rows of data object according to a boolean mask. 
        Either returns a filtered data object or filters in place.
        
        Arguments
        ---------
        mask : pandas.Series[Bool]
                Identifying which rows to filter.
        inplace : Bool, default False
                Whether to apply the filtering in place.
        '''
        logging.info(f'{self.df.shape} - {self.name} - filtering data')

        assert len(mask)==len(self.df), 'Mask hasn\'t same length as data'
        logging.info(f'Keeping {mask.sum()} of {len(mask)} lines after filtering.')
        if inplace: self.df = self.df[mask]
        else: return Data(self.df[mask].copy(), self.name, self.type)
        
    def select(self, subset=None, axis=0, name=None, drop=False):
        '''
        Selects from data object on index or columns.
        
        Arguments
        ---------
        subset : Any, whatever the index dtype is.
                Loc identifier for selecting from data object
        axis : Integer, default 0
                Which axis to select from. Rows with 0, columns with 1.
        name : String, optional
                Arbitrary and optional new name for the resulting selected data object.
        drop : Bool, default False
                Whether the index of the selected subset should be retained.                
        '''
        data = self.copy()
        if axis==0: data.df = data.df.loc[subset]
        elif axis==1: data.df = data.df.loc[:, subset]
        if drop: self.df.drop(subset, axis=axis, inplace=True)
        return data

    def sample(self, n, replace=True, seed=0):
        '''
        Sample random rows from data object.
        
        Arguments
        ---------
        n : Integer or Float
                How many rows (in case of integer) or what share of the dataset (in case of float)
                to sample. 
        replace : Bool, default True
                Whether rows are replaced and thus could be sampled more than once.
        seed : Integer, default 0
                Defining the seed for reproducibility when sampling.
        '''
        logging.info(f'{self.df.shape} - {self.name} - sampling data')
        n, frac = (int(n), None) if n > 1 else (None, float(n))
        data = Data(self.df.sample(random_state=seed, n=n, frac=frac,replace=replace), self.name, self.type)
        return data

    @classmethod
    def concat(cls, data_objects, axis=1, keys=None, name='Concatenated'):
        '''
        Class method. Concatenate data objects.
        
        Arguments
        ---------
        data_objects : List[lfd.Data] 
                List of data objects to concatenate. 
        axis : Integer, default 0
                Which axis to concatenate in. Rows with 0, columns with 1.
        keys: List[String] 
                Names to add as multiindex level after concatenation
        name : String, optional
                Arbitrary and optional new name for the resulting concatenated data object.
        '''
        data = Data(pd.concat([d.df for d in data_objects], axis=axis, keys=keys), name=name, type=self.type)
        return data

    def generate(self, n, name='Generated'):
        '''
        Generate a random data object based on an existing one by sampling randomly from its values.
        
        Arguments
        ---------
        n : Integer
                How many rows the new data object should contain. 
        name : String, optional
                Arbitrary and optional new name for the generated data object.
        '''
        generated = pd.DataFrame()
        for var in self.df:
            generated[var] = np.random.choice(self.df[var].unique(), size=n)
        return Data(generated, name=name, type=self.type)
    
    def add_noise(self, seed=0):
        '''
        Add 6 random variables to a data object, 3 categorical and 3 continuous ones.
        Often used to assess where random variables show up in a model's feature importance.
        
        Arguments
        ---------
        seed : Integer, default 0
                Defining the seed for reproducibility when sampling.
        '''
        logging.info(f'{self.df.shape} - {self.name} - adding random noise')
        np.random.seed(seed)
        n = self.df.shape[0]
        self.df['RANDOM1'] = np.random.choice([0, 1], p=[0.5, 0.5], size=n)
        self.df['RANDOM2'] = np.random.choice([0, 1], p=[0.1, 0.9], size=n)
        self.df['RANDOM3'] = np.random.choice([0, 1], p=[0.95, 0.05], size=n)
        self.df['RANDOM4'] = np.random.normal(loc=1000, scale=500, size=n)
        self.df['RANDOM5'] = np.random.normal(loc=-500, scale=100, size=n)
        self.df['RANDOM6'] = np.random.normal(loc=0, scale=1, size=n)

    def apply_functions(self, functions):
        '''
        Apply multiple functions on a data object, useful for custom logic.
        
        Arguments
        ---------
        functions : List[Function] 
                List of functions to apply on a data object.
        '''
        for function in functions:
            self.df = function(self.df)

    def save(self, file_name: str):
        '''
        Saves a data object.
        
        Arguments
        ---------
        file_name : String
                Path where the data should be written. Depending on the given extension,
                it saves a CSV or Parquet file.
        '''
        logging.info('Saving data')
        if os.path.isdir(file_name):
            with open(os.path.join(file_name, f'meta.json'), 'w') as f:
                json.dump({
                    'name': self.name,
                    'type': self.type,
                    'shape': list(self.df.shape),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'memory': get_memory(self)
                }, f, indent=4)
            self.df.to_parquet(os.path.join(file_name, f'data.parquet'))

        extension = file_name.split('.')[-1]
        if extension == 'parquet':
            self.df.to_parquet(file_name)
        elif extension == 'csv':
            self.df.to_csv(file_name)
        else:
            logging.error('Did not recognize extensions. Write to .csv or .parquet.')
        self.df.to_parquet(file_name)

    
    def load(self, data, name='Data', nrows=None):
        '''
        Loads a data object.
        
        Arguments
        ---------
        data : pandas.DataFrame, lfd.Data, or String
                Source for the data, type will be inferred.
        name : String, default 'Data'
                Arbitrary name for the data object.
        '''
        if isinstance(data, pd.DataFrame):
            self.df = data
            self.name = name
        elif isinstance(data, Data):
            self.df = data.df
            self.name = name
        elif isinstance(data, str):
            if os.path.isdir(data):
                if 'meta.json' in os.listdir(data):
                    with open(os.path.join(data, 'meta.json'), 'r') as f:
                        json_data = json.load(f)
                        self.name = json_data['name']
                else:
                    self.name = name
                data = os.path.join(data, 'data.parquet') if 'data.parquet' in os.listdir(data) else os.path.join(data, 'data.csv')
            extension = data.split('.')[-1]
            if extension == 'parquet':
                logging.info(f'Reading parquet file: {data}')
                self.df = pd.read_parquet(data)
            elif extension == 'csv':
                logging.info(f'Reading csv file: {data}')
                self.df = pd.read_csv(data, index_col=0, nrows=nrows)
            else:
                logging.error('Only parquet and csv are implemented for now. Did not read any data.')
        else:
            self.df = pd.DataFrame()
            self.name = name
        return self

    def analyse(self, broken_by=None, bins=5, to_excel=True, path='analysis.xlsx'):
        '''
        Analyse variables with respect to a given variable.

        Arguments
        ---------
        broken_by : String, optional
                Variable by which all other variables should be broken.
                Should not be a continuous variable.
        bins : Integer, default 5
                Number of bins to bin continuous variables into.
        to_excel : Bool, default True
                Whether to store the analysis in an excel file. If False, the dataframe 
                will be returned.
        path : String, default 'analysis.xlsx'
                Path to excel file to store the analysis. Only used if to_excel=True.
        '''
        logging.info(f'Generating analysis for {broken_by}')
        analysis = analyse(self.df, broken_by, bins)
        if to_excel:
            with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
                analysis.to_excel(writer, sheet_name='Analysis')
                writer.save()
            logging.info(f'Analysis written to {path}')
        else: return analysis

    def copy(self):
        '''
        Create a hard copy of a data object.
        
        Arguments
        ---------
        '''
        return deepcopy(self)

    def add_index(self, index, value):
        ''''''
        self.df[index] = value
        self.df.set_index(index, append=True, inplace=True)
        self.df.index = self.df.index.swaplevel()

    def set_dtypes(self):
        '''
        Sets the categorical dtype for categorical variables, and downsize numeric variables.
        
        Arguments
        ---------
        '''
        self.df = set_dtypes(self.df)

    def get_memory(self):
        '''
        Return a string with information on the size of the object.
        
        Arguments
        ---------
        '''
        return get_memory(self.df)

    def _summary(self):
        '''
        Returns a pandas Series with meta information.
        '''
        return pd.Series(dict(Name = self.name, Shape = self.df.shape, Type = self.type))
        
    def __repr__(self):
        print(f'This is a DATA object. Shape is {self.df.shape}.\n')
        display(pd.concat((self.df.head(3), self.df.tail(3))))
        return ''


def analyse(df, broken_by=None, bins=5, bins_method='quantile'): 
    if bins_method=='quantile': func = pd.qcut
    elif bins_method=='distance': func = pd.cut
    analyses, variables = [], []
    if broken_by:
        broken_by_series = df[broken_by]
        if broken_by_series.dtype.kind in 'iufc': broken_by_series = func(broken_by_series, bins, duplicates='drop')
    for variable in df.columns:
        if variable==broken_by: continue
        var_series = df[variable]
        if var_series.dtype.kind in 'iufc': var_series = func(var_series, bins, duplicates='drop')
        analysis = analyse_one_feature(var_series, broken_by_series if broken_by else None)
        analyses.append(analysis)
        variables.append(variable)
    analyses = pd.concat(analyses, keys=variables) if analyses else pd.DataFrame()
    return analyses
    

def analyse_one_feature(variable, broken_by=None):
    ''' Analyse one feature broken by a variable. Output is a pandas df. '''
    variable = variable.astype(str).fillna('MISSING').replace('nan|', 'MISSING')
    if broken_by is not None:
        broken_by = broken_by.astype(str).fillna('MISSING').fillna('MISSING').replace('nan|', 'MISSING')
        info = pd.concat((
            pd.crosstab(variable, broken_by, margins=True).iloc[:-1].add_prefix('count_'),
            pd.crosstab(variable, broken_by, margins=True, normalize=1).add_prefix('share1_'),
            pd.crosstab(variable, broken_by, margins=True, normalize=0).iloc[:-1].add_prefix('share2_'),
            ), axis=1)
    else:
        info = pd.concat((
            variable.value_counts().rename('counts'),
            variable.value_counts(normalize=True).rename('shares'),
            ), axis=1)
    return info.head(20)
