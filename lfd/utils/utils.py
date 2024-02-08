import pandas as pd
import logging


def set_dtypes(data):
    initial_memory = get_memory(data)
    for catvar in data.select_dtypes(['object', 'string']).columns:
        data[catvar] = data[catvar].astype('category')
        data[catvar].cat.set_categories(data[catvar].cat.categories.astype(str))

    for intvar in data.select_dtypes('integer').columns:
        data[intvar] = pd.to_numeric(data[intvar], downcast='integer')

    for floatvar in data.select_dtypes('float').columns:
        data[floatvar] = pd.to_numeric(data[floatvar], downcast='float')
    logging.info(f'Set data types: went from {initial_memory} Mb to {get_memory(data)} Mb')
    return data


def get_memory(df): 
    return round((df.memory_usage(deep=True)/1024**2).sum(), 2)
