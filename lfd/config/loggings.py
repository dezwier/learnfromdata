import os
import logging
import logging.config
import sys
from io import StringIO


def _get_logging(stdout=True, stdout_level='INFO', log_path=None, log_level='DEBUG'):
    '''
    This function returns a configuration for for the logging module.
    This is used for the Python logging on any ml pipeline. It contains
    3 handlers, one for stderr, stdout and a file.
    '''
    config = {
        'version': 1,
        'formatters': {
            'my_formatter': {
                'format': "[%(asctime)s] [%(levelname)-8s] -- %(message)s",
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console_stderr': {
                'class': 'logging.StreamHandler',
                'level': 'WARNING',
                'formatter': 'my_formatter',
                'stream': sys.stderr
            },
        },
        'root': {
            # In general, this should be kept at 'NOTSET' to ensure it does
            # not interfere with the log levels set for each handler
            'level': 'NOTSET',
            'handlers': ['console_stderr']
        },
    }
    if stdout:
        config['handlers']['console_stdout'] = {
                'class': 'logging.StreamHandler',
                'level': stdout_level.upper(),
                'formatter': 'my_formatter',
                'stream': sys.stdout
            }
        config['root']['handlers'].append('console_stdout')
    if log_path:
        config['handlers']['file'] = {
                'class': 'logging.FileHandler',
                'level': log_level.upper(),
                'formatter': 'my_formatter',
                'filename': log_path,
                'encoding': 'utf8'
            }
        config['root']['handlers'].append('file')
    return config

def set_logging(stdout=True, stdout_level='INFO', log_dir=None, log_level='DEBUG'):
    '''
    Configure logging, to standard output and/or a log file for any use of lfd.
    
    Arguments
    ---------
    stdout : Bool, default True
            Whether to log to standard output.
    stdout_level : String, default 'INFO' 
            Should be in (in 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG').
            What level should be logged to standard output. Level and above is logged.
    log_dir : String, default None
            Path to logfile. If given, all logging is written to this file. If parent 
            directory doesn't exist, it is created.
    log_level : String, default 'DEBUG' 
            Should be in (in 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG').
            What level should be logged to the logfile. Level and above is logged.
    '''
    log_path = os.path.join(log_dir, 'logs.log') if log_dir else None
    if log_dir and not os.path.exists(log_dir): os.mkdir(log_dir)
    logging.config.dictConfig(_get_logging(stdout, stdout_level, log_path, log_level))