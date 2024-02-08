'''
This module collects methods for configuration purposes. 
'''
from .params import get_params
from .loggings import set_logging
set_logging()

from .docs import generate_doc, get_lfd_doc
