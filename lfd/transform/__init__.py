from .general import Transformer
from .uniselector import UniSelector
from .biselector import BiSelector
from .imputer import Imputer
from .encoder import Encoder
from .expander import Expander
from .standardizer import Standardizer
from .binner import Binner

from enum import Enum
class TransformEnum(Enum):
    transformer = Transformer
    uniselector = UniSelector
    binner = Binner
    imputer = Imputer
    encoder = Encoder
    biselector = BiSelector
    expander = Expander
    standardizer = Standardizer