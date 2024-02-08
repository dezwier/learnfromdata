from .general import Model
from .xgboost import Xgboost
from .regression import Regression
from .neuralnet import NeuralNet
from .decisiontree import DecisionTree
from .isotonic import Isotonic
from .gaussianmixture import GaussianMixture

from enum import Enum
class ModelEnum(Enum):
    model = Model
    regression = Regression
    xgboost = Xgboost
    neuralnet = NeuralNet
    decisiontree = DecisionTree
    isotonic = Isotonic
    gaussianmixture = GaussianMixture
