from .general import Plotter
from .model import PlotterModel
from .visual import PlotterVisual
from .graph import PlotterGraph
from .board import run_app

from enum import Enum
class PlotterEnum(Enum):
    plotter = Plotter
    modelplotter = PlotterModel
    visualplotter = PlotterVisual
    graphplotter = PlotterGraph