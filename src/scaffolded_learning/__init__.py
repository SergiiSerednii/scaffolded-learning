__version__ = "0.1.0"

from .nets.mlp_multiclass import ScaffoldedMLPMultiClass
from .nets.mlp_binary import ScaffoldedMLPBinary
from .nets.vgg import ScaffoldedVGG
from .nets.gpt import ScaffoldedGPT

__all__ = ["ScaffoldedMLPMultiClass", "ScaffoldedMLPBinary", "ScaffoldedVGG", "ScaffoldedGPT"]