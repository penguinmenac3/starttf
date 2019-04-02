import tensorflow as tf

NO_PARAMS = object()
hyperparams = None

# If version is 1.x use the fallback implementation of module.
if tf.__version__.startswith("1."):
    from starttf.modules.module import Module
else:
    from starttf.modules.module2 import Module


# Should be moved to extra lib
from starttf.modules.encoders import Encoder, MultiResolutionEncoder
from starttf.modules.tile_2d import Tile2D, InverseTile2D, UpsamplingFeaturePassthrough, FeaturePassthrough
from starttf.modules.loss import CompositeLoss