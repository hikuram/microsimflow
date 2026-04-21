import numpy as np

# --- Initialize shared random number generator ---
rng = np.random.default_rng()


def set_random_seed(seed):
    """Reinitialize the shared generator by fixing the seed externally."""
    global rng
    rng = np.random.default_rng(seed)


from .background import *
from .fill_deform import *
from .postprocess import *
