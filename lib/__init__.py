import random
import torch
import numpy as np


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 0
print("Setting random seed {}".format(seed))
fix_seeds(seed)
