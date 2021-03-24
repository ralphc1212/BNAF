import numpy as np
import torch

class data_lognormal:

    def __init__(self, location):
        with open(location+'/lognormal.out', 'r') as f:
            lines = f.readlines()

        self.all = torch.from_numpy(np.array([float(x) for x in lines])).float()

        del lines
        f.close()
