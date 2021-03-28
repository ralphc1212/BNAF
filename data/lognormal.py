import numpy as np
import torch

class data_lognormal:

    def __init__(self, location):
        with open(location+'/lognormal_100.out', 'r') as f:
            lines = f.readlines()

        self.all = torch.from_numpy(np.array([float(x) for x in lines])).unsqueeze(1).float()

        print(np.unique(self.all))
        exit()
        del lines
        f.close()
