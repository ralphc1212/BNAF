import numpy as np
import torch

class data_lognormal:

    def __init__(self, location):
        # with open(location+'/lognormal_100.out', 'r') as f:
        #     lines = f.readlines()

        x = np.loadtxt(location+'/lognormal_100.out', dtype=np.float64)

        self.all = torch.from_numpy(x).unsqueeze(1).double()


        # del lines
        # f.close()
