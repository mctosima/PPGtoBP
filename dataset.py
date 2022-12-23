import torch
import numpy as np
import glob

"""
We have 500 subjects, each subject has 3 column of signal.
Each subjects has different number of samples.


We want to create a dataset class that can be used by PyTorch Dataloader.
"""

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = glob.glob(root_dir + "/*.csv")
        self.files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # load csv into numpy
        data = np.genfromtxt(self.files[idx], delimiter=',')
        data = data.T
        # get label from filename
        # label = int(self.files[idx].split('/')[-1].split('_')[0])
        # convert to tensor
        data = torch.from_numpy(data).float()
        # label = torch.tensor(label)
        # apply transform
        if self.transform:
            data = self.transform(data)
        return data