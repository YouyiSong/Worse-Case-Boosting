import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class DataSet(torch.utils.data.Dataset):
    def __init__(self, set, path, IDs, labels, width, height, num_label, phase):
        super(DataSet, self).__init__()
        self.set = set
        self.path = path
        self.IDs = IDs
        self.labels = labels
        self.num_label = num_label
        if phase == 'Training':
            self.transform = transforms.Compose([transforms.Resize([width, height], interpolation=2),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.RandomVerticalFlip(p=0.5),
                                                 transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.Resize([width, height], interpolation=2),
                                                 transforms.ToTensor()])
   
def __len__(self):
        return len(self.IDs)

    
    def __getitem__(self, idx):
        img = Image.open(self.path + self.set + '\\' + self.IDs[idx])
        label = self.labels[idx]
        target = torch.zeros(self.num_label, dtype=torch.float)
        target[label] = 1
        img = self.transform(img)
        return img, target
