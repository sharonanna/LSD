import os
#from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class RocksMap(Dataset):
    def __init__(self, image_dir, mask_dir, slope_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.slope_dir = slope_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        slope_path = os.path.join(self.slope_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        mask = np.load(mask_path).astype(np.float32)
        image = torch.tensor(np.load(img_path)).view(1, mask.shape[0], mask.shape[1])
        slope = torch.tensor(np.load(slope_path)).view(1, mask.shape[0], mask.shape[1])
        mask = torch.tensor(mask, dtype=torch.float )

        return image, mask, slope