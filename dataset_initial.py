import os
#from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class RocksMap(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        #image = np.load(img_path)
        mask = np.load(mask_path).astype(np.float32)
        #print(f"mask:{ mask.shape}")
        image = torch.tensor(np.load(img_path)).view(1, mask.shape[0], mask.shape[1])
        #print(f"image:{image.shape}")
        #mask = torch.tensor(mask,dtype=torch.float).view(1, mask.shape[0], mask.shape[1])
        mask = torch.tensor(mask, dtype=torch.float )


        # if self.transform is not None:
        #     augmentations = self.transform(image=image, mask=mask)
        #     image = augmentations["image"]
        #     mask = augmentations["mask"]

        return image, mask


