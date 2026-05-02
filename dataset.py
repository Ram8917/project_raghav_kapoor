# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from config import resize_x, resize_y, batch_size

class LWADataset(Dataset):
    def __init__(self, lwa_paths, temp_paths=None):
        self.lwa_paths = lwa_paths
        self.temp_paths = temp_paths
        
        # Transform for LWA images (resizes and converts to float tensor [0.0, 1.0])
        self.lwa_transform = T.Compose([
            T.Resize((resize_y, resize_x)), # Note: PyTorch expects (height, width)
            T.ToTensor()
        ])
        
        # Transform for label masks (Nearest neighbor to keep 0-5 integers intact)
        self.mask_transform = T.Resize((resize_y, resize_x), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.lwa_paths)

    def __getitem__(self, idx):
        # Load LWA PNG image (1-channel grayscale)
        lwa_img = Image.open(self.lwa_paths[idx]).convert('L')
        image = self.lwa_transform(lwa_img)

        # Load Temperature PNG image (Integer classes 0-5)
        if self.temp_paths is not None:
            temp_img = Image.open(self.temp_paths[idx])
            temp_img = self.mask_transform(temp_img)
            
            # Convert directly to Long tensor for CrossEntropyLoss (do NOT use ToTensor here)
            mask = torch.from_numpy(np.array(temp_img)).long()
            return image, mask
        
        return image

def get_dataloader(lwa_paths, temp_paths=None, is_train=True):
    dataset = LWADataset(lwa_paths, temp_paths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

def inferloader(list_of_img_paths):
    dataset = LWADataset(list_of_img_paths)
    loader = DataLoader(dataset, batch_size=len(list_of_img_paths), shuffle=False)
    return next(iter(loader))