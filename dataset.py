# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from config import batch_size

class ClimateDataset(Dataset):
    def __init__(self, lwa_paths, label_paths=None):
        self.lwa_paths = lwa_paths
        self.label_paths = label_paths
        
        # Only convert to tensor (0-1 scaling), NO resizing needed anymore
        self.lwa_transform = T.ToTensor()

    def __len__(self):
        return len(self.lwa_paths)

    def __getitem__(self, idx):
        # Load and convert LWA
        lwa_img = Image.open(self.lwa_paths[idx]).convert('L')
        input_tensor = self.lwa_transform(lwa_img)

        # Load and convert Labels
        if self.label_paths is not None:
            label_img = Image.open(self.label_paths[idx])
            
            # No resizing, just direct array to tensor conversion
            label_tensor = torch.from_numpy(np.array(label_img)).long()
            return input_tensor, label_tensor

        return input_tensor

def get_dataloader(lwa_paths, label_paths=None, is_train=True):
    dataset = ClimateDataset(lwa_paths, label_paths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4)

def inferloader(list_of_img_paths):
    dataset = ClimateDataset(list_of_img_paths)
    loader = DataLoader(dataset, batch_size=len(list_of_img_paths), shuffle=False)
    return next(iter(loader))