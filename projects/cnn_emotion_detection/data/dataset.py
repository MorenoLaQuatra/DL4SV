import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import torchvision.transforms as T

class CustomDataset(Dataset):
    """
    Dataset class for Facial Emotion Detection using Hugging Face Datasets.
    """
    def __init__(self, config, processor=None, split='train'):
        """
        Args:
            config: Configuration object.
            processor: HuggingFace ImageProcessor (e.g. from ViT or ResNet).
            split (str): 'train' or 'test'. 
                         Note: This specific dataset might only have 'train', so we might need to split manually.
        """
        self.config = config
        self.processor = processor
        self.split = split
        
        print(f"Loading dataset: {config.data.path}...")
        # Load the dataset from HF Hub
        # Since this dataset might just have a 'train' split, we'll load it and split if necessary
        try:
            # First try loading the requested split directly
            self.dataset = load_dataset(config.data.path, split=split)
        except:
            # If 'val' or 'test' is requested but not found, fallback to splitting 'train'
            full_dataset = load_dataset(config.data.path, split='train')
            
            # Simple manual split (deterministic based on seed)
            split_idx = int(len(full_dataset) * config.data.split_ratio)
            
            # This is a bit simplistic for 'train'/'val'/'test' logic but serves the template
            if split == 'train':
                self.dataset = full_dataset.select(range(split_idx))
            else: # val or test
                self.dataset = full_dataset.select(range(split_idx, len(full_dataset)))
        
        print(f"Loaded {len(self.dataset)} samples for split: {split}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {'input': tensor, 'target': tensor}
        """
        item = self.dataset[idx]
        image = item['image'].convert("RGB") # Ensure RGB
        label = item['label']

        if self.processor:
            # The processor returns a dict with 'pixel_values'
            # We assume standard HF processor output
            processed = self.processor(images=image, return_tensors="pt")
            input_tensor = processed['pixel_values'].squeeze(0) # Remove batch dim added by processor
        else:
            # Fallback if no processor provided (should not happen if configured correctly)
            # Just a basic transform
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
            input_tensor = transform(image)

        return {
            'input': input_tensor,
            'target': torch.tensor(label, dtype=torch.long)
        }
