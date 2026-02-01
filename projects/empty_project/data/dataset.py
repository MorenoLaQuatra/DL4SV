import torch
from torch.utils.data import Dataset
import os

# Placeholder imports - uncomment as needed
# from PIL import Image # For vision
# import torchaudio # For audio

class CustomDataset(Dataset):
    """
    Template for a custom dataset.
    Adapt this class to load images or audio files based on the project requirements.
    """
    def __init__(self, config, processor=None, split='train'):
        """
        Args:
            config: Configuration object containing data parameters.
            processor: Optional processor (e.g., HuggingFace feature extractor/tokenizer).
            split (str): 'train', 'val', or 'test'. Used to filter data.
        """
        self.config = config
        self.processor = processor
        self.split = split
        self.data_dir = config.data.path
        
        self.samples = [] 
        
        # --- PLACEHOLDER LOGIC ---
        # 1. Scan directory for files
        # 2. Filter by split (e.g., first 80% for train)
        # 3. Store file paths and labels in self.samples
        
        # Hints:
        # if config.data.modality == 'vision':
        #     # Scan for images (.jpg, .png)
        #     pass
        # elif config.data.modality == 'audio':
        #     # Scan for audio (.wav, .mp3)
        #     pass
        
        # For now, generating dummy data to make the template runnable
        self.dummy_size = 100 if split == 'train' else 20
        print(f"Initialized {split} dataset with {self.dummy_size} dummy samples.")

    def __len__(self):
        # return len(self.samples)
        return self.dummy_size

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: Dictionary containing input tensors and labels.
                  e.g., {'pixel_values': ..., 'labels': ...} for HuggingFace
                  or    {'input': ..., 'target': ...} for generic PyTorch
        """
        
        # --- PLACEHOLDER LOGIC ---
        # file_path, label = self.samples[idx]
        
        # 1. Load Data
        # if self.config.data.modality == 'vision':
        #     # image = Image.open(file_path).convert("RGB")
        #     # if self.processor:
        #     #     inputs = self.processor(images=image, return_tensors="pt")
        #     #     # Return dict compatible with HF models or your custom model
        #     #     return {'pixel_values': inputs['pixel_values'].squeeze(), 'labels': torch.tensor(label)}
        #     pass
            
        # elif self.config.data.modality == 'audio':
        #     # waveform, sr = torchaudio.load(file_path)
        #     # if self.processor:
        #     #     inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt")
        #     #     return {'input_values': inputs['input_values'].squeeze(), 'labels': torch.tensor(label)}
        #     pass

        # 2. Return dummy data (Replace this!)
        if self.config.data.modality == 'vision':
            # Dummy image: (3, 224, 224)
            return {'input': torch.randn(3, 224, 224), 'target': torch.tensor(0)}
        elif self.config.data.modality == 'audio':
            # Dummy audio: (1, 16000) - 1 second at 16kHz
            return {'input': torch.randn(1, 16000), 'target': torch.tensor(0)}
        
        return {'input': torch.randn(10), 'target': torch.tensor(0)}