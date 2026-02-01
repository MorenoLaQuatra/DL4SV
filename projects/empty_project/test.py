import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yaml_config_manager import load_config
from data.dataset import CustomDataset
from models.model import get_model

def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    acc = correct / total
    return acc

def main():
    # 1. Load Configuration
    config = load_config()
    
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Data
    test_dataset = CustomDataset(config, split='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=False, 
        num_workers=config.data.num_workers
    )

    # 3. Model
    model = get_model(config).to(device)

    # 4. Load Checkpoint
    checkpoint_path = os.path.join(config.training.save_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Testing with random weights.")

    # 5. Evaluate
    print("Starting evaluation...")
    test_acc = test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()