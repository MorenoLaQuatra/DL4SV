import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from transformers import AutoImageProcessor

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yaml_config_manager import load_config
from data.dataset import CustomDataset
from models.model import get_model
from utils.exp_manager import ExperimentManager

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, exp_manager):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        # Handle dict structure
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    # 1. Load Configuration
    config = load_config()
    
    # 2. Setup Experiment Manager
    exp_manager = ExperimentManager(config)
    
    # 3. Setup Device
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 4. Data Preparation
    # Use a standard pre-trained processor to handle resizing/normalization consistently
    print("Initializing HuggingFace Image Processor...")
    # Using microsoft/resnet-50 processor as a standard baseline for 224x224 normalization
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    
    train_dataset = CustomDataset(config, processor=processor, split='train')
    # Assuming the dataset class handles splitting if 'val'/ 'test' aren't explicit on Hub
    val_dataset = CustomDataset(config, processor=processor, split='val')

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=True, 
        num_workers=config.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=False, 
        num_workers=config.data.num_workers
    )

    # 5. Model Setup
    model = get_model(config).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr)

    # 6. Training Loop
    os.makedirs(config.training.save_dir, exist_ok=True)
    best_val_acc = 0.0

    print("Starting training...")
    for epoch in range(config.training.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, exp_manager)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{config.training.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Log metrics
        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }
        exp_manager.log_metrics(metrics, step=epoch)

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(config.training.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    exp_manager.end()

if __name__ == '__main__':
    main()
