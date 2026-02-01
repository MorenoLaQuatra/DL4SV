import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    """
    A simple CNN baseline.
    """
    def __init__(self, config):
        super(CustomCNN, self).__init__()
        num_classes = config.model.num_classes
        hidden_dim = config.model.hidden_dim
        dropout = config.model.dropout
        
        # Example architecture for 224x224 images
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 56
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # Force 4x4 spatial output
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CustomTransformer(nn.Module):
    """
    Placeholder for a Transformer model (Vision or Audio).
    """
    def __init__(self, config):
        super(CustomTransformer, self).__init__()
        # TODO: Implement Transformer architecture
        # You might want to use nn.TransformerEncoder or generic attention blocks
        pass

    def forward(self, x):
        pass

def get_model(config):
    if config.model.name == "custom_cnn":
        return CustomCNN(config)
    elif config.model.name == "transformer":
        return CustomTransformer(config)
    else:
        raise ValueError(f"Unknown model name: {config.model.name}")