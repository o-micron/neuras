import torch
import torch.nn as nn

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, image_dim, num_classes):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def forward(self, z, labels):
        # Embed the labels
        label_embed = self.label_embedding(labels)
        
        # Concatenate noise and label embedding
        x = torch.cat([z, label_embed], dim=1)
        return self.net(x)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, image_dim, hidden_dim, num_classes):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.net = nn.Sequential(
            nn.Linear(image_dim + num_classes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1 (real/fake probability)
        )
    
    def forward(self, x, labels):
        # Embed the labels
        label_embed = self.label_embedding(labels)
        
        # Concatenate image and label embedding
        x = torch.cat([x, label_embed], dim=1)
        return self.net(x)
