import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
latent_dim = 100
hidden_dim = 256
image_dim = 28 * 28  # MNIST images are 28x28
num_classes = 10  # Digits 0-9
batch_size = 64
epochs = 500
learning_rate = 0.0002
model_path = 'conditional_generator.pth'

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

# Initialize networks
generator = Generator(latent_dim, hidden_dim, image_dim, num_classes).to(device)
discriminator = Discriminator(image_dim, hidden_dim, num_classes).to(device)

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Loss function
criterion = nn.BCELoss()

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Function to save model
def save_model(generator, discriminator, g_optimizer, d_optimizer, epoch, losses):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_losses': losses['g_losses'],
        'd_losses': losses['d_losses'],
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'image_dim': image_dim
    }, model_path)
    print(f"Model saved at epoch {epoch}")

# Function to load model
def load_model():
    if os.path.exists(model_path):
        print("Loading saved model...")
        checkpoint = torch.load(model_path, map_location=device)
        
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        losses = {
            'g_losses': checkpoint['g_losses'],
            'd_losses': checkpoint['d_losses']
        }
        
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch, losses
    else:
        print("No saved model found. Starting training from scratch.")
        return 0, {'g_losses': [], 'd_losses': []}

# Training function
def train_gan(start_epoch, losses):
    g_losses = losses['g_losses']
    d_losses = losses['d_losses']
    
    for epoch in range(start_epoch, epochs):
        for i, (real_images, real_labels) in enumerate(train_loader):
            # Move to device
            real_images = real_images.view(-1, image_dim).to(device)
            real_labels = real_labels.to(device)
            batch_size = real_images.size(0)
            
            # Create labels
            real_targets = torch.ones(batch_size, 1).to(device)
            fake_targets = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real images
            real_outputs = discriminator(real_images, real_labels)
            d_loss_real = criterion(real_outputs, real_targets)
            
            # Fake images
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            fake_images = generator(z, fake_labels)
            fake_outputs = discriminator(fake_images.detach(), fake_labels)
            d_loss_fake = criterion(fake_outputs, fake_targets)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            # Generate fake images and try to fool discriminator
            fake_outputs = discriminator(fake_images, fake_labels)
            g_loss = criterion(fake_outputs, real_targets)  # Try to make discriminator think fake is real
            
            g_loss.backward()
            g_optimizer.step()
            
            # Store losses for plotting
            if i % 200 == 0:
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(train_loader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
        
        # Save model after each epoch
        current_losses = {'g_losses': g_losses, 'd_losses': d_losses}
        save_model(generator, discriminator, g_optimizer, d_optimizer, epoch, current_losses)
    
    return g_losses, d_losses

# Function to generate samples for specific digits
def generate_specific_digit(generator, digit, num_samples=16):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        # Create labels for the specific digit
        labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
        generated_images = generator(z, labels).cpu()
        generated_images = generated_images.view(-1, 28, 28)
        
        # Plot generated images
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated_images[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f'Digit: {digit}')
        plt.tight_layout()
        plt.show()

# Function to generate samples for all digits
def generate_all_digits(generator, num_samples_per_digit=5):
    generator.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(num_classes, num_samples_per_digit, figsize=(12, 15))
        
        for digit in range(num_classes):
            z = torch.randn(num_samples_per_digit, latent_dim).to(device)
            labels = torch.full((num_samples_per_digit,), digit, dtype=torch.long).to(device)
            generated_images = generator(z, labels).cpu()
            generated_images = generated_images.view(-1, 28, 28)
            
            for j in range(num_samples_per_digit):
                ax = axes[digit, j]
                ax.imshow(generated_images[j], cmap='gray')
                ax.axis('off')
                if j == 0:
                    ax.set_title(f'Digit: {digit}', fontsize=12)
        
        plt.tight_layout()
        plt.show()

# Main training logic
def main():
    # Check if we should load a saved model or start fresh
    start_epoch, losses = load_model()
    
    # Only train if we haven't reached the total epochs yet
    if start_epoch < epochs:
        print("Starting/resuming GAN training...")
        g_losses, d_losses = train_gan(start_epoch, losses)
    else:
        print("Training already completed. Loading final losses.")
        g_losses, d_losses = losses['g_losses'], losses['d_losses']
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.show()
    
    # Generate samples for all digits
    print("Generating samples for all digits...")
    generate_all_digits(generator)
    
    # Interactive part: Let user choose which digit to generate
    while True:
        try:
            user_input = input("\nEnter a digit (0-9) to generate, or 'q' to quit: ")
            
            if user_input.lower() == 'q':
                print("Goodbye!")
                break
            
            digit = int(user_input)
            if 0 <= digit <= 9:
                print(f"Generating images of digit {digit}...")
                generate_specific_digit(generator, digit)
            else:
                print("Please enter a digit between 0 and 9.")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

# Run the main function
if __name__ == "__main__":
    main()