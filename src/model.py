import torch
import torch.nn as nn
import os

# Function to save model
def save_model(model_path, generator, discriminator, g_optimizer, d_optimizer, epoch, losses, latent_dim, hidden_dim, image_dim, num_classes):
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
def load_model(device, model_path, generator, discriminator, g_optimizer, d_optimizer):
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
