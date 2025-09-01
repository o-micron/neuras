import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

class NetworkVisualizer:
    def __init__(self, generator, discriminator, latent_dim, image_dim, device):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.device = device
        self.writer = SummaryWriter('runs/gan_visualization')
    
    def text_summary(self):
        """Print text summary of networks"""
        print("=" * 60)
        print("GENERATOR ARCHITECTURE")
        print("=" * 60)
        self._print_model_structure(self.generator, "Generator")
        
        print("\n" + "=" * 60)
        print("DISCRIMINATOR ARCHITECTURE")
        print("=" * 60)
        self._print_model_structure(self.discriminator, "Discriminator")
        
        # Also use torchsummary if available
        try:
            print("\n" + "=" * 60)
            print("DETAILED SUMMARY (torchsummary)")
            print("=" * 60)
            summary(self.generator, [(self.latent_dim,), (1,)], device=self.device.type)
            summary(self.discriminator, [(self.image_dim,), (1,)], device=self.device.type)
        except:
            print("torchsummary not available")
    
    def _print_model_structure(self, model, model_name):
        """Manual model structure printing"""
        print(f"{model_name} Structure:")
        total_params = 0
        for name, module in model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params
            print(f"  {name}: {module} | Parameters: {num_params:,}")
        
        print(f"Total {model_name} Parameters: {total_params:,}")
        print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    def visualize_architecture_diagram(self):
        """Create manual architecture diagrams"""
        # Generator diagram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Generator architecture
        gen_layers = []
        for name, module in self.generator.named_children():
            if hasattr(module, 'net'):
                for i, layer in enumerate(module.net):
                    if isinstance(layer, nn.Linear):
                        gen_layers.append(f"Linear({layer.in_features}→{layer.out_features})")
                    elif isinstance(layer, nn.ReLU):
                        gen_layers.append("ReLU")
                    elif isinstance(layer, nn.Tanh):
                        gen_layers.append("Tanh")
        
        ax1.barh(range(len(gen_layers)), [1]*len(gen_layers), tick_label=gen_layers)
        ax1.set_title('Generator Architecture')
        ax1.set_xlabel('Layers')
        
        # Discriminator architecture
        disc_layers = []
        for name, module in self.discriminator.named_children():
            if hasattr(module, 'net'):
                for i, layer in enumerate(module.net):
                    if isinstance(layer, nn.Linear):
                        disc_layers.append(f"Linear({layer.in_features}→{layer.out_features})")
                    elif isinstance(layer, nn.LeakyReLU):
                        disc_layers.append("LeakyReLU(0.2)")
                    elif isinstance(layer, nn.Sigmoid):
                        disc_layers.append("Sigmoid")
        
        ax2.barh(range(len(disc_layers)), [1]*len(disc_layers), tick_label=disc_layers)
        ax2.set_title('Discriminator Architecture')
        ax2.set_xlabel('Layers')
        
        plt.tight_layout()
        plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_activations(self, epoch):
        """Visualize layer activations"""
        self.generator.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            # Create test inputs
            test_z = torch.randn(1, self.latent_dim).to(self.device)
            test_label = torch.tensor([5]).to(self.device)
            test_image = torch.randn(1, self.image_dim).to(self.device)
            
            # Capture activations manually
            gen_activations = self._get_activations(self.generator, test_z, test_label)
            disc_activations = self._get_activations(self.discriminator, test_image, test_label)
            
            # Plot activations
            self._plot_layer_activations(gen_activations, "Generator", epoch)
            self._plot_layer_activations(disc_activations, "Discriminator", epoch)
    
    def _get_activations(self, model, *inputs):
        """Get activations from each layer"""
        activations = OrderedDict()
        x = inputs
        
        def hook(module, input, output, name):
            activations[name] = output.detach().cpu().numpy().flatten()
        
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid)):
                hook_handle = module.register_forward_hook(
                    lambda m, i, o, n=name: hook(m, i, o, n)
                )
                hooks.append(hook_handle)
        
        # Forward pass
        model(*inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def _plot_layer_activations(self, activations, network_name, epoch):
        """Plot layer activations"""
        fig, axes = plt.subplots(len(activations), 1, figsize=(12, 3 * len(activations)))
        if len(activations) == 1:
            axes = [axes]
        
        for idx, (layer_name, activation) in enumerate(activations.items()):
            ax = axes[idx]
            # Take first 50 values for visualization
            values = activation[:50]
            ax.bar(range(len(values)), values, alpha=0.7)
            ax.set_title(f"{network_name} - {layer_name}")
            ax.set_ylabel("Activation Value")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.writer.add_figure(f"{network_name}/Layer_Activations", fig, epoch)
        plt.close()
    
    def visualize_weight_distributions(self, epoch):
        """Visualize weight distributions without Graphviz"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Generator weights
        gen_weights = []
        gen_layer_names = []
        for name, param in self.generator.named_parameters():
            if 'weight' in name:
                gen_weights.append(param.cpu().detach().numpy().flatten())
                gen_layer_names.append(name)
        
        ax1.boxplot(gen_weights, labels=gen_layer_names)
        ax1.set_title('Generator Weight Distributions')
        ax1.set_ylabel('Weight Value')
        ax1.tick_params(axis='x', rotation=45)
        
        # Discriminator weights
        disc_weights = []
        disc_layer_names = []
        for name, param in self.discriminator.named_parameters():
            if 'weight' in name:
                disc_weights.append(param.cpu().detach().numpy().flatten())
                disc_layer_names.append(name)
        
        ax2.boxplot(disc_weights, labels=disc_layer_names)
        ax2.set_title('Discriminator Weight Distributions')
        ax2.set_ylabel('Weight Value')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.writer.add_figure("Weights/Distributions", fig, epoch)
        plt.close()
    
    def visualize_gradients(self, epoch):
        """Visualize gradient distributions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Generator gradients
        gen_grads = []
        gen_layer_names = []
        for name, param in self.generator.named_parameters():
            if param.grad is not None and 'weight' in name:
                gen_grads.append(param.grad.cpu().numpy().flatten())
                gen_layer_names.append(name)
        
        if gen_grads:
            ax1.boxplot(gen_grads, labels=gen_layer_names)
            ax1.set_title('Generator Gradient Distributions')
            ax1.set_ylabel('Gradient Value')
            ax1.tick_params(axis='x', rotation=45)
        
        # Discriminator gradients
        disc_grads = []
        disc_layer_names = []
        for name, param in self.discriminator.named_parameters():
            if param.grad is not None and 'weight' in name:
                disc_grads.append(param.grad.cpu().numpy().flatten())
                disc_layer_names.append(name)
        
        if disc_grads:
            ax2.boxplot(disc_grads, labels=disc_layer_names)
            ax2.set_title('Discriminator Gradient Distributions')
            ax2.set_ylabel('Gradient Value')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.writer.add_figure("Gradients/Distributions", fig, epoch)
        plt.close()
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()
