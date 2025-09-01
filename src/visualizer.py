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
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                num_params = sum(p.numel() for p in module.parameters())
                total_params += num_params
                print(f"  {name}: {module} | Parameters: {num_params:,}")
        
        print(f"Total {model_name} Parameters: {total_params:,}")
        print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    def visualize_architecture_diagram(self):
        """Create manual architecture diagrams that actually work"""
        # Get layer information for both networks
        gen_layers = self._get_layer_info(self.generator)
        disc_layers = self._get_layer_info(self.discriminator)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot Generator Architecture
        if gen_layers:
            gen_names = [f"{name}\n({layer_type})" for name, layer_type, _ in gen_layers]
            gen_params = [params for _, _, params in gen_layers]
            
            bars1 = ax1.barh(range(len(gen_layers)), gen_params, color='skyblue')
            ax1.set_yticks(range(len(gen_layers)))
            ax1.set_yticklabels(gen_names, fontsize=9)
            ax1.set_xlabel('Number of Parameters')
            ax1.set_title('Generator Architecture', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add parameter counts on bars
            for i, bar in enumerate(bars1):
                width = bar.get_width()
                ax1.text(width + max(gen_params)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{gen_params[i]:,}', ha='left', va='center', fontsize=8)
        
        # Plot Discriminator Architecture
        if disc_layers:
            disc_names = [f"{name}\n({layer_type})" for name, layer_type, _ in disc_layers]
            disc_params = [params for _, _, params in disc_layers]
            
            bars2 = ax2.barh(range(len(disc_layers)), disc_params, color='lightcoral')
            ax2.set_yticks(range(len(disc_layers)))
            ax2.set_yticklabels(disc_names, fontsize=9)
            ax2.set_xlabel('Number of Parameters')
            ax2.set_title('Discriminator Architecture', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add parameter counts on bars
            for i, bar in enumerate(bars2):
                width = bar.get_width()
                ax2.text(width + max(disc_params)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{disc_params[i]:,}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Network architecture diagram saved as 'network_architecture.png'")
    
    def _get_layer_info(self, model):
        """Extract layer information from model"""
        layers = []
        for name, module in model.named_modules():
            # Skip empty modules and containers
            if len(list(module.children())) == 0 and not isinstance(module, nn.Sequential):
                layer_type = module.__class__.__name__
                num_params = sum(p.numel() for p in module.parameters())
                
                # Get input/output sizes for linear layers
                if isinstance(module, nn.Linear):
                    layer_info = f"Linear({module.in_features}→{module.out_features})"
                else:
                    layer_info = layer_type
                
                layers.append((name, layer_info, num_params))
        
        return layers
    
    def visualize_network_flow(self):
        """Visualize the data flow through the network"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Generator flow
        gen_flow = self._simulate_forward_pass(self.generator, "generator")
        self._plot_network_flow(ax1, gen_flow, "Generator Data Flow")
        
        # Discriminator flow
        disc_flow = self._simulate_forward_pass(self.discriminator, "discriminator")
        self._plot_network_flow(ax2, disc_flow, "Discriminator Data Flow")
        
        plt.tight_layout()
        plt.savefig('network_data_flow.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _simulate_forward_pass(self, model, model_type):
        """Simulate forward pass to get layer outputs"""
        model.eval()
        with torch.no_grad():
            if model_type == "generator":
                # Create dummy input for generator
                dummy_input = torch.randn(1, self.latent_dim).to(self.device)
                dummy_label = torch.tensor([0]).to(self.device)
                output = model(dummy_input, dummy_label)
                return output.shape[1]  # Return output size
            else:
                # Create dummy input for discriminator
                dummy_input = torch.randn(1, self.image_dim).to(self.device)
                dummy_label = torch.tensor([0]).to(self.device)
                output = model(dummy_input, dummy_label)
                return output.shape[1]  # Return output size
    
    def _plot_network_flow(self, ax, final_output_size, title):
        """Plot network flow diagram"""
        # Simple representation - you can enhance this based on your actual architecture
        layers = ['Input', 'Hidden 1', 'Hidden 2', 'Hidden 3', 'Output']
        sizes = [100, 256, 256, 256, final_output_size]  # Adjust based on your architecture
        
        y_pos = range(len(layers))
        ax.barh(y_pos, sizes, color='lightblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layers)
        ax.set_xlabel('Layer Size')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add size labels
        for i, size in enumerate(sizes):
            ax.text(size + max(sizes)*0.01, i, f'{size}', va='center', fontweight='bold')
    
    def visualize_activations(self, epoch):
        """Visualize layer activations with proper hook handling"""
        self.generator.eval()
        self.discriminator.eval()
        
        # Generator activations
        gen_activations = self._capture_activations(self.generator, "generator")
        if gen_activations:
            self._plot_activations(gen_activations, "Generator", epoch)
        
        # Discriminator activations
        disc_activations = self._capture_activations(self.discriminator, "discriminator")
        if disc_activations:
            self._plot_activations(disc_activations, "Discriminator", epoch)
    
    def _capture_activations(self, model, model_type):
        """Capture activations from model layers"""
        activations = {}
        hooks = []
        
        def hook_fn(module, input, output, name):
            activations[name] = output.detach().cpu().numpy().flatten()[:100]  # First 100 values
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid)):
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
                hooks.append(hook)
        
        # Forward pass
        try:
            with torch.no_grad():
                if model_type == "generator":
                    dummy_z = torch.randn(1, self.latent_dim).to(self.device)
                    dummy_label = torch.tensor([0]).to(self.device)
                    _ = model(dummy_z, dummy_label)
                else:
                    dummy_input = torch.randn(1, self.image_dim).to(self.device)
                    dummy_label = torch.tensor([0]).to(self.device)
                    _ = model(dummy_input, dummy_label)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return activations
    
    def _plot_activations(self, activations, network_name, epoch):
        """Plot layer activations"""
        if not activations:
            return
            
        fig, axes = plt.subplots(len(activations), 1, figsize=(12, 3 * len(activations)))
        if len(activations) == 1:
            axes = [axes]
        
        for idx, (layer_name, activation) in enumerate(activations.items()):
            ax = axes[idx]
            values = activation[:50]  # First 50 values for clarity
            ax.bar(range(len(values)), values, alpha=0.7, color='blue')
            ax.set_title(f"{network_name} - {layer_name}", fontsize=10)
            ax.set_ylabel("Activation")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.writer.add_figure(f"{network_name}/Layer_Activations", fig, epoch)
        plt.close()
        print(f"{network_name} activations visualized for epoch {epoch}")
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()
