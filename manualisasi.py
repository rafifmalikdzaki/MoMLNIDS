from skripsi_code.utils.loss import EntropyLoss
import torch
import torch.nn as nn
from torchsummary import summary
from skripsi_code.model.FeatureExtractor import DGFeatExt
from skripsi_code.model.Discriminator import DomainDiscriminator
from skripsi_code.model.Classifier import ClassifierANN

class MoMLDNIDS(nn.Module):
    def __init__(
        self,
        input_nodes,
        hidden_nodes,
        classifier_nodes,
        num_domains,
        num_class,
        single_layer=True,
    ):
        super().__init__()
        self.FeatureExtractorLayer = DGFeatExt(input_nodes, hidden_nodes)
        self.DomainClassifier = DomainDiscriminator(
            hidden_nodes[-1], num_domains, classifier_nodes
        )
        self.LabelClassifier = ClassifierANN(
            hidden_nodes[-1],
            num_class,
            classifier_nodes if single_layer else classifier_nodes,
        )
        
        # Dictionary to store intermediate outputs
        self.intermediate_outputs = {}

        # Register forward hooks
        self._register_hooks()

    def _register_hooks(self):
        # Register hooks for all submodules, including layers within Sequential containers
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Sequential):
                # For Sequential modules, register hooks on individual sub-layers
                for sub_name, sub_layer in layer.named_children():
                    sub_layer.register_forward_hook(self._get_intermediate_output(f"{name}.{sub_name}", sub_layer))
            else:
                # Register hook for non-Sequential layers directly
                layer.register_forward_hook(self._get_intermediate_output(name, layer))

    def _get_intermediate_output(self, layer_name, layer):
        def hook(module, input, output):
            layer_type = type(layer).__name__
            self.intermediate_outputs[layer_name] = output
            print(f"\nLayer: {layer_name} ({layer_type})")
            print(f"Output: {output}")

            # Print weights, biases, and gradients (if applicable)
            if hasattr(module, 'weight') and module.weight is not None:
                print(f"Weight: {module.weight}")
                if module.weight.grad is not None:
                    print(f"Weight Gradient: {module.weight.grad}")
            if hasattr(module, 'bias') and module.bias is not None:
                print(f"Bias: {module.bias}")
                if module.bias.grad is not None:
                    print(f"Bias Gradient: {module.bias.grad}")
        return hook

    def forward(self, x):
        print("Input:", x)
        invariant_features = self.FeatureExtractorLayer(x)
        output_class = self.LabelClassifier(invariant_features)
        output_domain = self.DomainClassifier(invariant_features)
        return output_class, output_domain


if __name__ == "__main__":
    x = torch.randn(5, 16).to("cuda")  # Batch of 5, input size of 16
    model = MoMLDNIDS(
        input_nodes=x.size(dim=1),
        hidden_nodes=[8, 8, 8],
        classifier_nodes=[8, 8],
        num_domains=3,
        num_class=2,
        single_layer=True,
    ).to("cuda")
    
    # Define loss functions
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    entropy_criterion = EntropyLoss()
    
    # Placeholder target labels for classification and domain
    class_labels = torch.randint(0, 2, (5,)).to("cuda")  # Example: binary classification labels
    domain_labels = torch.randint(0, 3, (5,)).to("cuda")  # Example: 3 possible domains

    # Forward pass with intermediate results printed
    output_class, output_domain = model(x)

    # Compute the loss
    class_loss = class_criterion(output_class, class_labels)
    domain_loss = domain_criterion(output_domain, domain_labels)
    entropy_loss = entropy_criterion(output_class)
    loss = class_loss + domain_loss + entropy_loss
    print(f"\nClass Loss: {class_loss.item()}, Domain Loss: {domain_loss.item()}, Entropy Loss: {entropy_loss.item()}, Total Loss: {loss.item()}")

    # Backward pass to calculate gradients
    loss.backward()

    # Print the parameters and gradients
    for name, param in model.named_parameters():
        print(f"\nLayer: {name}")
        print(f"Parameter: {param.data}")
        if param.grad is not None:
            print(f"Gradient: {param.grad}")
        print("-" * 50)

    # Model summary
    summary(model, (x.size(dim=1),))

