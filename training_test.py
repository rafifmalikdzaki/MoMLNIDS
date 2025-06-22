#!/usr/bin/env python3
"""
Minimal training test to verify that the training loop can start on a small sample.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from skripsi_code.model.MoMLNIDS import MoMLDNIDS
from skripsi_code.utils.loss import EntropyLoss
import numpy as np

def create_dummy_data(batch_size=32, feature_dim=43, num_classes=2, num_domains=3):
    """Create dummy data for testing."""
    # Create dummy features
    features = torch.randn(batch_size, feature_dim)
    
    # Create dummy labels
    class_labels = torch.randint(0, num_classes, (batch_size,))
    domain_labels = torch.randint(0, num_domains, (batch_size,))
    
    return features, class_labels, domain_labels

def test_training_step():
    """Test that a basic training step can be performed."""
    print("Testing basic training step...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create model
        model = MoMLDNIDS(
            input_nodes=43,
            hidden_nodes=[64, 32, 16, 10],
            classifier_nodes=[64, 32, 16],
            num_domains=3,
            num_class=2,
        ).to(device)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create loss functions
        class_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()
        entropy_loss = EntropyLoss()
        
        # Create dummy data
        features, class_labels, domain_labels = create_dummy_data()
        features = features.to(device)
        class_labels = class_labels.to(device)
        domain_labels = domain_labels.to(device)
        
        print(f"✓ Created dummy data - Features: {features.shape}, Class labels: {class_labels.shape}, Domain labels: {domain_labels.shape}")
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        class_output, domain_output = model(features)
        
        # Calculate losses
        class_loss = class_criterion(class_output, class_labels)
        domain_loss = domain_criterion(domain_output, domain_labels)
        entropy_loss_val = entropy_loss(domain_output)
        
        # Combined loss
        total_loss = class_loss + domain_loss + 0.1 * entropy_loss_val
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful")
        print(f"  - Class loss: {class_loss.item():.4f}")
        print(f"  - Domain loss: {domain_loss.item():.4f}")
        print(f"  - Entropy loss: {entropy_loss_val.item():.4f}")
        print(f"  - Total loss: {total_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_training_steps():
    """Test multiple training steps to ensure stability."""
    print("\nTesting multiple training steps...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        model = MoMLDNIDS(
            input_nodes=43,
            hidden_nodes=[64, 32, 16, 10],
            classifier_nodes=[64, 32, 16],
            num_domains=3,
            num_class=2,
        ).to(device)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create loss functions
        class_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()
        entropy_loss = EntropyLoss()
        
        losses = []
        num_steps = 5
        
        for step in range(num_steps):
            # Create dummy data for each step
            features, class_labels, domain_labels = create_dummy_data()
            features = features.to(device)
            class_labels = class_labels.to(device)
            domain_labels = domain_labels.to(device)
            
            # Training step
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            class_output, domain_output = model(features)
            
            # Calculate losses
            class_loss = class_criterion(class_output, class_labels)
            domain_loss = domain_criterion(domain_output, domain_labels)
            entropy_loss_val = entropy_loss(domain_output)
            
            # Combined loss
            total_loss = class_loss + domain_loss + 0.1 * entropy_loss_val
            losses.append(total_loss.item())
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            print(f"  Step {step + 1}/{num_steps}: Loss = {total_loss.item():.4f}")
        
        print(f"✓ Multiple training steps successful")
        print(f"  - Average loss: {np.mean(losses):.4f}")
        print(f"  - Loss std: {np.std(losses):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Multiple training steps failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run training tests."""
    print("🔥 Running training loop tests...")
    print("=" * 50)
    
    success_count = 0
    total_tests = 2
    
    if test_training_step():
        success_count += 1
    
    if test_multiple_training_steps():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Training test results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All training tests passed! Training loop can start successfully.")
        return 0
    else:
        print("❌ Some training tests failed.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
