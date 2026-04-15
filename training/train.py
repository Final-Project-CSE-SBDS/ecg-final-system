import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add project root to sys.path to resolve imports cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mamba_model import MambaECGClassifier
from utils.dataset import get_dataloaders

def train_model(epochs=10, batch_size=64, lr=1e-3, data_dir='./data', save_path='./saved_models/best_mamba.pth', max_records=None):
    print("Initializing DataLoaders...")
    train_loader, test_loader, class_weights = get_dataloaders(data_dir=data_dir, batch_size=batch_size, max_records=max_records)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize PyTorch Mamba Model
    model = MambaECGClassifier(input_dim=1, num_classes=2, d_model=64, num_layers=2).to(device)
    
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    best_acc = 0.0
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        train_loss = running_loss / total
        train_acc = 100 * correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f">>> Best model saved tightly with Validation Accuracy: {best_acc:.2f}%")

    print("Training Completed.")
    return model

if __name__ == '__main__':
    # Running directly trains the model
    # Limiting to 5 records for demonstration purposes so it trains fast.
    # To train on whole dataset, set max_records = None
    train_model(epochs=5, max_records=5)
