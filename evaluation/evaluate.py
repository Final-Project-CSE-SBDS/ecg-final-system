import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mamba_model import MambaECGClassifier
from utils.dataset import get_dataloaders

def evaluate_model(model_path='./saved_models/best_mamba.pth', data_dir='./data', batch_size=64, max_records=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    _, test_loader, _ = get_dataloaders(data_dir=data_dir, batch_size=batch_size, max_records=max_records)
    
    # Initialize model and load weights
    model = MambaECGClassifier(num_classes=2)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please train the model first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("Running Inference over Test Set...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    # Calculate Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Linear-Time Mamba Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision                  : {precision:.4f}")
    print(f"Recall                     : {recall:.4f}")
    print(f"F1-Score                   : {f1:.4f}")
    
    print("\n--- Detailed Classification Report ---")
    classes = ['Normal', 'Arrhythmia']
    # Adjust target names if some classes are missing in test split
    unique_targets = np.unique(all_targets)
    target_names = [classes[i] for i in unique_targets]
    print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))
    
    # Generate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Mamba ECG Classifier Confusion Matrix')
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    print("Confusion matrix saved to 'results/confusion_matrix.png'.")
    # plt.show()

if __name__ == '__main__':
    evaluate_model()
