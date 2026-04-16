import os
import sys

from training.train import train_model
from evaluation.evaluate import evaluate_model
from deployment.export_onnx import export_to_onnx
from deployment.export_tflite import convert_onnx_to_tflite
from app.inference import run_wearable_inference

def main():
    print("=========================================================")
    print("EFFICIENT LONG-SEQUENCE ECG CLASSIFICATION USING MAMBA")
    print("=========================================================")
    
    print("\n[STEP 1] Data Processing & Model Training")
    # For a quick demo test, max_records=5 limits MIT-BIH records.
    # Set max_records=None to use the entire dataset for full robustness.
    train_model(epochs=5, max_records=5)
    
    print("\n[STEP 2] Model Evaluation & Metrics")
    evaluate_model(max_records=5)
    
    print("\n[STEP 3] Exporting to Deployment Format (ONNX)")
    export_to_onnx()
    
    print("\n[STEP 4] Edge Quantization (TFLite)")
    convert_onnx_to_tflite()
    
    print("\n[STEP 5] Wearable Deployment Demonstration")
    # run_wearable_inference()
    import torch
    import numpy as np
    import pandas as pd
    from utils.dataset import get_dataloaders
    print("Fetching 1 Normal and 1 Arrhythmia sample from Dataset for Verification...")
    _, test_loader, _ = get_dataloaders(max_records=5)
    normal_beat, arr_beat = None, None
    for inputs, targets in test_loader:
        for i in range(len(targets)):
            if targets[i] == 0 and normal_beat is None:
                normal_beat = inputs[i].detach().cpu().numpy()
            elif targets[i] == 1 and arr_beat is None:
                arr_beat = inputs[i].detach().cpu().numpy()
        if normal_beat is not None and arr_beat is not None:
            break
            
    pd.DataFrame(normal_beat.reshape(1, 187)).to_csv('normal_test.csv', index=False, header=False)
    pd.DataFrame(arr_beat.reshape(1, 187)).to_csv('arrhythmia_test.csv', index=False, header=False)
    
    print("\n--- TESTING NORMAL SAMPLE ---")
    run_wearable_inference(input_csv='normal_test.csv')
    
    print("\n--- TESTING ARRHYTHMIA SAMPLE ---")
    run_wearable_inference(input_csv='arrhythmia_test.csv')
    
    print("\n PROJECT PIPELINE COMPLETED.")

if __name__ == '__main__':
    main()
