import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ECGDatasetProcessor:
    """
    A standalone module to process ECG datasets (like MIT-BIH).
    """
    def __init__(self, window_size=200, test_size=0.2, random_state=42):
        self.window_size = window_size
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        
    def load_data(self, filepath):
        """Loads ECG dataset from a CSV file."""
        logger.info(f"Loading data from {filepath}...")
        try:
            # Assuming no header for standard MIT-BIH raw CSVs, or automatically inferring
            df = pd.read_csv(filepath, header=None)
            logger.info(f"Data loaded successfully. Original shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            raise
            
    def preprocess_data(self, df):
        """Preprocesses the ECG data: fills missing, pads/truncates, normalizes, encodes labels."""
        logger.info("Starting preprocessing...")
        
        # 1. Handle missing values (forward fill then backward fill)
        if df.isnull().values.any():
            logger.info("Handling missing values...")
            df = df.fillna(method='ffill').fillna(method='bfill')
            
        # Assuming the last column is the label
        X_raw = df.iloc[:, :-1].values
        y_raw = df.iloc[:, -1].values
        
        # 2. Encode labels (Normal = 0, Arrhythmia = 1)
        # Note: In MIT-BIH, 0 is usually Normal, and 1,2,3,4 are abnormal.
        logger.info("Encoding labels (Normal=0, Arrhythmia=1)...")
        y_encoded = np.where(y_raw == 0.0, 0, 1)
        
        # 3. Segment/pad into fixed windows
        num_samples = X_raw.shape[0]
        seq_len = X_raw.shape[1]
        
        logger.info(f"Adjusting sequence length from {seq_len} to {self.window_size}...")
        X_adjusted = np.zeros((num_samples, self.window_size))
        
        if seq_len > self.window_size:
            # Truncate
            X_adjusted = X_raw[:, :self.window_size]
        else:
            # Pad with zeros
            X_adjusted[:, :seq_len] = X_raw
            
        # 4. Normalize signals
        logger.info("Normalizing signals using MinMaxScaler...")
        # Scaler fits on 2D array, which is precisely our shape (num_samples, window_size)
        X_normalized = self.scaler.fit_transform(X_adjusted.T).T # Normalize per sample or across dataset? 
        # Usually we normalize per dataset, let's just fit_transform directly on the flattened structure or transpose to scale features 
        # Wait, standardizing per signal (min 0 max 1 per sequence) is better for ECG:
        X_normalized = np.zeros_like(X_adjusted)
        for i in range(num_samples):
            val_min = np.min(X_adjusted[i])
            val_max = np.max(X_adjusted[i])
            if val_max - val_min > 0:
                X_normalized[i] = (X_adjusted[i] - val_min) / (val_max - val_min)
            else:
                X_normalized[i] = X_adjusted[i]
                
        logger.info(f"Preprocessing completed. Processed signal shape: {X_normalized.shape}")
        return X_raw, X_normalized, y_encoded
        
    def plot_sample(self, X_raw, X_processed, sample_idx=0, save_path="processed/sample_plot.png"):
        """Plots a raw vs processed ECG signal."""
        logger.info(f"Plotting sample index {sample_idx}...")
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(X_raw[sample_idx], color='blue')
        plt.title('Raw ECG Signal')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        
        plt.subplot(2, 1, 2)
        plt.plot(X_processed[sample_idx], color='green')
        plt.title(f'Processed ECG Signal (Normalized & Padded/Truncated to {self.window_size})')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
        # plt.show() # Uncomment to show interactively
        
    def split_and_save(self, X, y, output_dir="processed"):
        """Splits the data and saves to CSV."""
        logger.info("Splitting dataset into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine X and y
        train_data = np.column_stack((X_train, y_train))
        test_data = np.column_stack((X_test, y_test))
        
        # Save to CSV
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")
        
        logger.info(f"Saving train data to {train_path}...")
        pd.DataFrame(train_data).to_csv(train_path, index=False, header=False)
        
        logger.info(f"Saving test data to {test_path}...")
        pd.DataFrame(test_data).to_csv(test_path, index=False, header=False)
        
        logger.info("Printing Dataset Summary:")
        print("="*40)
        print("DATASET MODULE SUMMARY")
        print("="*40)
        print(f"Total samples processed: {X.shape[0]}")
        print(f"Processed feature shape: {X.shape}")
        
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(['Normal (0)', 'Arrhythmia (1)'], counts))
        print(f"Overall Class Distribution: {class_dist}")
        
        print(f"Train samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print("="*40)

def main():
    # Ensure a sample dataset exists to test the module
    # In a real scenario, this would be updated to 'data/mitbih_train.csv' or similar
    input_csv = "sample_ecg.csv" 
    
    if not os.path.exists(input_csv):
        logger.error(f"Input file {input_csv} not found. Please provide a valid dataset.")
        return
        
    processor = ECGDatasetProcessor(window_size=200)
    
    try:
        # 1. Load Data
        df = processor.load_data(input_csv)
        
        # 2. Preprocess Data
        X_raw, X_processed, y_processed = processor.preprocess_data(df)
        
        # 3. Visualization
        processor.plot_sample(X_raw, X_processed, sample_idx=0)
        
        # 4. Split and Save
        processor.split_and_save(X_processed, y_processed)
        
        logger.info("Dataset Processing Module executed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
