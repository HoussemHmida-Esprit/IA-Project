"""
LSTM Time-Series Forecasting Module
Predicts future accident counts based on historical data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib


class AccidentTimeSeriesDataset(Dataset):
    """PyTorch Dataset for time-series accident data"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class AccidentLSTM(nn.Module):
    """LSTM model for accident count prediction"""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(AccidentLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class AccidentForecaster:
    """Time-series forecasting for accident counts"""
    
    def __init__(self, data_path: str, sequence_length: int = 30):
        """
        Initialize forecaster
        
        Args:
            data_path: Path to cleaned accidents data
            sequence_length: Number of past days to use for prediction
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.daily_counts = None
        
    def prepare_time_series_data(self):
        """
        Convert transactional accident data to time-series format
        
        Returns:
            DataFrame with daily accident counts
        """
        print("Preparing time-series data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Create date column
        if 'date' not in df.columns:
            # Combine year, month, day
            if all(col in df.columns for col in ['year', 'month', 'day']):
                df['date'] = pd.to_datetime(
                    df[['year', 'month', 'day']].rename(
                        columns={'year': 'year', 'month': 'month', 'day': 'day'}
                    )
                )
            else:
                raise ValueError("Cannot create date column. Need year, month, day columns.")
        else:
            df['date'] = pd.to_datetime(df['date'])
        
        # Aggregate by date
        daily_counts = df.groupby('date').size().reset_index(name='accident_count')
        
        # Fill missing dates with 0
        date_range = pd.date_range(
            start=daily_counts['date'].min(),
            end=daily_counts['date'].max(),
            freq='D'
        )
        daily_counts = daily_counts.set_index('date').reindex(date_range, fill_value=0)
        daily_counts = daily_counts.reset_index().rename(columns={'index': 'date'})
        
        # Add time features
        daily_counts['day_of_week'] = daily_counts['date'].dt.dayofweek
        daily_counts['month'] = daily_counts['date'].dt.month
        daily_counts['day_of_month'] = daily_counts['date'].dt.day
        daily_counts['is_weekend'] = (daily_counts['day_of_week'] >= 5).astype(int)
        
        self.daily_counts = daily_counts
        
        print(f"✓ Created time-series data: {len(daily_counts)} days")
        print(f"  Date range: {daily_counts['date'].min()} to {daily_counts['date'].max()}")
        print(f"  Total accidents: {daily_counts['accident_count'].sum()}")
        print(f"  Avg accidents/day: {daily_counts['accident_count'].mean():.2f}")
        
        return daily_counts
    
    def create_sequences(self, data, train_split=0.8):
        """
        Create sequences for LSTM training
        
        Args:
            data: Array of accident counts
            train_split: Fraction of data for training
        
        Returns:
            X_train, y_train, X_test, y_test
        """
        # Normalize data
        data_normalized = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(data_normalized) - self.sequence_length):
            seq = data_normalized[i:i + self.sequence_length]
            target = data_normalized[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Split into train/test
        split_idx = int(len(sequences) * train_split)
        
        X_train = sequences[:split_idx]
        y_train = targets[:split_idx]
        X_test = sequences[split_idx:]
        y_test = targets[split_idx:]
        
        print(f"✓ Created sequences:")
        print(f"  Training: {len(X_train)} sequences")
        print(f"  Testing: {len(X_test)} sequences")
        print(f"  Sequence length: {self.sequence_length} days")
        
        return X_train, y_train, X_test, y_test
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train LSTM model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        print(f"\nTraining LSTM model on {self.device}...")
        
        # Create datasets
        train_dataset = AccidentTimeSeriesDataset(X_train, y_train)
        val_dataset = AccidentTimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = AccidentLSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(sequences)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        print("✓ Training complete!")
        
        return train_losses, val_losses
    
    def predict(self, sequence):
        """
        Make prediction for next day
        
        Args:
            sequence: Last N days of accident counts
        
        Returns:
            Predicted accident count for next day
        """
        self.model.eval()
        
        # Normalize sequence
        sequence_normalized = self.scaler.transform(sequence.reshape(-1, 1))
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction_normalized = self.model(sequence_tensor).cpu().numpy()
        
        # Denormalize
        prediction = self.scaler.inverse_transform(prediction_normalized)
        
        return max(0, int(prediction[0, 0]))
    
    def forecast_next_week(self, last_sequence):
        """
        Forecast accident counts for next 7 days
        
        Args:
            last_sequence: Last N days of accident counts
        
        Returns:
            Array of 7 predicted counts
        """
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(7):
            # Predict next day
            next_pred = self.predict(current_sequence)
            predictions.append(next_pred)
            
            # Update sequence (rolling window)
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        return np.array(predictions)
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'sequence_length': self.sequence_length
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = AccidentLSTM().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.sequence_length = checkpoint['sequence_length']
        
        print(f"✓ Model loaded from {path}")


def main():
    """Example usage"""
    # Initialize forecaster
    forecaster = AccidentForecaster(
        data_path='data/cleaned_accidents.csv',
        sequence_length=30
    )
    
    # Prepare data
    daily_counts = forecaster.prepare_time_series_data()
    
    # Create sequences
    accident_counts = daily_counts['accident_count'].values
    X_train, y_train, X_test, y_test = forecaster.create_sequences(accident_counts)
    
    # Train model
    train_losses, val_losses = forecaster.train_model(
        X_train, y_train, X_test, y_test,
        epochs=100,
        batch_size=32
    )
    
    # Save model
    forecaster.save_model('models/lstm_forecaster.pth')
    
    # Forecast next week
    last_sequence = accident_counts[-30:]
    next_week = forecaster.forecast_next_week(last_sequence)
    
    print("\nNext week forecast:")
    for i, count in enumerate(next_week, 1):
        print(f"  Day {i}: {count} accidents")
    
    print("\n✓ LSTM forecasting complete!")


if __name__ == "__main__":
    main()
