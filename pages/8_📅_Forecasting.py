"""
Time-Series Forecasting Page - LSTM Predictions

Predicts future accident counts based on historical patterns.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.lstm_forecasting import AccidentForecaster
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

st.set_page_config(
    page_title="Accident Forecasting",
    page_icon="📅",
    layout="wide"
)

st.title("Accident Forecasting")
st.markdown("*Predict future accident counts using LSTM neural networks*")

if not TORCH_AVAILABLE:
    st.error("""
    ⚠️ **PyTorch not installed**
    
    To use this feature, install PyTorch:
    ```bash
    pip install torch
    ```
    """)
    st.stop()

# Check if data exists
DATA_PATH = Path("data/cleaned_accidents.csv")
MODEL_PATH = Path("models/lstm_forecaster.pth")

if not DATA_PATH.exists():
    st.error(f"⚠️ Data file not found: {DATA_PATH}")
    st.stop()

# Sidebar settings
st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Mode",
    options=["View Existing Model", "Train New Model"],
    help="View predictions from existing model or train a new one"
)

sequence_length = st.sidebar.slider(
    "Sequence Length (days)",
    min_value=7,
    max_value=60,
    value=30,
    help="Number of past days to use for prediction"
)

# Main content
st.header("What is Time-Series Forecasting?")
st.markdown("""
**LSTM (Long Short-Term Memory)** networks are a type of neural network designed for sequential data.
They can learn patterns in historical accident data to predict future counts.

**Use Cases:**
- Resource planning (ambulances, police)
- Seasonal trend analysis
- Early warning systems
- Budget allocation
""")

st.divider()

# Initialize forecaster
@st.cache_resource
def init_forecaster(data_path, seq_length):
    """Initialize forecaster"""
    forecaster = AccidentForecaster(str(data_path), sequence_length=seq_length)
    return forecaster

forecaster = init_forecaster(DATA_PATH, sequence_length)

# Prepare time-series data
@st.cache_data
def load_time_series(_forecaster):
    """Load and prepare time-series data"""
    return _forecaster.prepare_time_series_data()

with st.spinner("Loading time-series data..."):
    try:
        daily_counts = load_time_series(forecaster)
        st.success(f"✅ Loaded {len(daily_counts)} days of data")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Display data overview
st.header("Historical Data Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Days",
        f"{len(daily_counts):,}"
    )

with col2:
    st.metric(
        "Date Range",
        f"{daily_counts['date'].min().strftime('%Y-%m-%d')} to {daily_counts['date'].max().strftime('%Y-%m-%d')}"
    )

with col3:
    st.metric(
        "Total Accidents",
        f"{daily_counts['accident_count'].sum():,}"
    )

with col4:
    st.metric(
        "Avg per Day",
        f"{daily_counts['accident_count'].mean():.1f}"
    )

# Plot historical data
st.subheader("Historical Accident Counts")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily_counts['date'],
    y=daily_counts['accident_count'],
    mode='lines',
    name='Daily Accidents',
    line=dict(color='#636EFA', width=1)
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Number of Accidents",
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Mode: View existing model or train new
if mode == "Train New Model":
    st.header("Train LSTM Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.number_input(
            "Training Epochs",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="More epochs = better training but slower"
        )
    
    with col2:
        batch_size = st.number_input(
            "Batch Size",
            min_value=16,
            max_value=128,
            value=32,
            step=16
        )
    
    if st.button("Start Training", type="primary"):
        with st.spinner("Training LSTM model... This may take several minutes."):
            try:
                # Create sequences
                accident_counts = daily_counts['accident_count'].values
                X_train, y_train, X_test, y_test = forecaster.create_sequences(
                    accident_counts,
                    train_split=0.8
                )
                
                # Train model
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                train_losses, val_losses = forecaster.train_model(
                    X_train, y_train, X_test, y_test,
                    epochs=epochs,
                    batch_size=batch_size
                )
                
                progress_bar.progress(100)
                status_text.success("✅ Training complete!")
                
                # Save model
                forecaster.save_model(str(MODEL_PATH))
                
                # Plot training history
                st.subheader("Training History")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=train_losses,
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    y=val_losses,
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    xaxis_title="Epoch",
                    yaxis_title="Loss (MSE)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during training: {e}")
                import traceback
                st.code(traceback.format_exc())

else:  # View existing model
    st.header("Forecast Future Accidents")
    
    if not MODEL_PATH.exists():
        st.warning("""
        ⚠️ **No trained model found**
        
        Please train a model first by selecting "Train New Model" in the sidebar.
        """)
        st.stop()
    
    # Load model
    with st.spinner("Loading trained model..."):
        try:
            forecaster.load_model(str(MODEL_PATH))
            st.success("✅ Model loaded successfully")
        except FileNotFoundError:
            st.error(f"Model file not found: {MODEL_PATH}")
            st.stop()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("💡 Try training a new model using 'Train New Model' mode in the sidebar.")
            st.stop()
    
    # Forecast settings
    forecast_days = st.slider(
        "Forecast Horizon (days)",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to forecast into the future"
    )
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                # Get last sequence
                accident_counts = daily_counts['accident_count'].values
                last_sequence = accident_counts[-sequence_length:]
                
                # Forecast
                if forecast_days <= 7:
                    predictions = forecaster.forecast_next_week(last_sequence)
                    predictions = predictions[:forecast_days]
                else:
                    # Extend forecast beyond 7 days
                    predictions = []
                    current_sequence = last_sequence.copy()
                    
                    for _ in range(forecast_days):
                        next_pred = forecaster.predict(current_sequence)
                        predictions.append(next_pred)
                        current_sequence = np.append(current_sequence[1:], next_pred)
                    
                    predictions = np.array(predictions)
                
                # Create forecast dates
                last_date = daily_counts['date'].max()
                forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                
                # Display forecast
                st.subheader("Forecast Results")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Predicted",
                        f"{int(predictions.sum()):,}"
                    )
                
                with col2:
                    st.metric(
                        "Avg per Day",
                        f"{predictions.mean():.1f}"
                    )
                
                with col3:
                    st.metric(
                        "Peak Day",
                        f"{int(predictions.max())}"
                    )
                
                # Forecast table
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Day': [d.strftime('%A') for d in forecast_dates],
                    'Predicted Accidents': predictions.astype(int)
                })
                
                st.dataframe(forecast_df, hide_index=True, use_container_width=True)
                
                # Visualization
                st.subheader("Forecast Visualization")
                
                # Show last 60 days + forecast
                recent_days = 60
                recent_data = daily_counts.tail(recent_days)
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=recent_data['date'],
                    y=recent_data['accident_count'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='#636EFA', width=2)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=predictions,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#EF553B', width=2, dash='dash'),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Accidents",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
                import traceback
                st.code(traceback.format_exc())

st.divider()

# Information
st.header("About LSTM Forecasting")
st.markdown("""
### How It Works

1. **Data Preparation**: Convert accident records to daily counts
2. **Sequence Creation**: Use past N days to predict next day
3. **LSTM Training**: Learn patterns in historical data
4. **Forecasting**: Predict future counts using learned patterns

### Model Architecture

- **Input**: Last 30 days of accident counts
- **LSTM Layers**: 2 layers with 64 hidden units
- **Output**: Predicted count for next day
- **Rolling Forecast**: Iteratively predict multiple days ahead

### Limitations

- Assumes patterns continue (no major changes)
- Cannot predict unprecedented events
- Accuracy decreases for longer horizons
- Weather and special events not included

### Best Practices

- Retrain model regularly with new data
- Use for short-term planning (1-7 days)
- Combine with domain knowledge
- Monitor actual vs predicted performance
""")
