# Advanced ML Models for Accident Analysis

This directory contains three advanced machine learning modules for accident prediction and analysis.

## 📁 Available Modules

### 1. **TabTransformer** (`tab_transformer.py`)
**Purpose:** Deep learning model using transformer architecture for tabular data

**Features:**
- Learned embeddings for categorical features
- Multi-head self-attention mechanism
- Processes categorical and numerical features separately
- State-of-the-art architecture for tabular classification

**Use Case:** Predict collision types with better feature interaction modeling than traditional ML

**How to Run:**
```bash
python models/tab_transformer.py
```

**Key Components:**
- `TabTransformer`: Main model class with attention layers
- `AccidentTabTransformer`: Wrapper for training and prediction
- Multi-head attention for feature interactions
- Embedding layers for categorical features

---

### 2. **Explainable AI** (`explainable_ai.py`)
**Purpose:** Make model predictions interpretable using SHAP values

**Features:**
- SHAP (SHapley Additive exPlanations) for feature importance
- Global feature importance visualization
- Individual prediction explanations (waterfall plots)
- Feature dependence plots

**Use Case:** Understand which features contribute most to predictions

**How to Run:**
```bash
python models/explainable_ai.py
```

**Outputs:**
- `shap_global_summary.png` - Overall feature importance
- `shap_hour_dependence.png` - How hour affects predictions
- `feature_importance_shap.csv` - Ranked feature importance

**Key Methods:**
- `compute_shap_values()` - Calculate SHAP values
- `plot_global_summary()` - Feature importance plot
- `plot_dependence()` - Feature interaction plot
- `plot_waterfall()` - Single prediction explanation

---

### 3. **LSTM Forecasting** (`lstm_forecasting.py`)
**Purpose:** Time-series forecasting of daily accident counts

**Features:**
- LSTM (Long Short-Term Memory) neural network
- Predicts future accident counts based on historical patterns
- Handles sequential dependencies in time-series data
- 7-day ahead forecasting

**Use Case:** Predict how many accidents will occur in the next week

**How to Run:**
```bash
python models/lstm_forecasting.py
```

**Key Components:**
- `AccidentLSTM`: LSTM model architecture
- `AccidentForecaster`: Training and prediction wrapper
- Time-series data preparation
- Rolling window forecasting

**Outputs:**
- `lstm_forecaster.pth` - Trained LSTM model
- Next 7 days accident count predictions

---

## 🚀 Quick Start Guide

### Prerequisites
```bash
pip install torch shap matplotlib
```

### 1. Train TabTransformer
```python
from models.tab_transformer import AccidentTabTransformer

# Initialize
transformer = AccidentTabTransformer('data/model_ready.csv')

# Prepare data
X_cat, X_num, y, cat_dims = transformer.load_and_prepare_data()

# Train
transformer.train(X_cat, X_num, y, cat_dims, epochs=50)

# Predict
prediction, probs, attention = transformer.predict(
    categorical_data={'lum': 1, 'atm': 1, ...},
    numerical_data={'hour': 14, 'num_users': 2}
)
```

### 2. Explain Model Predictions
```python
from models.explainable_ai import AccidentXAI

# Initialize
xai = AccidentXAI(
    model_path='models/xgb_nopca_multitarget.pkl',
    data_path='data/model_ready.csv'
)

# Load and compute
xai.load_model_and_data()
xai.compute_shap_values(sample_size=1000)

# Visualize
xai.plot_global_summary(save_path='shap_summary.png')
xai.plot_dependence('hour', interaction_feature='lum')

# Get importance
importance_df = xai.get_feature_importance()
```

### 3. Forecast Accidents
```python
from models.lstm_forecasting import AccidentForecaster

# Initialize
forecaster = AccidentForecaster(
    data_path='data/cleaned_accidents.csv',
    sequence_length=30
)

# Prepare time-series
daily_counts = forecaster.prepare_time_series_data()

# Create sequences and train
X_train, y_train, X_test, y_test = forecaster.create_sequences(
    daily_counts['accident_count'].values
)
forecaster.train_model(X_train, y_train, X_test, y_test, epochs=100)

# Forecast next week
last_30_days = daily_counts['accident_count'].values[-30:]
next_week = forecaster.forecast_next_week(last_30_days)
print(f"Next week predictions: {next_week}")
```

---

## 📊 Model Comparison

| Model | Type | Best For | Training Time | Interpretability |
|-------|------|----------|---------------|------------------|
| **Random Forest** | Tree Ensemble | Baseline, Fast | Fast | High |
| **XGBoost** | Gradient Boosting | Best Accuracy | Medium | Medium |
| **TabTransformer** | Deep Learning | Feature Interactions | Slow | Low |
| **LSTM** | Recurrent NN | Time-Series | Medium | Low |

---

## 🎯 When to Use Each Model

### Use **TabTransformer** when:
- You have large datasets (>10k samples)
- Complex feature interactions are important
- You want state-of-the-art performance
- GPU is available for training

### Use **Explainable AI** when:
- You need to explain predictions to stakeholders
- Understanding feature importance is critical
- Building trust in model decisions
- Regulatory compliance requires interpretability

### Use **LSTM Forecasting** when:
- Predicting future accident counts
- Temporal patterns are important
- Planning resource allocation
- Analyzing seasonal trends

---

## 📈 Integration with Dashboard

To integrate these models into your Streamlit dashboard:

### 1. Add TabTransformer to Prediction Page
```python
# In pages/5_🔮_Prediction.py
from models.tab_transformer import AccidentTabTransformer

# Load model
transformer = AccidentTabTransformer('data/model_ready.csv')
transformer.load_model('models/tab_transformer_best.pth', cat_dims, num_classes)

# Add to model selection dropdown
```

### 2. Add Explainability Page
Create `pages/7_🔍_Explainability.py`:
```python
import streamlit as st
from models.explainable_ai import AccidentXAI

st.title("Model Explainability")

# Load XAI
xai = AccidentXAI(model_path, data_path)
xai.load_model_and_data()
xai.compute_shap_values()

# Display plots
st.pyplot(xai.plot_global_summary())
```

### 3. Add Forecasting Page
Create `pages/8_📅_Forecasting.py`:
```python
import streamlit as st
from models.lstm_forecasting import AccidentForecaster

st.title("Accident Forecasting")

# Load forecaster
forecaster = AccidentForecaster(data_path, sequence_length=30)
forecaster.load_model('models/lstm_forecaster.pth')

# Forecast and display
next_week = forecaster.forecast_next_week(last_sequence)
st.line_chart(next_week)
```

---

## 🔧 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
transformer.train(..., batch_size=64)  # Instead of 128

# Or use CPU
device = torch.device('cpu')
```

### SHAP Takes Too Long
```python
# Reduce sample size
xai.compute_shap_values(sample_size=500)  # Instead of 1000
```

### LSTM Not Converging
```python
# Increase epochs or adjust learning rate
forecaster.train_model(..., epochs=200, learning_rate=0.0001)
```

---

## 📚 References

- **TabTransformer**: [Huang et al., 2020](https://arxiv.org/abs/2012.06678)
- **SHAP**: [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
- **LSTM**: [Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)

---

## 🤝 Contributing

To add new models:
1. Create a new `.py` file in `models/` directory
2. Follow the same structure (class-based with `main()` function)
3. Add documentation to this README
4. Test with sample data

---

**Last Updated:** January 2026
**Maintainer:** Your Team
