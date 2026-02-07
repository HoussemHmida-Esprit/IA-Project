# 🔄 RNN Model Location & Explanation

## Where is the RNN Model?

### ✅ RNN Model = LSTM (Objective 2)

**RNN (Recurrent Neural Network)** is the general category.
**LSTM (Long Short-Term Memory)** is a specific type of RNN that we implemented.

---

## 📁 File Locations

### 1. LSTM Implementation
**File:** `models/lstm_forecasting.py`

This contains:
- `AccidentLSTM` class - The RNN/LSTM model
- Training functions
- Prediction functions
- Data preprocessing for time-series

### 2. Trained Model
**File:** `models/lstm_forecaster.pth`

This is the saved trained model (PyTorch checkpoint).

### 3. Dashboard Page
**File:** `pages/8_📅_Forecasting.py`

This is where users can:
- Train new LSTM models
- Load existing models
- Make 7-day forecasts
- Visualize predictions

---

## 🧠 What is LSTM/RNN?

### RNN vs LSTM

```
RNN (Recurrent Neural Network)
├── Simple RNN (basic, has vanishing gradient problem)
├── LSTM (Long Short-Term Memory) ← What you implemented
└── GRU (Gated Recurrent Unit)
```

### Why LSTM for Your Project?

**Problem:** Predict how many accidents will happen next week

**Why RNN/LSTM?**
- Accidents have **temporal patterns** (time-based)
- Monday ≠ Saturday (different patterns)
- Summer ≠ Winter (seasonal trends)
- Need to remember **past patterns** to predict future

**Traditional ML can't do this:**
```
Random Forest: Predicts one accident at a time
❌ Can't predict "How many accidents tomorrow?"

LSTM: Looks at past 30 days → Predicts next day
✅ Can forecast future accident counts
```

---

## 🏗️ LSTM Architecture (Your Implementation)

### Data Transformation

```python
# Original Data (Transactional)
Row 1: Accident on 2024-01-15 at 14:30
Row 2: Accident on 2024-01-15 at 16:45
Row 3: Accident on 2024-01-16 at 09:20
...

# Transformed to Time-Series
Date         | Accident Count
─────────────┼───────────────
2024-01-15   | 245
2024-01-16   | 198
2024-01-17   | 223
2024-01-18   | 267
...

# Create Sequences (30-day windows)
Input: [245, 198, 223, 267, ..., 189]  (30 days)
Output: 201                              (day 31)
```

### Model Architecture

```
Input: 30 days of accident counts
  ↓
┌─────────────────────────────────┐
│   LSTM Layer 1 (64 units)       │  ← Learns short-term patterns
│   - Remembers recent trends     │
│   - Dropout 0.2                 │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│   LSTM Layer 2 (64 units)       │  ← Learns long-term patterns
│   - Remembers weekly cycles     │
│   - Dropout 0.2                 │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│   Fully Connected Layer         │  ← Makes final prediction
│   64 → 1                        │
└─────────────────────────────────┘
  ↓
Output: Predicted accident count for tomorrow
```

### How LSTM "Remembers"

```python
# LSTM has 3 gates (like memory management)

1. Forget Gate: "What should I forget?"
   - Forgets old patterns that don't matter anymore
   - Example: Forget accident spike from 3 weeks ago

2. Input Gate: "What should I remember?"
   - Remembers important new information
   - Example: Remember that weekends have fewer accidents

3. Output Gate: "What should I output?"
   - Decides what to use for prediction
   - Example: Use weekly pattern + recent trend
```

---

## 📊 LSTM vs TabTransformer

| Aspect | LSTM (RNN) | TabTransformer |
|--------|------------|----------------|
| **Purpose** | Time-series forecasting | Classification |
| **Input** | Sequence of numbers | Single record |
| **Output** | Future value | Category |
| **Task** | "How many accidents tomorrow?" | "What type of collision?" |
| **Data Type** | Temporal (time-based) | Tabular (features) |
| **Memory** | Sequential memory | Attention mechanism |

### Example Tasks

**LSTM (Time-Series):**
```python
# Input: Past 30 days
[245, 198, 223, 267, 189, 234, ...]

# Output: Next day prediction
Predicted: 212 accidents tomorrow
```

**TabTransformer (Classification):**
```python
# Input: Single accident features
{lum: 3, agg: 1, int: 2, hour: 18, ...}

# Output: Collision type
Predicted: "Side collision" (51% confidence)
```

---

## 🎯 Your LSTM Implementation Details

### Configuration
```python
AccidentLSTM(
    input_size=1,        # 1 feature (accident count)
    hidden_size=64,      # 64 LSTM units
    num_layers=2,        # 2 LSTM layers
    dropout=0.2,         # 20% dropout
    sequence_length=30   # Look at past 30 days
)
```

### Training
```python
Epochs: 100
Batch Size: 32
Optimizer: Adam (learning_rate=0.001)
Loss: MSE (Mean Squared Error)
Device: CPU
```

### Data Flow
```
Historical Data (2005-2024)
  ↓
Aggregate by day → Daily counts
  ↓
Create sequences → [30 days] → [next day]
  ↓
Normalize → Scale to [0, 1]
  ↓
Train LSTM → Learn patterns
  ↓
Predict → Forecast next 7 days
  ↓
Denormalize → Convert back to actual counts
```

---

## 🚀 How to Use LSTM in Dashboard

### Step 1: Navigate to Forecasting Page
```
Dashboard → 📅 Forecasting
```

### Step 2: Train or Load Model
```python
# Option A: Train new model
Click "Train New Model"
- Loads historical data
- Trains for 100 epochs
- Saves to models/lstm_forecaster.pth

# Option B: Load existing model
Automatically loads if lstm_forecaster.pth exists
```

### Step 3: Generate Forecast
```python
# Model predicts next 7 days
Today: 245 accidents
Tomorrow: 212 accidents (predicted)
Day 2: 198 accidents (predicted)
Day 3: 223 accidents (predicted)
...
Day 7: 189 accidents (predicted)
```

### Step 4: View Visualization
```
Chart shows:
- Historical data (blue line)
- Predictions (red line)
- Confidence intervals (shaded area)
```

---

## 🔬 Technical Comparison

### All Three Models Side-by-Side

| Model | Type | Input | Output | Use Case |
|-------|------|-------|--------|----------|
| **Random Forest** | Tree Ensemble | Features | Collision Type | Classification |
| **XGBoost** | Gradient Boosting | Features | Collision Type | Classification |
| **TabTransformer** | Transformer | Features | Collision Type | Classification |
| **LSTM** | RNN | Time Sequence | Future Count | Forecasting |

### Different Problems, Different Models

```
Problem 1: "What type of collision will this be?"
→ Use: TabTransformer / XGBoost / Random Forest
→ Input: [lum=3, agg=1, hour=18, ...]
→ Output: "Side collision"

Problem 2: "How many accidents next week?"
→ Use: LSTM (RNN)
→ Input: [245, 198, 223, 267, ...] (past 30 days)
→ Output: [212, 198, 223, ...] (next 7 days)
```

---

## 📈 LSTM Performance

### What It Learns

1. **Weekly Patterns**
   - Monday-Friday: More accidents (work commute)
   - Saturday-Sunday: Fewer accidents

2. **Seasonal Trends**
   - Summer: More accidents (more travel)
   - Winter: Fewer accidents (less travel)

3. **Holiday Effects**
   - Before holidays: Spike in accidents
   - During holidays: Drop in accidents

4. **Long-term Trends**
   - Overall increase/decrease over years
   - Policy changes impact

### Evaluation Metrics
```python
MSE (Mean Squared Error): How far off predictions are
MAE (Mean Absolute Error): Average prediction error
R² Score: How well model fits data (0-1, higher is better)
```

---

## 🎓 Key Differences Summary

### LSTM (RNN) - Time-Series Forecasting
```
Purpose: Predict FUTURE values
Input: Sequence of past values
Memory: Sequential (remembers order)
Output: Continuous number
Example: "212 accidents tomorrow"
```

### TabTransformer - Classification
```
Purpose: Classify CURRENT record
Input: Single record with features
Memory: Attention (learns relationships)
Output: Category label
Example: "Side collision"
```

### Both Use Deep Learning
```
LSTM: Recurrent connections (loops)
TabTransformer: Attention mechanism (no loops)

Both: Neural networks trained with backpropagation
```

---

## 📁 Quick Reference

### Files to Check

1. **LSTM Implementation**
   ```
   models/lstm_forecasting.py
   - AccidentLSTM class (lines 20-80)
   - Training function (lines 100-200)
   - Prediction function (lines 250-300)
   ```

2. **LSTM Dashboard**
   ```
   pages/8_📅_Forecasting.py
   - Training interface
   - Prediction visualization
   - Model loading
   ```

3. **Trained Model**
   ```
   models/lstm_forecaster.pth
   - Saved weights
   - Model configuration
   - Training history
   ```

---

## 🎯 Summary

**RNN Model Location:**
- ✅ Implementation: `models/lstm_forecasting.py`
- ✅ Trained Model: `models/lstm_forecaster.pth`
- ✅ Dashboard: `pages/8_📅_Forecasting.py`

**What It Does:**
- Predicts future accident counts (time-series forecasting)
- Uses past 30 days to predict next 7 days
- Learns weekly and seasonal patterns

**Difference from TabTransformer:**
- LSTM: Sequential data → Future prediction
- TabTransformer: Tabular data → Classification

Both are deep learning, but solve different problems! 🚀
