# 🎯 Hyperparameter Optimization Guide

## Overview

This guide explains how to optimize all models in your project using Optuna, a state-of-the-art hyperparameter optimization framework.

---

## 🚀 Quick Start

### Step 1: Install Optuna
```bash
pip install optuna
```

### Step 2: Run Optimization
```bash
cd models
python hyperparameter_optimization.py
```

This will:
- Optimize Random Forest (30 trials, ~30 minutes)
- Optimize XGBoost (30 trials, ~45 minutes)
- Optimize TabTransformer (15 trials, ~2-3 hours)
- Generate comparison reports

### Step 3: Update Dashboard
```bash
python update_dashboard_with_optimized.py
```

### Step 4: Restart Dashboard
```bash
cd ..
streamlit run app.py
```

---

## 📊 What Gets Optimized?

### Random Forest
```python
Hyperparameters:
├── n_estimators: 50-300 (number of trees)
├── max_depth: 10-50 (tree depth)
├── min_samples_split: 2-20
├── min_samples_leaf: 1-10
├── max_features: ['sqrt', 'log2', None]
└── class_weight: ['balanced', 'balanced_subsample']

Search Space: ~1,000,000 combinations
Trials: 30 (samples best combinations)
Time: ~30 minutes
```

### XGBoost
```python
Hyperparameters:
├── n_estimators: 100-500 (number of boosting rounds)
├── learning_rate: 0.01-0.3 (step size)
├── max_depth: 3-15 (tree depth)
├── subsample: 0.6-1.0 (data sampling)
├── colsample_bytree: 0.6-1.0 (feature sampling)
├── reg_alpha: 0-1.0 (L1 regularization)
├── reg_lambda: 0-2.0 (L2 regularization)
├── min_child_weight: 1-10
└── gamma: 0-1.0 (minimum loss reduction)

Search Space: ~10,000,000 combinations
Trials: 30
Time: ~45 minutes
```

### TabTransformer
```python
Hyperparameters:
├── d_model: [32, 64, 128] (embedding dimension)
├── num_heads: [2, 4, 8] (attention heads)
├── num_layers: 2-4 (transformer blocks)
├── d_ff: [64, 128, 256] (feed-forward dimension)
├── dropout: 0.1-0.3 (regularization)
├── embedding_dim: [8, 16, 32] (categorical embeddings)
├── learning_rate: 0.0001-0.01 (optimizer step size)
└── batch_size: [64, 128, 256] (training batch size)

Search Space: ~100,000 combinations
Trials: 15 (fewer due to training time)
Time: ~2-3 hours
```

---

## 🔬 How Optuna Works

### 1. Bayesian Optimization
```
Traditional Grid Search:
├── Try ALL combinations (exhaustive)
├── Time: Days/weeks for large spaces
└── Inefficient

Optuna (Bayesian):
├── Try smart combinations (guided by previous results)
├── Time: Hours
├── Learns which areas are promising
└── Efficient
```

### 2. Trial Process
```
For each trial:
1. Optuna suggests hyperparameters
2. Train model with those parameters
3. Evaluate on validation set
4. Report score back to Optuna
5. Optuna learns and suggests better parameters
6. Repeat

After all trials:
→ Best parameters found!
```

### 3. Visualization
```
Optuna tracks:
├── Best score over time
├── Parameter importance
├── Optimization history
└── Parallel coordinate plots
```

---

## 📈 Expected Improvements

### Based on Typical Results

```
Random Forest:
├── Baseline: 21.64%
├── Expected Optimized: 25-30%
└── Improvement: +15-40%

XGBoost:
├── Baseline: ~40% (needs retraining)
├── Expected Optimized: 45-50%
└── Improvement: +10-25%

TabTransformer:
├── Baseline: 44.97%
├── Expected Optimized: 48-52%
└── Improvement: +5-15%
```

**Note:** Deep learning models (TabTransformer) typically show smaller improvements because they're already well-tuned. Traditional ML models often show larger gains.

---

## 🎯 Optimization Strategy

### What Optuna Does

1. **Exploration Phase (First 10 trials)**
   - Tries diverse parameter combinations
   - Explores the search space
   - Identifies promising regions

2. **Exploitation Phase (Remaining trials)**
   - Focuses on best-performing regions
   - Fine-tunes parameters
   - Converges to optimal values

3. **Final Selection**
   - Returns best parameters found
   - Trains final model with those parameters
   - Evaluates on test set

---

## 📁 Output Files

### After Optimization

```
models/
├── rf_optimized.pkl              ← Optimized Random Forest
├── xgb_optimized.pkl             ← Optimized XGBoost
├── tab_transformer_optimized.pth ← Optimized TabTransformer
├── optimization_results.csv      ← Summary table
├── optimization_results_detailed.txt ← Full details
└── optimization_metrics.json     ← For dashboard display
```

### Results Format

**optimization_results.csv:**
```csv
Model,Baseline Accuracy,Optimized Accuracy,Improvement (%)
Random Forest,0.2164,0.2850,+31.70%
XGBoost,0.4200,0.4850,+15.48%
TabTransformer,0.4497,0.4920,+9.41%
```

**optimization_results_detailed.txt:**
```
Random Forest:
  Baseline Accuracy: 0.2164
  Optimized Accuracy: 0.2850
  Improvement: +31.70%
  Best Parameters:
    n_estimators: 250
    max_depth: 35
    min_samples_split: 5
    min_samples_leaf: 2
    max_features: sqrt
    class_weight: balanced
```

---

## 🎨 Dashboard Integration

### Before Optimization
```
Model: Random Forest (SKLEARN)
├── Collision Accuracy: 21.64%
├── Severity Accuracy: N/A
└── Overall F1: 0.180
```

### After Optimization
```
Model: Random Forest (SKLEARN)
🎯 Optimized Model - Hyperparameters tuned with Optuna

├── Baseline Accuracy: 21.64%
├── Optimized Accuracy: 28.50%
├── Improvement: +31.70% ↑
└── F1-Score: 0.245
```

---

## ⚙️ Advanced Usage

### Custom Number of Trials

```python
# Quick optimization (fewer trials)
optimizer.train_random_forest(n_trials=10)  # ~10 minutes
optimizer.train_xgboost(n_trials=10)        # ~15 minutes
optimizer.train_tabtransformer(n_trials=5)  # ~1 hour

# Thorough optimization (more trials)
optimizer.train_random_forest(n_trials=50)  # ~50 minutes
optimizer.train_xgboost(n_trials=50)        # ~75 minutes
optimizer.train_tabtransformer(n_trials=30) # ~6 hours
```

### Optimize Single Model

```python
from hyperparameter_optimization import ModelOptimizer

optimizer = ModelOptimizer(data_path='../data/model_ready.csv')

# Optimize only Random Forest
optimizer.train_random_forest(n_trials=30)
optimizer.generate_report()
```

### Resume Optimization

```python
# Optuna can resume from previous studies
study = optuna.load_study(
    study_name='rf_optimization',
    storage='sqlite:///optuna.db'
)
study.optimize(objective, n_trials=20)  # Continue optimization
```

---

## 🔍 Understanding the Results

### Metrics Explained

**Accuracy:**
- Percentage of correct predictions
- Higher is better
- Your data: 8 classes, so random = 12.5%

**F1-Score:**
- Harmonic mean of precision and recall
- Balances false positives and false negatives
- Range: 0-1, higher is better

**Improvement %:**
- Relative improvement over baseline
- Formula: (optimized - baseline) / baseline × 100
- Example: 21.64% → 28.50% = +31.70% improvement

### What's a Good Improvement?

```
Improvement Range:
├── 0-5%:   Small (but still valuable)
├── 5-15%:  Moderate (good optimization)
├── 15-30%: Large (excellent optimization)
└── 30%+:   Exceptional (rare, usually means baseline was poor)
```

---

## 🚨 Troubleshooting

### Issue: Optimization is too slow

**Solution 1:** Reduce trials
```python
optimizer.train_tabtransformer(n_trials=5)  # Instead of 15
```

**Solution 2:** Use smaller data subset
```python
# Edit hyperparameter_optimization.py
subset_size = min(20000, len(y))  # Instead of 50000
```

**Solution 3:** Reduce epochs for TabTransformer
```python
# In optimize_tabtransformer function
epochs=5  # Instead of 10
```

### Issue: Out of memory

**Solution:** Reduce batch size
```python
# TabTransformer will automatically try smaller batches
# Or manually set in optimization:
batch_size = trial.suggest_categorical('batch_size', [32, 64])
```

### Issue: No improvement

**Possible reasons:**
1. Model already well-tuned (TabTransformer case)
2. Need more trials (try 50+ trials)
3. Search space too narrow (expand ranges)
4. Data quality issues (check preprocessing)

---

## 📚 Technical Details

### Cross-Validation

```python
# 3-fold cross-validation used during optimization
scores = cross_val_score(model, X_train, y_train, cv=3)

Why 3-fold?
├── Faster than 5-fold or 10-fold
├── Still provides good estimate
└── Balances speed vs accuracy
```

### Scoring Metric

```python
# Accuracy used as optimization metric
scoring='accuracy'

Why accuracy?
├── Simple and interpretable
├── Appropriate for multi-class classification
├── Matches your evaluation metric
```

### Random State

```python
random_state=42

Why fixed seed?
├── Reproducible results
├── Fair comparison between trials
├── Consistent baseline
```

---

## 🎓 Learning Resources

### Optuna Documentation
- Official docs: https://optuna.readthedocs.io/
- Tutorials: https://optuna.readthedocs.io/en/stable/tutorial/
- Examples: https://github.com/optuna/optuna-examples

### Hyperparameter Tuning
- Scikit-learn guide: https://scikit-learn.org/stable/modules/grid_search.html
- XGBoost tuning: https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
- Deep learning tuning: https://pytorch.org/tutorials/

---

## 🎯 Best Practices

### 1. Start Small
```
First run:
├── 10 trials per model
├── Check if optimization works
└── Estimate total time

Second run:
├── 30+ trials per model
├── Get better results
└── Use overnight if needed
```

### 2. Monitor Progress
```
Watch for:
├── Accuracy improving over trials
├── Convergence (scores plateau)
├── Time per trial
└── Memory usage
```

### 3. Compare Fairly
```
Always:
├── Use same train/test split
├── Use same random seed
├── Use same evaluation metric
└── Test on held-out test set
```

### 4. Document Results
```
Save:
├── Best parameters
├── Baseline vs optimized scores
├── Training time
├── Date of optimization
└── Optuna version
```

---

## 📊 Example Output

### Console Output
```
============================================================
HYPERPARAMETER OPTIMIZATION - ALL MODELS
============================================================

Loading data...
✓ Data loaded: 1099868 samples
  Train: 879894, Test: 219974
  Features: 6
  Classes: 8

============================================================
RANDOM FOREST OPTIMIZATION
============================================================

1. Baseline Model (current parameters)...
   Baseline Accuracy: 0.2164
   Baseline F1-Score: 0.1802

2. Optimizing hyperparameters (30 trials)...
[I 2026-02-07 16:30:15] Trial 0 finished with value: 0.2245
[I 2026-02-07 16:31:22] Trial 1 finished with value: 0.2389
[I 2026-02-07 16:32:18] Trial 2 finished with value: 0.2567
...
[I 2026-02-07 16:58:42] Trial 29 finished with value: 0.2834

   Best parameters found:
     n_estimators: 250
     max_depth: 35
     min_samples_split: 5
     min_samples_leaf: 2
     max_features: sqrt
     class_weight: balanced

3. Training optimized model...
   Optimized Accuracy: 0.2850
   Optimized F1-Score: 0.2451

✅ Improvement: +31.70%
✓ Saved to: rf_optimized.pkl

============================================================
OPTIMIZATION RESULTS SUMMARY
============================================================

Model            Baseline Accuracy  Optimized Accuracy  Improvement (%)
Random Forest    0.2164            0.2850              +31.70%
XGBoost          0.4200            0.4850              +15.48%
TabTransformer   0.4497            0.4920              +9.41%

✓ Results saved to: optimization_results.csv
✓ Detailed results saved to: optimization_results_detailed.txt

============================================================
OPTIMIZATION COMPLETE!
============================================================
```

---

## 🎉 Summary

**What You Get:**
- ✅ Optimized models with better performance
- ✅ Detailed comparison reports
- ✅ Dashboard integration with improvement metrics
- ✅ Best hyperparameters for each model
- ✅ Reproducible optimization process

**Time Investment:**
- Setup: 5 minutes
- Optimization: 3-4 hours (can run overnight)
- Dashboard update: 2 minutes
- **Total: ~4 hours** (mostly automated)

**Expected Results:**
- Random Forest: +15-40% improvement
- XGBoost: +10-25% improvement
- TabTransformer: +5-15% improvement

**Worth It?**
- ✅ Significantly better predictions
- ✅ Professional-grade optimization
- ✅ Impressive for portfolio/presentation
- ✅ Learn advanced ML techniques

**Start optimizing now!** 🚀
