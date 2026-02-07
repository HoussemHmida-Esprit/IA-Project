"""
Model Performance Comparison
Compares Random Forest, XGBoost, and TabTransformer on accident prediction
"""
import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Import TabTransformer
from tab_transformer import AccidentTabTransformer


def load_data():
    """Load and prepare data"""
    print("Loading data...")
    df = pd.read_csv('../data/model_ready.csv')
    
    # Features
    categorical_features = ['lum', 'agg', 'int', 'day_of_week']
    numerical_features = ['hour', 'num_users']
    all_features = categorical_features + numerical_features
    
    X = df[all_features]
    y = df['col']  # Collision type
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Data loaded: {len(X)} samples, {len(all_features)} features")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, categorical_features, numerical_features


def evaluate_sklearn_model(model_path, X_test, y_test):
    """Evaluate sklearn model (RF or XGBoost)"""
    print(f"\nEvaluating {model_path.stem}...")
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = joblib.load(f)
    
    if isinstance(model_data, dict):
        model = model_data['model']
    else:
        model = model_data
    
    # Handle MultiOutputClassifier
    if hasattr(model, 'estimators_'):
        model = model.estimators_[0]  # Use collision predictor
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Accuracy: {accuracy:.4f}")
    
    return {
        'model': model_path.stem,
        'accuracy': accuracy,
        'predictions': y_pred
    }


def evaluate_tabtransformer(model_path, X_test, y_test, categorical_features, numerical_features):
    """Evaluate TabTransformer model"""
    print(f"\nEvaluating TabTransformer...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    categorical_encoders = checkpoint['categorical_encoders']
    categorical_dims = [len(enc.classes_) for enc in categorical_encoders.values()]
    num_classes = len(checkpoint['target_encoder'].classes_)
    
    tab_transformer = AccidentTabTransformer('../data/model_ready.csv')
    tab_transformer.load_model(
        str(model_path),
        categorical_dims=categorical_dims,
        num_classes=num_classes
    )
    
    # Predict on test set
    predictions = []
    
    print("Making predictions...")
    for idx, row in X_test.iterrows():
        categorical_data = {f: row[f] for f in categorical_features}
        numerical_data = {f: row[f] for f in numerical_features}
        
        pred, _, _ = tab_transformer.predict(categorical_data, numerical_data)
        predictions.append(pred)
        
        if len(predictions) % 1000 == 0:
            print(f"  Predicted {len(predictions)}/{len(X_test)} samples...")
    
    y_pred = np.array(predictions)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Accuracy: {accuracy:.4f}")
    
    return {
        'model': 'TabTransformer',
        'accuracy': accuracy,
        'predictions': y_pred
    }


def create_comparison_report(results, y_test):
    """Create comprehensive comparison report"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    # Sort by accuracy
    results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    # Print summary table
    print("\n{:<20} {:<15}".format("Model", "Accuracy"))
    print("-" * 35)
    for result in results_sorted:
        print("{:<20} {:<15.4f}".format(result['model'], result['accuracy']))
    
    # Best model
    best_model = results_sorted[0]
    print(f"\n🏆 Best Model: {best_model['model']} (Accuracy: {best_model['accuracy']:.4f})")
    
    # Detailed classification report for best model
    print(f"\nDetailed Classification Report for {best_model['model']}:")
    print(classification_report(y_test, best_model['predictions']))
    
    return results_sorted


def plot_comparison(results):
    """Create comparison visualizations"""
    print("\nCreating comparison plots...")
    
    # Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [r['model'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(models, accuracies, color=colors[:len(models)])
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved comparison plot to model_comparison.png")
    
    plt.close()


def save_results(results):
    """Save results to CSV"""
    df = pd.DataFrame([
        {
            'Model': r['model'],
            'Accuracy': r['accuracy']
        }
        for r in results
    ])
    
    df.to_csv('model_comparison_results.csv', index=False)
    print("✓ Saved results to model_comparison_results.csv")


def main():
    """Main comparison function"""
    print("="*60)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("Random Forest vs XGBoost vs TabTransformer")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test, cat_features, num_features = load_data()
    
    results = []
    
    # Evaluate Random Forest
    rf_path = Path('rf_pca_multitarget.pkl')
    if rf_path.exists():
        try:
            rf_result = evaluate_sklearn_model(rf_path, X_test, y_test)
            results.append(rf_result)
        except Exception as e:
            print(f"Error evaluating Random Forest: {e}")
    
    # Evaluate XGBoost
    xgb_path = Path('xgb_nopca_multitarget.pkl')
    if xgb_path.exists():
        try:
            xgb_result = evaluate_sklearn_model(xgb_path, X_test, y_test)
            results.append(xgb_result)
        except Exception as e:
            print(f"Error evaluating XGBoost: {e}")
    
    # Evaluate TabTransformer
    tt_path = Path('tab_transformer_best.pth')
    if tt_path.exists():
        try:
            tt_result = evaluate_tabtransformer(
                tt_path, X_test, y_test, cat_features, num_features
            )
            results.append(tt_result)
        except Exception as e:
            print(f"Error evaluating TabTransformer: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("\n❌ No models found to compare!")
        return
    
    # Create comparison report
    results_sorted = create_comparison_report(results, y_test)
    
    # Create visualizations
    plot_comparison(results_sorted)
    
    # Save results
    save_results(results_sorted)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
