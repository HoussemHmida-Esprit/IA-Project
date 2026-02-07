"""
Update Dashboard with Optimized Models
Copies optimized models to dashboard and updates metrics display
"""
import shutil
import pickle
import json
from pathlib import Path

def update_models():
    """Copy optimized models to replace current models"""
    print("Updating dashboard with optimized models...")
    
    updates = []
    
    # Random Forest
    if Path('rf_optimized.pkl').exists():
        shutil.copy('rf_optimized.pkl', 'rf_pca_multitarget.pkl')
        print("✓ Updated Random Forest")
        updates.append("Random Forest")
    
    # XGBoost
    if Path('xgb_optimized.pkl').exists():
        shutil.copy('xgb_optimized.pkl', 'xgb_nopca_multitarget.pkl')
        print("✓ Updated XGBoost")
        updates.append("XGBoost")
    
    # TabTransformer
    if Path('tab_transformer_optimized.pth').exists():
        shutil.copy('tab_transformer_optimized.pth', 'tab_transformer_best.pth')
        print("✓ Updated TabTransformer")
        updates.append("TabTransformer")
    
    if updates:
        print(f"\n✅ Dashboard updated with {len(updates)} optimized models!")
        print("   Restart Streamlit to see the new models.")
    else:
        print("\n⚠️ No optimized models found. Run hyperparameter_optimization.py first.")
    
    return updates

def create_metrics_file():
    """Create a JSON file with optimization metrics for dashboard"""
    metrics = {}
    
    # Load Random Forest metrics
    if Path('rf_optimized.pkl').exists():
        with open('rf_optimized.pkl', 'rb') as f:
            data = pickle.load(f)
            metrics['Random Forest'] = data.get('metrics', {})
    
    # Load XGBoost metrics
    if Path('xgb_optimized.pkl').exists():
        with open('xgb_optimized.pkl', 'rb') as f:
            data = pickle.load(f)
            metrics['XGBoost'] = data.get('metrics', {})
    
    # Save metrics
    if metrics:
        with open('optimization_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print("✓ Created optimization_metrics.json")
        return metrics
    
    return None

if __name__ == "__main__":
    print("="*60)
    print("DASHBOARD UPDATE UTILITY")
    print("="*60)
    print()
    
    # Update models
    updates = update_models()
    
    # Create metrics file
    print()
    metrics = create_metrics_file()
    
    if metrics:
        print("\n" + "="*60)
        print("OPTIMIZATION IMPROVEMENTS")
        print("="*60)
        for model, data in metrics.items():
            print(f"\n{model}:")
            print(f"  Baseline:  {data.get('baseline_accuracy', 0):.4f}")
            print(f"  Optimized: {data.get('optimized_accuracy', 0):.4f}")
            print(f"  Improvement: {data.get('improvement_pct', 0):+.2f}%")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Restart your Streamlit dashboard:")
    print("   streamlit run app.py")
    print()
    print("2. Navigate to the Prediction page")
    print("3. Select a model to see improved performance!")
