"""
Quick Test - Hyperparameter Optimization
Tests the optimization system with minimal trials
"""
from hyperparameter_optimization import ModelOptimizer

def quick_test():
    """Run quick optimization test (2 trials each)"""
    print("="*60)
    print("QUICK OPTIMIZATION TEST")
    print("="*60)
    print("\nThis will run 2 trials per model to verify everything works.")
    print("Full optimization will take much longer (3-4 hours).\n")
    
    optimizer = ModelOptimizer(data_path='../data/model_ready.csv')
    
    # Test Random Forest (fast)
    print("\n1. Testing Random Forest optimization...")
    try:
        optimizer.train_random_forest(n_trials=2)
        print("✅ Random Forest optimization works!")
    except Exception as e:
        print(f"❌ Random Forest failed: {e}")
        return False
    
    # Test XGBoost (medium)
    print("\n2. Testing XGBoost optimization...")
    try:
        optimizer.train_xgboost(n_trials=2)
        print("✅ XGBoost optimization works!")
    except Exception as e:
        print(f"❌ XGBoost failed: {e}")
        return False
    
    # Generate report
    print("\n3. Generating report...")
    try:
        optimizer.generate_report()
        print("✅ Report generation works!")
    except Exception as e:
        print(f"❌ Report failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("TEST PASSED!")
    print("="*60)
    print("\nAll optimization functions work correctly.")
    print("You can now run the full optimization:")
    print("  python hyperparameter_optimization.py")
    
    return True

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)
