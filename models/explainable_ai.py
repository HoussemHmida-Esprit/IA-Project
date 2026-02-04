"""
Explainable AI Module using SHAP
Provides interpretability for XGBoost models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path

# Feature name mappings for better visualization
FEATURE_NAMES = {
    'lum': 'Lighting',
    'atm': 'Weather',
    'agg': 'Location',
    'int': 'Intersection',
    'hour': 'Hour',
    'day_of_week': 'Day of Week',
    'month': 'Month',
    'num_users': 'People Involved'
}


class AccidentXAI:
    """Explainable AI for accident prediction models"""
    
    def __init__(self, model_path: str, data_path: str):
        """
        Initialize XAI module
        
        Args:
            model_path: Path to trained model (.pkl)
            data_path: Path to model-ready data (.csv)
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.X = None
        self.y = None
        
    def load_model_and_data(self):
        """Load trained model and data"""
        # Load model
        with open(self.model_path, 'rb') as f:
            model_data = joblib.load(f)
        
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.feature_names = model_data.get('features', None)
        else:
            self.model = model_data
            self.feature_names = None
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Define features
        if self.feature_names is None:
            self.feature_names = ['lum', 'atm', 'agg', 'int', 'hour', 'day_of_week', 'month', 'num_users']
        
        # Prepare features
        self.X = df[self.feature_names].copy()
        
        # Get target (severity)
        if 'max_severity' in df.columns:
            self.y = df['max_severity']
        elif 'grav' in df.columns:
            self.y = df['grav']
        else:
            self.y = None
        
        print(f"✓ Loaded model from {self.model_path}")
        print(f"✓ Loaded data: {len(self.X)} samples, {len(self.feature_names)} features")
        
    def compute_shap_values(self, sample_size: int = 1000):
        """
        Compute SHAP values for the model
        
        Args:
            sample_size: Number of samples to use (for speed)
        """
        print("Computing SHAP values...")
        
        # Sample data for faster computation
        if len(self.X) > sample_size:
            sample_idx = np.random.choice(len(self.X), sample_size, replace=False)
            X_sample = self.X.iloc[sample_idx]
        else:
            X_sample = self.X
        
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Compute SHAP values
        self.shap_values = self.explainer(X_sample)
        
        print(f"✓ SHAP values computed for {len(X_sample)} samples")
        
    def plot_global_summary(self, save_path: str = None):
        """
        Create global summary plot showing feature importance
        
        Args:
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Run compute_shap_values() first.")
        
        plt.figure(figsize=(10, 6))
        
        # Rename features for better readability
        feature_names_display = [FEATURE_NAMES.get(f, f) for f in self.feature_names]
        
        # Create summary plot
        shap.summary_plot(
            self.shap_values,
            features=self.shap_values.data,
            feature_names=feature_names_display,
            show=False
        )
        
        plt.title("Global Feature Importance (SHAP)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved global summary plot to {save_path}")
        
        return plt.gcf()
    
    def plot_dependence(self, feature: str, interaction_feature: str = None, save_path: str = None):
        """
        Create dependence plot for a specific feature
        
        Args:
            feature: Feature to analyze (e.g., 'hour')
            interaction_feature: Feature to color by (e.g., 'lum')
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Run compute_shap_values() first.")
        
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found in model features")
        
        feature_idx = self.feature_names.index(feature)
        
        plt.figure(figsize=(10, 6))
        
        # Get display names
        feature_display = FEATURE_NAMES.get(feature, feature)
        interaction_display = FEATURE_NAMES.get(interaction_feature, interaction_feature) if interaction_feature else None
        
        # Create dependence plot
        shap.dependence_plot(
            feature_idx,
            self.shap_values.values,
            self.shap_values.data,
            feature_names=self.feature_names,
            interaction_index=interaction_feature if interaction_feature else "auto",
            show=False
        )
        
        plt.title(f"SHAP Dependence: {feature_display}", fontsize=14, fontweight='bold')
        plt.xlabel(feature_display, fontsize=12)
        plt.ylabel(f"SHAP value for {feature_display}", fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved dependence plot to {save_path}")
        
        return plt.gcf()
    
    def plot_waterfall(self, sample_idx: int, save_path: str = None):
        """
        Create waterfall plot for a single prediction
        
        Args:
            sample_idx: Index of sample to explain
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Run compute_shap_values() first.")
        
        plt.figure(figsize=(10, 8))
        
        # Rename features
        feature_names_display = [FEATURE_NAMES.get(f, f) for f in self.feature_names]
        
        # Create waterfall plot
        shap.plots.waterfall(
            self.shap_values[sample_idx],
            show=False
        )
        
        plt.title(f"SHAP Explanation for Sample {sample_idx}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved waterfall plot to {save_path}")
        
        return plt.gcf()
    
    def get_feature_importance(self):
        """
        Get feature importance as DataFrame
        
        Returns:
            DataFrame with features and their mean absolute SHAP values
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Run compute_shap_values() first.")
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values.values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': [FEATURE_NAMES.get(f, f) for f in self.feature_names],
            'Feature_Code': self.feature_names,
            'Mean_Abs_SHAP': mean_abs_shap
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Mean_Abs_SHAP', ascending=False)
        importance_df = importance_df.reset_index(drop=True)
        
        return importance_df


def main():
    """Example usage"""
    # Initialize XAI
    xai = AccidentXAI(
        model_path='models/xgb_nopca_multitarget.pkl',
        data_path='data/model_ready.csv'
    )
    
    # Load model and data
    xai.load_model_and_data()
    
    # Compute SHAP values
    xai.compute_shap_values(sample_size=1000)
    
    # Create visualizations
    xai.plot_global_summary(save_path='models/shap_global_summary.png')
    xai.plot_dependence('hour', interaction_feature='lum', save_path='models/shap_hour_dependence.png')
    
    # Get feature importance
    importance = xai.get_feature_importance()
    print("\nFeature Importance:")
    print(importance)
    
    # Save importance
    importance.to_csv('models/feature_importance_shap.csv', index=False)
    print("\n✓ XAI analysis complete!")


if __name__ == "__main__":
    main()
