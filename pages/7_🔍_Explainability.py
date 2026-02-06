"""
Model Explainability Page - SHAP Analysis

Provides interpretability for machine learning models using SHAP values.
Shows which features are most important for predictions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.explainable_ai import AccidentXAI
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="Model Explainability",
    page_icon="🔍",
    layout="wide"
)

st.title("Model Explainability")
st.markdown("*Understand how the model makes predictions using SHAP values*")

if not SHAP_AVAILABLE:
    st.error("""
    ⚠️ **SHAP library not installed**
    
    To use this feature, install SHAP:
    ```bash
    pip install shap
    ```
    """)
    st.stop()

# Check if models exist
MODELS_DIR = Path("models")
DATA_PATH = Path("data/model_ready.csv")

available_models = {}
for model_file in MODELS_DIR.glob("*_multitarget.pkl"):
    model_name = model_file.stem.replace('_multitarget', '')
    if model_name in ['rf_pca', 'xgb_nopca']:
        friendly_name = "Random Forest" if model_name == 'rf_pca' else "XGBoost"
        available_models[friendly_name] = model_file

if not available_models:
    st.error("⚠️ No models found. Please train a model first.")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"⚠️ Data file not found: {DATA_PATH}")
    st.stop()

# Sidebar: Model selection
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox(
    "Select Model to Explain",
    options=list(available_models.keys())
)

sample_size = st.sidebar.slider(
    "Sample Size for SHAP",
    min_value=100,
    max_value=2000,
    value=500,
    step=100,
    help="Larger samples give more accurate results but take longer"
)

# Main content
st.header("What is SHAP?")
st.markdown("""
**SHAP (SHapley Additive exPlanations)** is a method to explain individual predictions by computing 
the contribution of each feature to the prediction.

- **Positive SHAP value**: Feature pushes prediction higher
- **Negative SHAP value**: Feature pushes prediction lower
- **Magnitude**: How much the feature matters
""")

st.divider()

# Initialize XAI
@st.cache_resource
def load_xai(model_path, data_path):
    """Load XAI module"""
    xai = AccidentXAI(str(model_path), str(data_path))
    xai.load_model_and_data()
    return xai

def compute_shap_if_needed(xai, sample_size):
    """Compute SHAP values if not already computed"""
    if xai.shap_values is None:
        xai.compute_shap_values(sample_size=sample_size)
    return xai

# Load model
with st.spinner(f"Loading {selected_model} model..."):
    try:
        xai = load_xai(available_models[selected_model], DATA_PATH)
        st.success(f"✅ Loaded {selected_model} model")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Compute SHAP values
with st.spinner("Computing SHAP values... This may take a minute."):
    try:
        xai = compute_shap_if_needed(xai, sample_size)
        st.success(f"✅ SHAP values computed for {sample_size} samples")
    except Exception as e:
        st.error(f"Error computing SHAP values: {e}")
        st.stop()

st.divider()

# Tab layout for different visualizations
tab1, tab2, tab3 = st.tabs(["Global Importance", "Feature Dependence", "Feature Importance Table"])

with tab1:
    st.header("Global Feature Importance")
    st.markdown("""
    This plot shows which features are most important across all predictions.
    Each point represents one sample, colored by feature value.
    """)
    
    try:
        fig = xai.plot_global_summary()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error creating summary plot: {e}")

with tab2:
    st.header("Feature Dependence Analysis")
    st.markdown("""
    Shows how a specific feature affects predictions and how it interacts with other features.
    """)
    
    # Feature selection
    feature_options = {
        'hour': 'Hour of Day',
        'lum': 'Lighting Conditions',
        'atm': 'Weather Conditions',
        'agg': 'Location Type',
        'int': 'Intersection Type',
        'day_of_week': 'Day of Week',
        'month': 'Month',
        'num_users': 'Number of People'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_feature = st.selectbox(
            "Select Feature to Analyze",
            options=list(feature_options.keys()),
            format_func=lambda x: feature_options[x]
        )
    
    with col2:
        interaction_feature = st.selectbox(
            "Color by Feature (Interaction)",
            options=['auto'] + list(feature_options.keys()),
            format_func=lambda x: 'Auto' if x == 'auto' else feature_options[x]
        )
    
    if st.button("Generate Dependence Plot"):
        with st.spinner("Creating dependence plot..."):
            try:
                interaction = None if interaction_feature == 'auto' else interaction_feature
                fig = xai.plot_dependence(selected_feature, interaction_feature=interaction)
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Error creating dependence plot: {e}")

with tab3:
    st.header("Feature Importance Rankings")
    st.markdown("""
    Mean absolute SHAP values for each feature, showing overall importance.
    """)
    
    try:
        importance_df = xai.get_feature_importance()
        
        # Display as table
        st.dataframe(
            importance_df[['Feature', 'Mean_Abs_SHAP']].style.format({
                'Mean_Abs_SHAP': '{:.4f}'
            }),
            hide_index=True,
            use_container_width=True
        )
        
        # Bar chart
        import plotly.express as px
        fig = px.bar(
            importance_df,
            x='Mean_Abs_SHAP',
            y='Feature',
            orientation='h',
            title='Feature Importance (Mean Absolute SHAP)',
            color='Mean_Abs_SHAP',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating importance table: {e}")

st.divider()

# Information section
st.header("About SHAP Analysis")
st.markdown("""
### How to Interpret SHAP Values

1. **Global Summary Plot**:
   - Features are ranked by importance (top to bottom)
   - Each dot is one prediction
   - Red = high feature value, Blue = low feature value
   - Position on x-axis shows impact on prediction

2. **Dependence Plot**:
   - Shows relationship between feature value and SHAP value
   - Helps identify non-linear patterns
   - Color shows interaction with another feature

3. **Feature Importance Table**:
   - Simple ranking of features by average impact
   - Higher values = more important features

### Why This Matters

Understanding feature importance helps:
- Build trust in model predictions
- Identify which factors matter most for accidents
- Guide policy decisions and interventions
- Debug model behavior
""")
