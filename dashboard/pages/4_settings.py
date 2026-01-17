"""
Settings Page - API configuration and risk limits.
"""
import streamlit as st
import os
from pathlib import Path
import sys

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from dashboard.services.trading_service import get_trading_service

st.set_page_config(page_title="Settings", page_icon="", layout="wide")

st.title("Settings")
st.markdown("Configure API credentials and trading parameters")

# Get services
trading_service = get_trading_service()

# API Configuration
st.header("API Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Kalshi API")

    # Show current status
    if trading_service.connected:
        st.success("Connected")
    else:
        st.warning("Not connected - running in demo mode")

    # API Key ID (masked)
    current_key_id = os.getenv('KALSHI_API_KEY_ID', '')
    masked_key = current_key_id[:8] + '...' if len(current_key_id) > 8 else current_key_id

    st.text_input(
        "API Key ID",
        value=masked_key,
        disabled=True,
        help="Set in .env file as KALSHI_API_KEY_ID"
    )

    # Private key path
    key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH', 'secrets/kalshi_private_key.pem')
    key_exists = Path(project_root / key_path).exists()

    st.text_input(
        "Private Key Path",
        value=key_path,
        disabled=True,
        help="Set in .env file as KALSHI_PRIVATE_KEY_PATH"
    )

    if key_exists:
        st.success("Private key file found")
    else:
        st.error("Private key file not found")

    # Test connection button
    if st.button("Test Connection", use_container_width=True):
        with st.spinner("Testing connection..."):
            success, message = trading_service.test_connection()

            if success:
                st.success(message)
            else:
                st.error(message)

with col2:
    st.subheader("FRED API (Economic Data)")

    fred_key = os.getenv('FRED_API_KEY', '')
    masked_fred = fred_key[:8] + '...' if len(fred_key) > 8 else 'Not set'

    st.text_input(
        "FRED API Key",
        value=masked_fred,
        disabled=True,
        help="Set in .env file as FRED_API_KEY"
    )

    if fred_key:
        st.success("FRED API key configured")
    else:
        st.warning("FRED API key not set")

    st.markdown("""
    Get a free FRED API key:
    [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
    """)

    st.subheader("NOAA Weather API")
    st.success("No API key required")
    st.markdown("Uses public api.weather.gov endpoint")

st.markdown("---")

# Data & Model Actions
st.header("Data & Model Actions")

col1, col2 = st.columns(2)

with col1:
    st.subheader("TSA Data")

    # Check if TSA data exists
    tsa_data_dir = project_root / "data" / "raw" / "tsa"
    tsa_files = list(tsa_data_dir.glob("*.parquet")) if tsa_data_dir.exists() else []

    if tsa_files:
        latest_file = max(tsa_files, key=lambda f: f.stat().st_mtime)
        from datetime import datetime
        file_age = datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)
        st.success(f"TSA data available ({file_age.days}d old)")
    else:
        st.warning("No TSA data collected yet")

    if st.button("Fetch TSA Data", use_container_width=True):
        with st.spinner("Fetching TSA data from tsa.gov..."):
            try:
                from data.collectors.tsa_collector import TSACollector
                from datetime import datetime, timedelta

                collector = TSACollector(data_dir=project_root / "data" / "raw")
                end_date = datetime.now() - timedelta(days=1)
                start_date = end_date - timedelta(days=365)  # Fetch last year

                data = collector.fetch_data(start_date, end_date, use_cache=False)
                st.success(f"Fetched {len(data)} days of TSA data")
                st.rerun()
            except Exception as e:
                st.error(f"Error fetching TSA data: {str(e)}")

with col2:
    st.subheader("Prediction Model")

    # Check if model exists
    models_dir = project_root / "models"
    model_dirs = [d for d in models_dir.glob("ensemble_*") if d.is_dir()] if models_dir.exists() else []

    if model_dirs:
        latest_model = max(model_dirs, key=lambda d: d.name)
        st.success(f"Model loaded: {latest_model.name}")
    else:
        st.warning("No trained model found")

    if st.button("Train Model", use_container_width=True):
        with st.spinner("Training ensemble model... This may take a few minutes."):
            try:
                from data.collectors.tsa_collector import TSACollector
                from models.ensemble import EnsembleModel
                from datetime import datetime, timedelta
                import pandas as pd
                import numpy as np

                # Load TSA data
                collector = TSACollector(data_dir=project_root / "data" / "raw")
                end_date = datetime.now() - timedelta(days=1)
                start_date = end_date - timedelta(days=365)

                tsa_data = collector.fetch_data(start_date, end_date)

                if tsa_data.empty:
                    st.error("No TSA data available. Please fetch TSA data first.")
                else:
                    # Build features from TSA passenger data
                    y = tsa_data['passengers']
                    X = pd.DataFrame(index=tsa_data.index)
                    X['day_of_week'] = X.index.dayofweek
                    X['month'] = X.index.month
                    X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)
                    X['lag_1'] = y.shift(1)
                    X['lag_7'] = y.shift(7)
                    X['rolling_mean_7'] = y.rolling(7).mean()
                    X['rolling_std_7'] = y.rolling(7).std()

                    # Drop NaN rows from lag features
                    valid_idx = X.dropna().index
                    X = X.loc[valid_idx]
                    y = y.loc[valid_idx]

                    # Create ensemble config (using only GBM to avoid LightGBM issues)
                    config = {
                        'model_dir': str(models_dir),
                        'model_configs': {
                            'exp': {
                                'type': 'exponential',
                                'seasonal_periods': [7]
                            },
                            'gbm': {
                                'type': 'gbm',
                                'lgb_params': {
                                    'n_estimators': 100,
                                    'learning_rate': 0.05,
                                    'verbosity': -1,
                                    'early_stopping_rounds': 50
                                }
                            }
                        },
                        'weights_method': 'learned',
                        'validation_window': 30
                    }

                    # Train and save
                    ensemble = EnsembleModel('ensemble', config)
                    ensemble.train(X, y)
                    save_path = ensemble.save()

                    st.success(f"Model trained and saved to {save_path.name}")
                    st.cache_resource.clear()
                    st.rerun()

            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

st.markdown("---")

# Credentials Setup Guide
st.header("Setup Guide")

with st.expander("How to configure Kalshi API credentials", expanded=not trading_service.connected):
    st.markdown("""
    ### Step 1: Get your API credentials from Kalshi

    1. Log into [Kalshi.com](https://kalshi.com)
    2. Go to **Settings** > **API Keys**
    3. Click **Create New API Key**
    4. Save the **API Key ID** shown
    5. Download the **Private Key** (.pem file) - this is only shown once!

    ### Step 2: Configure environment variables

    Create a `.env` file in the project root with:

    ```
    KALSHI_API_KEY_ID=your-api-key-id-here
    KALSHI_PRIVATE_KEY_PATH=secrets/kalshi_private_key.pem
    ```

    ### Step 3: Save your private key

    1. Create a `secrets/` folder in the project root
    2. Save your downloaded `.pem` file as `secrets/kalshi_private_key.pem`
    3. Ensure the secrets folder is in `.gitignore`

    ### Step 4: Restart the dashboard

    After configuring credentials, restart Streamlit to load them.
    """)

st.markdown("---")

# Risk Settings
st.header("Risk Settings")

st.info("Risk settings are configured in `configs/config.yaml` and cannot be modified from the dashboard for safety.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Position Limits")

    st.write(f"**Max contracts per trade**: 100")
    st.write(f"**Max total position**: 1,000 contracts")
    st.write(f"**Concentration limit**: 25%")

with col2:
    st.subheader("Loss Limits")

    st.write(f"**Max daily loss**: 15% of capital")
    st.write(f"**Max trade loss**: 5% of capital")
    st.write(f"**Stop loss**: 15%")

st.markdown("---")

# Trading Safety
st.header("Trading Safety Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Confirmation Required")
    st.write("All orders require explicit confirmation checkbox before submission")

with col2:
    st.subheader("Trade Cooldown")
    st.write("5-second minimum delay between trades to prevent accidental double-orders")

with col3:
    st.subheader("Size Limits")
    st.write("Maximum 100 contracts per trade enforced at order validation")

st.markdown("---")

# Model Settings
st.header("Model Settings")

st.info("Model configuration is in `configs/config.yaml`")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Parameters")

    st.write(f"**Prediction horizon**: 7 days")
    st.write(f"**Feature lookback**: 90 days")
    st.write(f"**Validation window**: 30 days")

with col2:
    st.subheader("Ensemble Components")

    st.write("- SARIMAX (time series)")
    st.write("- Exponential Smoothing")
    st.write("- Gradient Boosting (LightGBM)")
    st.write("- Neural Network")

st.markdown("---")

# Data Settings
st.header("Data Update Frequency")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("TSA Data", "24 hours")

with col2:
    st.metric("Weather Data", "1 hour")

with col3:
    st.metric("Airline Data", "48 hours")

with col4:
    st.metric("Economic Data", "168 hours")

st.markdown("---")

# System Info
st.header("System Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Paths")

    st.write(f"**Project root**: {project_root}")
    st.write(f"**Data directory**: {project_root / 'data'}")
    st.write(f"**Models directory**: {project_root / 'models'}")
    st.write(f"**Config file**: {project_root / 'configs' / 'config.yaml'}")

with col2:
    st.subheader("Version Info")

    import streamlit
    import pandas
    import plotly

    st.write(f"**Streamlit**: {streamlit.__version__}")
    st.write(f"**Pandas**: {pandas.__version__}")
    st.write(f"**Plotly**: {plotly.__version__}")

st.markdown("---")

# Clear Cache
st.header("Maintenance")

col1, col2 = st.columns(2)

with col1:
    if st.button("Clear Data Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Data cache cleared")

with col2:
    if st.button("Clear Resource Cache", use_container_width=True):
        st.cache_resource.clear()
        st.success("Resource cache cleared - services will reinitialize")
        st.rerun()

# Footer
st.markdown("---")
st.caption("Settings changes may require a dashboard restart to take effect")
