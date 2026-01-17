"""
Predictions Page - Model accuracy and predictions display.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from dashboard.services.model_service import get_model_service
from dashboard.services.trading_service import get_trading_service
from dashboard.services.data_service import load_tsa_data, get_weekly_tsa_data
from dashboard.components.charts import (
    create_prediction_chart,
    create_accuracy_chart,
    create_model_weights_chart,
    create_feature_importance_chart
)
from src.trading.contract_selector import ContractSelector

st.set_page_config(page_title="Predictions", page_icon="", layout="wide")

st.title("Model Predictions")
st.markdown("View prediction accuracy, model performance, and forecasts")


def format_number(num, decimals=0):
    if num is None:
        return "N/A"
    if decimals == 0:
        return f"{int(num):,}"
    return f"{num:,.{decimals}f}"


# Get services
model_service = get_model_service()

# Current Prediction Section
st.header("Current Forecast")

col1, col2 = st.columns([2, 1])

with col1:
    prediction = model_service.get_prediction(None)

    pred_cols = st.columns(4)

    with pred_cols[0]:
        st.metric(
            "Predicted Weekly Passengers",
            format_number(prediction.get('prediction'))
        )

    with pred_cols[1]:
        conf = prediction.get('confidence', 0)
        st.metric(
            "Confidence",
            f"{conf * 100:.1f}%" if conf else "N/A"
        )

    with pred_cols[2]:
        unc = prediction.get('uncertainty')
        st.metric(
            "Uncertainty",
            format_number(unc) if unc else "N/A"
        )

    with pred_cols[3]:
        lower = prediction.get('lower_bound')
        upper = prediction.get('upper_bound')
        if lower and upper:
            st.metric(
                "95% CI Range",
                f"{format_number(upper - lower)}"
            )

    st.info(f"""
    **Forecast Range**: {format_number(prediction.get('lower_bound'))} - {format_number(prediction.get('upper_bound'))}
    """)

with col2:
    # Model weights pie chart
    weights = model_service.get_model_weights()
    fig = create_model_weights_chart(weights)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Contract Recommendation Section
st.header("Recommended Trade")

trading_service = get_trading_service()
selector = ContractSelector(min_ev_threshold=0.02)

# Get contracts from Kalshi (or generate simulated ones for demo)
contracts = trading_service.get_available_contracts()

if not contracts and prediction.get('prediction'):
    # Generate simulated contracts for demo mode
    pred_val = prediction.get('prediction')
    base = round(pred_val / 500000) * 500000
    contracts = [
        {'ticker': f'DEMO-T{int(t)}', 'threshold': t, 'yes_price': 0.5, 'no_price': 0.5}
        for t in [base - 1000000, base - 500000, base, base + 500000, base + 1000000]
        if t > 0
    ]

if contracts and prediction.get('prediction') and prediction.get('uncertainty'):
    recommendation = selector.select_contract(
        prediction=prediction.get('prediction'),
        prediction_std=prediction.get('uncertainty'),
        contracts=contracts
    )

    if recommendation.get('contract'):
        if recommendation['expected_value'] >= 0.02:
            st.success(f"**{recommendation['side'].upper()}** on {recommendation['contract']}")
        else:
            st.warning(f"**{recommendation['side'].upper()}** on {recommendation['contract']} (Low EV)")

        rec_cols = st.columns(3)
        with rec_cols[0]:
            st.metric("Confidence", f"{recommendation['confidence']:.0%}")
        with rec_cols[1]:
            st.metric("Expected Value", f"{recommendation['expected_value']:.1%}")
        with rec_cols[2]:
            # Calculate profit for $100 bet
            ev_dollars = recommendation['expected_value'] * 100
            st.metric("EV ($100 bet)", f"${ev_dollars:.2f}")

        st.info(recommendation['reasoning'])
    else:
        st.warning("No positive EV contracts found - recommend HOLD")
        st.caption(recommendation.get('reasoning', ''))
else:
    if not contracts:
        st.info("Connect to Kalshi API to see contract recommendations")
    else:
        st.warning("Prediction data unavailable for contract selection")

st.markdown("---")

# Accuracy Over Time Section
st.header("Model Accuracy")

# Controls
col1, col2 = st.columns([1, 3])

with col1:
    lookback_days = st.selectbox(
        "Lookback Period",
        options=[7, 14, 30, 60, 90],
        index=2,
        format_func=lambda x: f"{x} days"
    )

    rolling_window = st.slider(
        "Rolling Average Window",
        min_value=3,
        max_value=14,
        value=7
    )

# Get accuracy history
accuracy_df = model_service.get_accuracy_history(days=lookback_days)

if not accuracy_df.empty:
    # Accuracy chart
    fig = create_accuracy_chart(accuracy_df, window=rolling_window)
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy metrics
    st.subheader("Accuracy Metrics")

    metric_cols = st.columns(4)

    perf = model_service.get_recent_performance(lookback_days)

    with metric_cols[0]:
        st.metric(
            "Mean Accuracy",
            f"{perf.get('mean_accuracy', 0) * 100:.1f}%"
        )

    with metric_cols[1]:
        st.metric(
            "Accuracy Std Dev",
            f"{perf.get('accuracy_std', 0) * 100:.1f}%"
        )

    with metric_cols[2]:
        rmse = perf.get('rmse', 0)
        st.metric(
            "RMSE",
            format_number(rmse) if rmse else "N/A"
        )

    with metric_cols[3]:
        mae = perf.get('mae', 0)
        st.metric(
            "MAE",
            format_number(mae) if mae else "N/A"
        )

else:
    st.warning("No accuracy history available")

st.markdown("---")

# Predictions vs Actuals
st.header("Predictions vs Actuals")

if not accuracy_df.empty and 'predicted' in accuracy_df.columns and 'actual' in accuracy_df.columns:
    fig = create_prediction_chart(accuracy_df)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Prediction comparison chart will appear once data is available")

st.markdown("---")

# Feature Importance
st.header("Feature Importance")

importance_df = model_service.get_feature_importance()

if not importance_df.empty:
    fig = create_feature_importance_chart(importance_df, top_n=5)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Feature importance not available - no trained model loaded")

# Footer
st.markdown("---")
st.caption("Predictions are updated when new TSA data becomes available (typically daily)")
