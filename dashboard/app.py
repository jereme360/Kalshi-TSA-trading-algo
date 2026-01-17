"""
Kalshi TSA Trading Dashboard
Main entry point for the Streamlit application.
"""
import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from dashboard.services.data_service import get_data_freshness, get_latest_tsa_value
from dashboard.services.model_service import get_model_service
from dashboard.services.trading_service import get_trading_service

# Page config
st.set_page_config(
    page_title="TSA Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


def format_number(num: float, decimals: int = 0) -> str:
    """Format number with commas."""
    if num is None:
        return "N/A"
    if decimals == 0:
        return f"{int(num):,}"
    return f"{num:,.{decimals}f}"


def format_pct(num: float) -> str:
    """Format as percentage."""
    if num is None:
        return "N/A"
    return f"{num * 100:+.1f}%"


def main():
    # Sidebar
    with st.sidebar:
        st.title("TSA Trading")
        st.markdown("---")

        # Connection status
        st.subheader("Status")

        # Check services
        trading_service = get_trading_service()
        model_service = get_model_service()
        data_freshness = get_data_freshness()

        # Kalshi connection
        if trading_service.connected:
            st.success("Kalshi: Connected")
        else:
            st.warning("Kalshi: Demo Mode")

        # Model status
        if model_service.model_loaded:
            st.success("Model: Loaded")
        else:
            st.info("Model: Using samples")

        # Data freshness
        tsa_status = data_freshness.get('tsa', {})
        if tsa_status.get('status') == 'fresh':
            st.success(f"TSA Data: {tsa_status.get('days_old', '?')}d old")
        elif tsa_status.get('status') == 'stale':
            st.warning(f"TSA Data: {tsa_status.get('days_old', '?')}d old")
        else:
            st.error("TSA Data: Unavailable")

        st.markdown("---")

        # Navigation
        st.subheader("Navigation")
        st.page_link("pages/1_predictions.py", label="Predictions")
        st.page_link("pages/2_trading.py", label="Trading")
        st.page_link("pages/3_portfolio.py", label="Portfolio")
        st.page_link("pages/4_settings.py", label="Settings")

    # Main content
    st.title("TSA Prediction Trading Dashboard")
    st.markdown("Real-time predictions and trading for Kalshi TSA checkpoint markets")

    # Quick summary metrics
    col1, col2, col3, col4 = st.columns(4)

    # Get latest data
    latest_tsa = get_latest_tsa_value()
    model_perf = model_service.get_recent_performance()
    market_data = trading_service.get_market_data()
    balance = trading_service.get_account_balance()

    with col1:
        st.metric(
            "Latest TSA Passengers",
            format_number(latest_tsa.get('passengers')),
            format_pct(latest_tsa.get('yoy_change'))
        )

    with col2:
        st.metric(
            "Model Accuracy",
            f"{model_perf.get('mean_accuracy', 0) * 100:.1f}%",
            f"{model_perf.get('hit_rate', 0) * 100:.0f}% hit rate"
        )

    with col3:
        st.metric(
            "Market Price (YES)",
            f"${market_data.get('yes_bid', 0):.2f} / ${market_data.get('yes_ask', 0):.2f}",
            f"Vol: {format_number(market_data.get('volume', 0))}"
        )

    with col4:
        st.metric(
            "Account Balance",
            f"${balance.get('balance', 0):,.2f}",
            f"${balance.get('available', 0):,.2f} available"
        )

    st.markdown("---")

    # Current prediction summary
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Current Prediction")

        prediction = model_service.get_prediction(None)

        pred_col1, pred_col2, pred_col3 = st.columns(3)

        with pred_col1:
            st.metric(
                "Weekly Passenger Forecast",
                format_number(prediction.get('prediction'))
            )

        with pred_col2:
            conf = prediction.get('confidence', 0)
            st.metric(
                "Confidence",
                f"{conf * 100:.1f}%" if conf else "N/A"
            )

        with pred_col3:
            lower = prediction.get('lower_bound')
            upper = prediction.get('upper_bound')
            if lower and upper:
                st.metric(
                    "95% Interval",
                    f"{format_number(lower)} - {format_number(upper)}"
                )
            else:
                st.metric("95% Interval", "N/A")

    with col2:
        st.subheader("Trading Signal")

        # Calculate signal
        pred_value = prediction.get('prediction', 18_500_000)
        threshold = 18_500_000  # Market threshold
        predicted_prob = 1 / (1 + np.exp(-(pred_value - threshold) / 500_000))

        market_price = market_data.get('yes_bid', 0.5)
        signal = trading_service.calculate_signal(predicted_prob, market_price)

        if signal['signal'] == 'HOLD':
            st.info(f" **{signal['signal']}**")
        elif 'YES' in signal['signal']:
            st.success(f" **{signal['signal']}**")
        else:
            st.error(f" **{signal['signal']}**")

        st.write(f"Edge: {signal['edge_pct']:+.1f}%")
        st.write(f"Signal strength: {signal['strength'] * 100:.0f}%")

    st.markdown("---")

    # Quick links
    st.subheader("Quick Actions")

    quick_col1, quick_col2, quick_col3 = st.columns(3)

    with quick_col1:
        if st.button("View Predictions", use_container_width=True):
            st.switch_page("pages/1_predictions.py")

    with quick_col2:
        if st.button("Place Trade", use_container_width=True):
            st.switch_page("pages/2_trading.py")

    with quick_col3:
        if st.button("View Portfolio", use_container_width=True):
            st.switch_page("pages/3_portfolio.py")


# Import numpy for sigmoid calculation
import numpy as np

if __name__ == "__main__":
    main()
