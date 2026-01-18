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
from src.trading.contract_selector import ContractSelector

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
    col1, col2, col3 = st.columns(3)

    # Get latest data
    latest_tsa = get_latest_tsa_value()
    model_perf = model_service.get_recent_performance()
    balance = trading_service.get_account_balance()

    with col1:
        st.metric(
            "Latest TSA Passengers",
            format_number(latest_tsa.get('passengers')),
            format_pct(latest_tsa.get('yoy_change'))
        )

    with col2:
        st.metric(
            "Recent Accuracy (30d)",
            f"{model_perf.get('mean_accuracy', 0) * 100:.1f}%"
        )

    with col3:
        st.metric(
            "Account Balance",
            f"${balance.get('balance', 0):,.2f}",
            f"${balance.get('available', 0):,.2f} available"
        )

    st.markdown("---")

    # Current prediction summary
    col1, col2 = st.columns([2, 1])

    prediction = model_service.get_prediction(None)

    # Convert to daily averages (Kalshi trades on daily average)
    weekly_pred = prediction.get('prediction')
    weekly_lower = prediction.get('lower_bound')
    weekly_upper = prediction.get('upper_bound')
    weekly_uncertainty = prediction.get('uncertainty')

    daily_pred = weekly_pred / 7 if weekly_pred else None
    daily_lower = weekly_lower / 7 if weekly_lower else None
    daily_upper = weekly_upper / 7 if weekly_upper else None
    daily_std = weekly_uncertainty / 7 if weekly_uncertainty else None

    with col1:
        st.subheader("Current Prediction (Daily Average)")

        pred_col1, pred_col2, pred_col3 = st.columns(3)

        with pred_col1:
            st.metric(
                "Daily Passenger Forecast",
                format_number(daily_pred) if daily_pred else "N/A"
            )

        with pred_col2:
            conf = prediction.get('confidence', 0)
            st.metric(
                "Confidence",
                f"{conf * 100:.1f}%" if conf else "N/A"
            )

        with pred_col3:
            if daily_lower and daily_upper:
                # Use millions format to avoid truncation
                lower_m = daily_lower / 1_000_000
                upper_m = daily_upper / 1_000_000
                st.metric(
                    "95% Interval",
                    f"{lower_m:.2f}M - {upper_m:.2f}M"
                )
            else:
                st.metric("95% Interval", "N/A")

    # Get contract recommendation
    contracts = trading_service.get_available_contracts()

    # Generate demo contracts if not connected
    if not contracts and daily_pred:
        base = round(daily_pred / 50000) * 50000
        contracts = [
            {'ticker': f'DEMO-{int(t/1000000)}.{int((t%1000000)/100000):02d}M', 'threshold': t, 'yes_price': 0.5, 'no_price': 0.5}
            for t in [base - 100000, base - 50000, base, base + 50000, base + 100000, base + 150000]
            if t > 0
        ]

    # Get recommendation from ContractSelector
    recommendation = None
    if contracts and daily_pred and daily_std:
        selector = ContractSelector(min_ev_threshold=0.01)
        recommendation = selector.select_contract(
            prediction=daily_pred,
            prediction_std=daily_std,
            contracts=contracts
        )

    with col2:
        st.subheader("Trading Recommendation")

        if recommendation and recommendation.get('contract'):
            # Find the threshold for the recommended contract
            rec_contract = next((c for c in contracts if c['ticker'] == recommendation['contract']), None)
            threshold = rec_contract['threshold'] if rec_contract else 0

            side = recommendation['side'].upper()
            ev = recommendation['expected_value']
            confidence = recommendation['confidence']

            if ev >= 0.02:
                st.success(f"**BUY {side}** on {format_number(threshold)} threshold")
                st.write(f"**Expected Value**: {ev:.1%}")
                st.write(f"**Confidence**: {confidence:.0%}")
                st.caption(recommendation.get('reasoning', ''))
            elif ev >= 0:
                st.warning(f"**BUY {side}** (Low EV)")
                st.write(f"Contract: {format_number(threshold)} threshold")
                st.write(f"Expected Value: {ev:.1%}")
                st.caption(recommendation.get('reasoning', ''))
            else:
                st.info("**HOLD** - No high-EV contracts")
                st.caption(recommendation.get('reasoning', ''))
        else:
            reason = recommendation.get('reasoning', 'No contracts available') if recommendation else 'Awaiting prediction data'
            st.info("**HOLD**")
            st.caption(reason)

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


if __name__ == "__main__":
    main()
