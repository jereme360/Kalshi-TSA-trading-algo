"""
Trading Page - View market data and place trades.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import time

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from dashboard.services.trading_service import get_trading_service
from dashboard.services.model_service import get_model_service
from dashboard.components.charts import create_order_book_chart
from src.trading.contract_selector import ContractSelector
from scipy.stats import norm

st.set_page_config(page_title="Trading", page_icon="", layout="wide")

st.title("Trading")

# Get services
trading_service = get_trading_service()
model_service = get_model_service()

# Connection status
if trading_service.connected:
    st.success("Connected to Kalshi API")
else:
    st.warning("Not connected to Kalshi - Demo Mode")

st.markdown("---")

# =============================================================================
# GET PREDICTION AND RECOMMENDATION FIRST
# =============================================================================
# Get model prediction and convert to daily average
prediction = model_service.get_prediction(None)
weekly_pred = prediction.get('prediction')
weekly_uncertainty = prediction.get('uncertainty')
confidence = prediction.get('confidence', 0) or 0

# Convert to daily (Kalshi trades on daily averages)
daily_pred = weekly_pred / 7 if weekly_pred else None
daily_std = weekly_uncertainty / 7 if weekly_uncertainty else None

# Get available contracts from Kalshi
contracts = trading_service.get_available_contracts()

if not contracts and daily_pred:
    # Demo mode: generate sample contracts based on typical thresholds
    base = round(daily_pred / 50000) * 50000
    contracts = [
        {'ticker': f'DEMO-{int(t/1000000)}.{int((t%1000000)/100000):02d}M', 'threshold': t, 'yes_price': 0.5, 'no_price': 0.5}
        for t in [base - 100000, base - 50000, base, base + 50000, base + 100000, base + 150000]
        if t > 0
    ]

# Get recommendation
recommendation = None
rec_contract = None
if contracts and daily_pred and daily_std:
    selector = ContractSelector(min_ev_threshold=0.01)
    recommendation = selector.select_contract(
        prediction=daily_pred,
        prediction_std=daily_std,
        contracts=contracts
    )
    if recommendation.get('contract'):
        rec_contract = next((c for c in contracts if c['ticker'] == recommendation['contract']), None)

# =============================================================================
# OPTIMAL CONTRACT SECTION (Replaces generic Current Market)
# =============================================================================
st.header("Optimal Contract")

if recommendation and rec_contract:
    threshold = rec_contract['threshold']
    side = recommendation['side'].upper()
    ev = recommendation['expected_value']
    rec_confidence = recommendation['confidence']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Recommendation")
        if ev >= 0.02:
            st.success(f"### BUY {side}")
        elif ev >= 0:
            st.warning(f"### BUY {side} (Low EV)")
        else:
            st.info("### HOLD")

        st.write(f"**Threshold**: {threshold:,}")
        st.write(f"**Ticker**: {rec_contract['ticker']}")

    with col2:
        st.subheader("Contract Prices")
        yes_price = rec_contract.get('yes_price', 0.5)
        no_price = rec_contract.get('no_price', 0.5)
        yes_ask = rec_contract.get('yes_ask', yes_price)
        no_ask = rec_contract.get('no_ask', no_price)

        st.metric("YES Price", f"${yes_price:.2f} / ${yes_ask:.2f}", help="Bid / Ask")
        st.metric("NO Price", f"${no_price:.2f} / ${no_ask:.2f}", help="Bid / Ask")

    with col3:
        st.subheader("Trade Metrics")
        st.metric("Expected Value", f"{ev:+.1%}")
        st.metric("Confidence", f"{rec_confidence:.0%}")
        st.write(f"**Volume**: {rec_contract.get('volume', 0):,}")

    st.caption(recommendation.get('reasoning', ''))

else:
    # Fallback to generic market data if no recommendation
    market_data = trading_service.get_market_data()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Market Info")
        market_title = market_data.get('title', 'TSA Weekly')
        market_id = market_data.get('market_id', 'N/A')
        st.write(f"**Market**: {market_title}")
        st.write(f"**Ticker**: {market_id}")

        expiration = market_data.get('expiration')
        if expiration:
            try:
                exp_date = datetime.fromisoformat(expiration.replace('Z', '+00:00'))
                days_to_exp = (exp_date - datetime.now(exp_date.tzinfo)).days
                st.write(f"**Expires**: {days_to_exp} days")
            except:
                pass

    with col2:
        st.subheader("Prices")
        yes_bid = market_data.get('yes_bid', 0) or 0
        yes_ask = market_data.get('yes_ask', 0) or 0
        spread = yes_ask - yes_bid if yes_ask and yes_bid else 0

        st.metric("YES Bid/Ask", f"${yes_bid:.2f} / ${yes_ask:.2f}")
        st.write(f"Spread: ${spread:.2f}")
        st.write(f"Last: ${market_data.get('last_price', 0) or 0:.2f}")

    with col3:
        st.subheader("Volume")
        st.metric("Volume", f"{market_data.get('volume', 0) or 0:,}")
        st.metric("Open Interest", f"{market_data.get('open_interest', 0) or 0:,}")

    st.info("No contract recommendation available - connect to Kalshi or wait for prediction")

st.markdown("---")

# =============================================================================
# ORDER BOOK SECTION
# =============================================================================
st.header("Order Book")

order_book = trading_service.get_order_book()

col1, col2 = st.columns([2, 1])

with col1:
    fig = create_order_book_chart(order_book)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top of Book")

    bids = order_book.get('bids', [])[:5]
    asks = order_book.get('asks', [])[:5]

    if bids:
        st.write("**Bids**")
        for bid in bids:
            st.write(f"${bid['price']/100:.2f} - {bid['size']} contracts")
    else:
        st.write("**Bids**: None")

    st.write("")

    if asks:
        st.write("**Asks**")
        for ask in asks:
            st.write(f"${ask['price']/100:.2f} - {ask['size']} contracts")
    else:
        st.write("**Asks**: None")

st.markdown("---")

# =============================================================================
# MODEL PREDICTION SUMMARY
# =============================================================================
st.header("Model Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    if daily_pred:
        st.metric("Predicted Daily Avg", f"{daily_pred:,.0f}")
    else:
        st.metric("Predicted Daily Avg", "N/A")

with col2:
    if daily_std:
        st.metric("Uncertainty (1 std)", f"±{daily_std:,.0f}")
    else:
        st.metric("Uncertainty", "N/A")

with col3:
    st.metric("Confidence", f"{confidence*100:.1f}%")

st.markdown("---")

# =============================================================================
# ALL CONTRACTS TABLE
# =============================================================================
st.header("Contract Analysis")

if contracts and daily_pred and daily_std:
    st.write(f"**Model Prediction**: {daily_pred:,.0f} daily average (±{daily_std:,.0f})")

    # Build analysis table
    analysis_data = []
    for contract in contracts:
        threshold = contract['threshold']
        yes_price = contract.get('yes_price', 0.5)
        no_price = contract.get('no_price', 0.5)

        # Calculate probability that actual > threshold
        prob_above = 1 - norm.cdf(threshold, loc=daily_pred, scale=daily_std)
        prob_below = 1 - prob_above

        # Calculate EV for YES and NO
        ev_yes = prob_above - yes_price
        ev_no = prob_below - no_price

        # Determine best side
        if ev_yes > ev_no and ev_yes > 0:
            best_side = "YES"
            best_ev = ev_yes
        elif ev_no > ev_yes and ev_no > 0:
            best_side = "NO"
            best_ev = ev_no
        else:
            best_side = "-"
            best_ev = max(ev_yes, ev_no)

        analysis_data.append({
            'Threshold': f"{threshold:,}",
            'YES Price': f"${yes_price:.2f}",
            'NO Price': f"${no_price:.2f}",
            'P(Above)': f"{prob_above:.0%}",
            'P(Below)': f"{prob_below:.0%}",
            'Best Side': best_side,
            'EV': f"{best_ev:+.1%}"
        })

    analysis_df = pd.DataFrame(analysis_data)
    st.dataframe(analysis_df, use_container_width=True, hide_index=True)

    # Highlight the best contract
    best_row = max(analysis_data, key=lambda x: float(x['EV'].replace('%', '').replace('+', '')) / 100)
    if float(best_row['EV'].replace('%', '').replace('+', '')) > 0:
        st.success(f"**Best Trade**: BUY {best_row['Best Side']} on {best_row['Threshold']} threshold (EV: {best_row['EV']})")

else:
    st.info("Connect to Kalshi API or wait for predictions to see contract analysis")

st.markdown("---")

# =============================================================================
# ORDER FORM SECTION
# =============================================================================
st.header("Place Order")

# Pre-fill with recommendation if available
default_side = 'YES'
default_side_index = 0
if recommendation and recommendation.get('side'):
    default_side = recommendation['side'].upper()
    default_side_index = 0 if default_side == 'YES' else 1

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Order Parameters")

    if recommendation and rec_contract:
        st.caption(f"Pre-filled with recommendation: {default_side} on {rec_contract['threshold']:,}")

    side = st.radio(
        "Side",
        options=['YES', 'NO'],
        index=default_side_index,
        horizontal=True,
        help="YES = bet volume exceeds threshold, NO = bet volume below threshold"
    )

    size = st.slider(
        "Contracts",
        min_value=1,
        max_value=1000,
        value=10,
        help="Maximum 1000 contracts per trade"
    )

    # Use market price from recommended contract
    if rec_contract:
        if side == 'YES':
            price = rec_contract.get('yes_ask', rec_contract.get('yes_price', 0.50))
        else:
            price = rec_contract.get('no_ask', rec_contract.get('no_price', 0.50))
    else:
        # Fallback to generic market data
        market_data = trading_service.get_market_data()
        yes_ask = market_data.get('yes_ask', 0.50) or 0.50
        if side == 'YES':
            price = yes_ask
        else:
            price = 1 - yes_ask

    st.metric("Market Price", f"${price:.2f}")

with col2:
    st.subheader("Order Preview")

    # Calculate cost and potential P&L
    cost = size * price
    max_profit = size * (1 - price)
    max_loss = cost

    st.write(f"**Side**: {side}")
    st.write(f"**Size**: {size} contracts")
    st.write(f"**Price**: ${price:.2f}")
    st.write(f"**Cost**: ${cost:.2f}")
    st.write(f"**Max Profit**: ${max_profit:.2f}")
    st.write(f"**Max Loss**: ${max_loss:.2f}")

    # Risk/reward
    rr_ratio = max_profit / max_loss if max_loss > 0 else 0
    st.write(f"**Risk/Reward**: 1:{rr_ratio:.1f}")

# Risk Warning
st.markdown("---")

with st.expander("Risk Warning", expanded=False):
    st.warning("""
    **Trading involves risk of loss.**

    - Prediction markets can result in total loss of investment
    - Past model performance does not guarantee future results
    - Maximum position: 1000 contracts per trade

    Only trade with funds you can afford to lose.
    """)

# Confirmation and Submit
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    confirmed = st.checkbox(
        "I understand the risks and confirm this order",
        value=False
    )

with col2:
    # Validate order
    valid, validation_msg = trading_service.validate_order(side.lower(), size, price)

    if not valid:
        st.error(validation_msg)
    else:
        st.success(validation_msg)

with col3:
    submit_disabled = not confirmed or not valid

    # Get ticker from recommended contract
    order_ticker = rec_contract.get('ticker') if rec_contract else None

    if st.button("Submit Order", disabled=submit_disabled, type="primary", use_container_width=True):
        if not order_ticker:
            st.error("No contract selected. Please wait for contract recommendation.")
        else:
            with st.spinner("Placing order..."):
                time.sleep(1)  # Minimum delay for safety

                result = trading_service.place_order(
                    side=side.lower(),
                    size=size,
                    price=price,
                    confirmed=confirmed,
                    ticker=order_ticker
                )

                if result.get('success'):
                    if result.get('demo'):
                        st.info(f"Demo order placed: {result.get('order_id')}")
                    else:
                        st.success(f"Order placed: {result.get('order_id')}")
                    st.json(result)
                else:
                    st.error(f"Order failed: {result.get('error')}")

# Footer
st.markdown("---")
st.caption("Orders are executed through Kalshi Exchange. Market data may be delayed.")
