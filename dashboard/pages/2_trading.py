"""
Trading Page - Place bets and view market data.
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

st.set_page_config(page_title="Trading", page_icon="", layout="wide")

st.title("Trading")
st.markdown("Place trades on Kalshi TSA markets")

# Get services
trading_service = get_trading_service()
model_service = get_model_service()

# Connection status
if trading_service.connected:
    st.success("Connected to Kalshi API")
else:
    st.warning("Demo Mode - Orders will not be executed")

st.markdown("---")

# Market Data Section
st.header("Market Data")

market_data = trading_service.get_market_data()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Market Info")
    st.write(f"**Market**: {market_data.get('title', 'TSA Weekly')}")
    st.write(f"**Market ID**: {market_data.get('market_id', 'N/A')}")

    expiration = market_data.get('expiration')
    if expiration:
        exp_date = datetime.fromisoformat(expiration.replace('Z', '+00:00'))
        days_to_exp = (exp_date - datetime.now(exp_date.tzinfo)).days
        st.write(f"**Expires**: {days_to_exp} days")

with col2:
    st.subheader("Prices")

    yes_bid = market_data.get('yes_bid', 0)
    yes_ask = market_data.get('yes_ask', 0)
    spread = yes_ask - yes_bid

    st.metric("YES Bid/Ask", f"${yes_bid:.2f} / ${yes_ask:.2f}")
    st.write(f"Spread: ${spread:.2f}")
    st.write(f"Last: ${market_data.get('last_price', 0):.2f}")

with col3:
    st.subheader("Volume")
    st.metric("Volume", f"{market_data.get('volume', 0):,}")
    st.metric("Open Interest", f"{market_data.get('open_interest', 0):,}")

st.markdown("---")

# Order Book
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

    st.write("**Bids**")
    for bid in bids:
        st.write(f"${bid['price']/100:.2f} - {bid['size']} contracts")

    st.write("")
    st.write("**Asks**")
    for ask in asks:
        st.write(f"${ask['price']/100:.2f} - {ask['size']} contracts")

st.markdown("---")

# Trading Signal
st.header("Trading Signal")

col1, col2 = st.columns([1, 1])

with col1:
    # Get model prediction
    prediction = model_service.get_prediction(None)
    pred_value = prediction.get('prediction', 18_500_000)

    # Convert to probability (simplified)
    threshold = 18_500_000  # This should come from the market definition
    predicted_prob = 1 / (1 + np.exp(-(pred_value - threshold) / 500_000))

    market_price = market_data.get('yes_bid', 0.5)
    signal = trading_service.calculate_signal(predicted_prob, market_price)

    # Display signal
    if signal['signal'] == 'HOLD':
        st.info(f"### Signal: **{signal['signal']}**")
    elif 'YES' in signal['signal']:
        st.success(f"### Signal: **{signal['signal']}**")
    else:
        st.error(f"### Signal: **{signal['signal']}**")

    st.write(f"**Model Probability**: {signal['predicted_prob']:.1%}")
    st.write(f"**Market Price**: {signal['market_price']:.1%}")
    st.write(f"**Edge**: {signal['edge_pct']:+.1f}%")
    st.write(f"**Signal Strength**: {signal['strength']*100:.0f}%")

with col2:
    st.subheader("Confidence")
    confidence = prediction.get('confidence', 0)

    # Confidence gauge
    st.metric("Model Confidence", f"{confidence*100:.1f}%")

    if confidence > 0.8:
        st.success("High confidence prediction")
    elif confidence > 0.6:
        st.info("Moderate confidence prediction")
    else:
        st.warning("Low confidence - trade with caution")

st.markdown("---")

# Order Form
st.header("Place Order")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Order Parameters")

    side = st.radio(
        "Side",
        options=['YES', 'NO'],
        horizontal=True,
        help="YES = bet price goes up, NO = bet price goes down"
    )

    size = st.slider(
        "Contracts",
        min_value=1,
        max_value=100,
        value=10,
        help="Maximum 100 contracts per trade"
    )

    # Price suggestion based on order book
    if side == 'YES':
        suggested_price = market_data.get('yes_ask', 0.65)
    else:
        suggested_price = 1 - market_data.get('yes_bid', 0.62)

    price = st.number_input(
        "Limit Price",
        min_value=0.01,
        max_value=0.99,
        value=round(suggested_price, 2),
        step=0.01,
        format="%.2f"
    )

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

with st.expander("Risk Warning", expanded=True):
    st.warning("""
    **Trading involves risk of loss.**

    - Prediction markets can result in total loss of investment
    - Past model performance does not guarantee future results
    - Maximum position: 100 contracts per trade
    - 5-second cooldown between trades enforced

    Only trade with funds you can afford to lose.
    """)

# Confirmation and Submit
st.markdown("---")

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

    if st.button("Submit Order", disabled=submit_disabled, type="primary", use_container_width=True):
        with st.spinner("Placing order..."):
            time.sleep(1)  # Minimum delay for safety

            result = trading_service.place_order(
                side=side.lower(),
                size=size,
                price=price,
                confirmed=confirmed
            )

            if result.get('success'):
                if result.get('demo'):
                    st.info(f"Demo order placed: {result.get('order_id')}")
                else:
                    st.success(f"Order placed: {result.get('order_id')}")

                st.json(result)
            else:
                st.error(f"Order failed: {result.get('error')}")

# Recent Orders
st.markdown("---")
st.header("Recent Orders")

trades = trading_service.get_trade_history(days=7)

if not trades.empty:
    st.dataframe(
        trades.tail(10).reset_index(),
        use_container_width=True,
        column_config={
            'timestamp': st.column_config.DatetimeColumn('Time'),
            'side': st.column_config.TextColumn('Side'),
            'size': st.column_config.NumberColumn('Size'),
            'price': st.column_config.NumberColumn('Price', format="$%.2f"),
            'pnl': st.column_config.NumberColumn('P&L', format="$%.2f"),
            'status': st.column_config.TextColumn('Status')
        }
    )
else:
    st.info("No recent trades")

# Footer
st.markdown("---")
st.caption("Orders are executed through Kalshi Exchange. Market data may be delayed.")
