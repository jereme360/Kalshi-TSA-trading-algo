"""
Portfolio Page - Positions and P&L tracking.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from dashboard.services.trading_service import get_trading_service
from dashboard.components.charts import (
    create_pnl_chart,
    create_positions_chart,
    create_risk_gauge
)

st.set_page_config(page_title="Portfolio", page_icon="", layout="wide")

st.title("Portfolio")
st.markdown("Track positions, P&L, and account performance")

# Get services
trading_service = get_trading_service()

# Account Summary
st.header("Account Summary")

balance = trading_service.get_account_balance()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Balance",
        f"${balance.get('balance', 0):,.2f}"
    )

with col2:
    st.metric(
        "Available",
        f"${balance.get('available', 0):,.2f}"
    )

with col3:
    st.metric(
        "Reserved",
        f"${balance.get('reserved', 0):,.2f}"
    )

with col4:
    positions_count = balance.get('open_positions', 0)
    st.metric(
        "Open Positions",
        positions_count
    )

st.markdown("---")

# Open Positions
st.header("Open Positions")

positions = trading_service.get_positions()

if positions:
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = create_positions_chart(positions)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Position summary
        total_invested = sum(p['size'] * p['avg_price'] for p in positions)
        total_unrealized = sum(p.get('unrealized_pnl', 0) for p in positions)

        st.metric("Total Invested", f"${total_invested:.2f}")
        st.metric(
            "Unrealized P&L",
            f"${total_unrealized:.2f}",
            delta=f"{(total_unrealized/total_invested*100):.1f}%" if total_invested > 0 else None
        )

    # Positions table
    st.subheader("Position Details")

    positions_df = pd.DataFrame(positions)

    st.dataframe(
        positions_df,
        use_container_width=True,
        column_config={
            'market_id': st.column_config.TextColumn('Market ID'),
            'title': st.column_config.TextColumn('Market'),
            'side': st.column_config.TextColumn('Side'),
            'size': st.column_config.NumberColumn('Contracts'),
            'avg_price': st.column_config.NumberColumn('Avg Price', format="$%.2f"),
            'current_price': st.column_config.NumberColumn('Current', format="$%.2f"),
            'unrealized_pnl': st.column_config.NumberColumn('Unrealized P&L', format="$%.2f"),
            'expiration': st.column_config.DatetimeColumn('Expiration')
        },
        hide_index=True
    )

else:
    st.info("No open positions")

st.markdown("---")

# P&L Chart
st.header("Profit & Loss")

col1, col2 = st.columns([3, 1])

with col1:
    lookback = st.selectbox(
        "Period",
        options=[7, 14, 30, 60, 90],
        index=2,
        format_func=lambda x: f"{x} days"
    )

pnl_df = trading_service.get_pnl_history(days=lookback)

if not pnl_df.empty:
    fig = create_pnl_chart(pnl_df)
    st.plotly_chart(fig, use_container_width=True)

    # P&L Statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_pnl = pnl_df['cumulative_pnl'].iloc[-1]
        st.metric(
            "Total P&L",
            f"${total_pnl:.2f}",
            delta=f"{(total_pnl/balance.get('balance', 10000)*100):.1f}%"
        )

    with col2:
        daily_avg = pnl_df['daily_pnl'].mean()
        st.metric(
            "Daily Average",
            f"${daily_avg:.2f}"
        )

    with col3:
        best_day = pnl_df['daily_pnl'].max()
        st.metric(
            "Best Day",
            f"${best_day:.2f}"
        )

    with col4:
        worst_day = pnl_df['daily_pnl'].min()
        st.metric(
            "Worst Day",
            f"${worst_day:.2f}"
        )

else:
    st.info("No P&L history available")

st.markdown("---")

# Risk Metrics
st.header("Risk Metrics")

col1, col2, col3 = st.columns(3)

# Calculate risk metrics from positions
if positions:
    total_exposure = sum(p['size'] * p['current_price'] for p in positions)
    max_position = max(p['size'] for p in positions)
    capital = balance.get('balance', 10000)

    with col1:
        fig = create_risk_gauge(
            total_exposure,
            capital * 0.5,  # 50% max exposure
            "Exposure"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_risk_gauge(
            max_position,
            1000,  # Max 1000 contracts
            "Largest Position"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        concentration = max_position / sum(p['size'] for p in positions) * 100 if positions else 0
        fig = create_risk_gauge(
            concentration,
            100,
            "Concentration"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Open positions to see risk metrics")

st.markdown("---")

# Trade History
st.header("Trade History")

col1, col2 = st.columns([3, 1])

with col1:
    history_days = st.selectbox(
        "Show history",
        options=[7, 14, 30, 60],
        index=2,
        format_func=lambda x: f"Last {x} days",
        key="history_days"
    )

with col2:
    export_btn = st.button("Export to CSV", use_container_width=True)

trades_df = trading_service.get_trade_history(days=history_days)

if not trades_df.empty:
    st.dataframe(
        trades_df.reset_index(),
        use_container_width=True,
        column_config={
            'timestamp': st.column_config.DatetimeColumn('Time'),
            'side': st.column_config.TextColumn('Side'),
            'size': st.column_config.NumberColumn('Contracts'),
            'price': st.column_config.NumberColumn('Price', format="$%.2f"),
            'pnl': st.column_config.NumberColumn('P&L', format="$%.2f"),
            'status': st.column_config.TextColumn('Status')
        }
    )

    # Export functionality
    if export_btn:
        csv = trades_df.to_csv()
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    # Trade statistics
    st.subheader("Trade Statistics")

    stat_cols = st.columns(4)

    with stat_cols[0]:
        total_trades = len(trades_df)
        st.metric("Total Trades", total_trades)

    with stat_cols[1]:
        winning_trades = (trades_df['pnl'] > 0).sum()
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")

    with stat_cols[2]:
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean())
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        st.metric("Profit Factor", f"{profit_factor:.2f}")

    with stat_cols[3]:
        total_volume = trades_df['size'].sum()
        st.metric("Total Volume", f"{total_volume:,}")

else:
    st.info("No trade history available")

# Footer
st.markdown("---")
st.caption("Portfolio data is updated in real-time when connected to Kalshi API")
