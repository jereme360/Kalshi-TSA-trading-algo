"""
Predictions Page - Model predictions and historical performance.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from dashboard.services.model_service import get_model_service
from dashboard.services.data_service import load_tsa_data
from dashboard.components.charts import (
    create_prediction_chart,
    create_feature_importance_chart
)

st.set_page_config(page_title="Predictions", page_icon="", layout="wide")

st.title("TSA Daily Predictions")
st.caption("Kalshi markets trade on daily averages, not weekly totals")


def format_number(num, decimals=0):
    if num is None:
        return "N/A"
    if decimals == 0:
        return f"{int(num):,}"
    return f"{num:,.{decimals}f}"


# Get services
model_service = get_model_service()

# =============================================================================
# CURRENT FORECAST SECTION
# =============================================================================
st.header("This Week's Forecast")

prediction = model_service.get_prediction(None)

# Convert weekly to daily averages (Kalshi trades on daily averages)
weekly_pred = prediction.get('prediction')
weekly_lower = prediction.get('lower_bound')
weekly_upper = prediction.get('upper_bound')
weekly_uncertainty = prediction.get('uncertainty')

daily_pred = weekly_pred / 7 if weekly_pred else None
daily_lower = weekly_lower / 7 if weekly_lower else None
daily_upper = weekly_upper / 7 if weekly_upper else None
daily_uncertainty = weekly_uncertainty / 7 if weekly_uncertainty else None

pred_cols = st.columns(4)

with pred_cols[0]:
    st.metric(
        "Predicted Daily Avg",
        format_number(daily_pred)
    )

with pred_cols[1]:
    conf = prediction.get('confidence', 0)
    st.metric(
        "Confidence",
        f"{conf * 100:.1f}%" if conf else "N/A"
    )

with pred_cols[2]:
    if daily_lower and daily_upper:
        # Use millions format to avoid truncation
        lower_m = daily_lower / 1_000_000
        upper_m = daily_upper / 1_000_000
        st.metric(
            "95% Range",
            f"{lower_m:.2f}M - {upper_m:.2f}M"
        )

with pred_cols[3]:
    if daily_uncertainty:
        st.metric(
            "Uncertainty (±)",
            f"±{format_number(daily_uncertainty)}",
            help="Standard deviation of prediction"
        )

st.markdown("---")

# =============================================================================
# HISTORICAL PERFORMANCE SECTION
# =============================================================================
st.header("Historical Performance (Since Jan 2023)")

# Load historical data and compute predictions
@st.cache_data(ttl=3600)
def get_historical_performance():
    """Compute historical predictions vs actuals from Jan 2023."""
    tsa_data = load_tsa_data(days=2000)  # Get all available data
    if tsa_data.empty:
        return pd.DataFrame()

    # Get passenger column
    col = 'passengers' if 'passengers' in tsa_data.columns else 'current_year'
    if col not in tsa_data.columns:
        return pd.DataFrame()

    # Calculate weekly totals (use ALL data for predictions)
    weekly_data = tsa_data[col].resample('W-SUN').sum()

    if len(weekly_data) < 60:
        return pd.DataFrame()

    results = []

    # Start from week 52 (need history) and compute for all weeks
    for i in range(52, len(weekly_data) - 1):  # Skip last (potentially incomplete) week
        try:
            week_date = weekly_data.index[i]

            # Only OUTPUT results from Jan 2023 onwards
            if week_date < pd.Timestamp('2023-01-01'):
                continue

            actual = weekly_data.iloc[i]

            # Skip incomplete weeks (less than 10M likely means incomplete data)
            if actual < 10000000:
                continue

            # Prediction: weighted combination of same-week-LY + recent trend
            same_week_ly = weekly_data.iloc[i - 52]
            recent_avg = weekly_data.iloc[max(0, i-4):i].mean()

            # YoY adjustment
            if i >= 104:
                prev_year_avg = weekly_data.iloc[i-104:i-52].mean()
                curr_year_avg = weekly_data.iloc[i-52:i].mean()
                yoy_factor = curr_year_avg / prev_year_avg if prev_year_avg > 0 else 1.0
            else:
                yoy_factor = 1.0

            predicted = 0.3 * same_week_ly * yoy_factor + 0.7 * recent_avg

            # Accuracy
            accuracy = 1 - abs(predicted - actual) / actual
            accuracy = max(0, min(1, accuracy))

            results.append({
                'date': week_date,
                'predicted': predicted,
                'actual': actual,
                'accuracy': accuracy,
                'error': predicted - actual,
                'error_pct': (predicted - actual) / actual * 100
            })

        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.set_index('date', inplace=True)
    return df


historical_df = get_historical_performance()

if not historical_df.empty:
    # Summary metrics
    metric_cols = st.columns(5)

    with metric_cols[0]:
        st.metric(
            "Weeks Analyzed",
            len(historical_df)
        )

    with metric_cols[1]:
        mean_acc = historical_df['accuracy'].mean()
        st.metric(
            "Mean Accuracy",
            f"{mean_acc * 100:.1f}%"
        )

    with metric_cols[2]:
        # Convert weekly RMSE to daily (divide by 7)
        rmse_daily = np.sqrt((historical_df['error'] ** 2).mean()) / 7
        st.metric(
            "RMSE (Daily)",
            format_number(rmse_daily)
        )

    with metric_cols[3]:
        mae_daily = historical_df['error'].abs().mean() / 7
        st.metric(
            "MAE (Daily)",
            format_number(mae_daily)
        )

    with metric_cols[4]:
        mape = (historical_df['error'].abs() / historical_df['actual']).mean() * 100
        st.metric(
            "MAPE",
            f"{mape:.1f}%"
        )

    st.markdown("---")

    # Predictions vs Actuals Chart (convert to daily for display)
    st.subheader("Predicted vs Actual Daily Average")

    # Create daily version of the data for charting
    daily_hist_df = historical_df.copy()
    daily_hist_df['predicted'] = daily_hist_df['predicted'] / 7
    daily_hist_df['actual'] = daily_hist_df['actual'] / 7

    fig = create_prediction_chart(daily_hist_df)
    st.plotly_chart(fig, use_container_width=True)

    # Recent performance table (show daily averages)
    st.subheader("Recent Weeks (Daily Averages)")

    recent_df = historical_df.tail(12).copy()
    recent_df = recent_df.reset_index()
    recent_df['date'] = recent_df['date'].dt.strftime('%Y-%m-%d')
    # Convert to daily averages
    recent_df['predicted'] = recent_df['predicted'].apply(lambda x: f"{x/7:,.0f}")
    recent_df['actual'] = recent_df['actual'].apply(lambda x: f"{x/7:,.0f}")
    recent_df['accuracy'] = recent_df['accuracy'].apply(lambda x: f"{x*100:.1f}%")
    recent_df['error_pct'] = recent_df['error_pct'].apply(lambda x: f"{x:+.1f}%")

    st.dataframe(
        recent_df[['date', 'predicted', 'actual', 'accuracy', 'error_pct']].rename(columns={
            'date': 'Week Ending',
            'predicted': 'Predicted Daily',
            'actual': 'Actual Daily',
            'accuracy': 'Accuracy',
            'error_pct': 'Error %'
        }),
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning("Historical data not available. TSA data may still be loading.")

st.markdown("---")

# =============================================================================
# FEATURE IMPORTANCE SECTION
# =============================================================================
st.header("Prediction Factors")

importance_df = model_service.get_feature_importance()

if not importance_df.empty:
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = create_feature_importance_chart(importance_df, top_n=6)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        **Key Factors:**
        - **Recent Weekly Avg**: Most recent 4-week average
        - **Same Week Last Year**: Seasonality adjustment
        - **YoY Growth Trend**: Year-over-year volume changes
        - **Weekly Seasonality**: Day-of-week patterns
        - **Holiday Proximity**: Impact of holidays
        - **Recent Volatility**: Short-term variance
        """)

# Footer
st.markdown("---")
st.caption("Data updated daily from TSA checkpoint data. Predictions use historical patterns and trend analysis.")
