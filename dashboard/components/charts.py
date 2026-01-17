"""
Plotly chart components for the dashboard.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


# Color scheme
COLORS = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4',
    'background': '#0E1117',
    'text': '#FAFAFA',
    'green': '#00D26A',
    'red': '#FF4B4B',
    'blue': '#3B82F6',
    'purple': '#8B5CF6',
    'orange': '#F59E0B',
    'gray': '#6B7280'
}


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent dark theme to figure."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text']),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def create_prediction_chart(predictions_df: pd.DataFrame,
                          show_confidence: bool = True) -> go.Figure:
    """
    Create prediction vs actual time series chart.

    Args:
        predictions_df: DataFrame with 'predicted', 'actual', and optionally 'confidence'
        show_confidence: Whether to show confidence bands

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(
        x=predictions_df.index,
        y=predictions_df['actual'],
        name='Actual',
        line=dict(color=COLORS['blue'], width=2),
        mode='lines'
    ))

    # Predicted values
    fig.add_trace(go.Scatter(
        x=predictions_df.index,
        y=predictions_df['predicted'],
        name='Predicted',
        line=dict(color=COLORS['primary'], width=2, dash='dot'),
        mode='lines'
    ))

    # Confidence bands if available
    if show_confidence and 'confidence' in predictions_df.columns:
        uncertainty = predictions_df['predicted'] * (1 - predictions_df['confidence']) * 0.1

        fig.add_trace(go.Scatter(
            x=predictions_df.index.tolist() + predictions_df.index.tolist()[::-1],
            y=(predictions_df['predicted'] + uncertainty).tolist() +
              (predictions_df['predicted'] - uncertainty).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,107,107,0.2)',
            line=dict(color='rgba(255,107,107,0)'),
            name='Confidence Band',
            showlegend=True
        ))

    fig.update_layout(
        title='Predictions vs Actual',
        xaxis_title='Date',
        yaxis_title='Passengers',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x unified'
    )

    return apply_dark_theme(fig)


def create_accuracy_chart(accuracy_df: pd.DataFrame,
                         window: int = 7) -> go.Figure:
    """
    Create accuracy over time chart with rolling average.

    Args:
        accuracy_df: DataFrame with 'accuracy' column
        window: Rolling window size

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Daily accuracy
    fig.add_trace(go.Scatter(
        x=accuracy_df.index,
        y=accuracy_df['accuracy'] * 100,
        name='Daily Accuracy',
        line=dict(color=COLORS['secondary'], width=1),
        mode='lines',
        opacity=0.5
    ))

    # Rolling average
    rolling_acc = accuracy_df['accuracy'].rolling(window=window).mean()
    fig.add_trace(go.Scatter(
        x=accuracy_df.index,
        y=rolling_acc * 100,
        name=f'{window}-Day Average',
        line=dict(color=COLORS['primary'], width=3),
        mode='lines'
    ))

    # Reference line at 85%
    fig.add_hline(y=85, line_dash='dash', line_color=COLORS['gray'],
                  annotation_text='Target (85%)')

    fig.update_layout(
        title='Model Accuracy Over Time',
        xaxis_title='Date',
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[50, 100]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x unified'
    )

    return apply_dark_theme(fig)


def create_model_weights_chart(weights: Dict[str, float]) -> go.Figure:
    """
    Create pie chart of ensemble model weights.

    Args:
        weights: Dict of model name to weight

    Returns:
        Plotly figure
    """
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['blue'],
              COLORS['purple'], COLORS['orange']]

    fig = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        hole=0.4,
        marker=dict(colors=colors[:len(weights)]),
        textinfo='label+percent',
        textposition='outside'
    )])

    fig.update_layout(
        title='Ensemble Model Weights',
        showlegend=False
    )

    return apply_dark_theme(fig)


def create_pnl_chart(pnl_df: pd.DataFrame) -> go.Figure:
    """
    Create cumulative P&L chart with daily bars.

    Args:
        pnl_df: DataFrame with 'daily_pnl' and 'cumulative_pnl'

    Returns:
        Plotly figure
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.1,
                       row_heights=[0.7, 0.3])

    # Cumulative P&L line
    fig.add_trace(go.Scatter(
        x=pnl_df.index,
        y=pnl_df['cumulative_pnl'],
        name='Cumulative P&L',
        line=dict(color=COLORS['secondary'], width=2),
        fill='tozeroy',
        fillcolor='rgba(78,205,196,0.2)'
    ), row=1, col=1)

    # Daily P&L bars
    colors = [COLORS['green'] if x >= 0 else COLORS['red']
              for x in pnl_df['daily_pnl']]

    fig.add_trace(go.Bar(
        x=pnl_df.index,
        y=pnl_df['daily_pnl'],
        name='Daily P&L',
        marker_color=colors
    ), row=2, col=1)

    fig.update_layout(
        title='Profit & Loss',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x unified'
    )

    fig.update_yaxes(title_text='Cumulative ($)', row=1, col=1)
    fig.update_yaxes(title_text='Daily ($)', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    return apply_dark_theme(fig)


def create_order_book_chart(order_book: Dict) -> go.Figure:
    """
    Create order book depth visualization.

    Args:
        order_book: Dict with 'bids' and 'asks' lists

    Returns:
        Plotly figure
    """
    bids = order_book.get('bids', [])
    asks = order_book.get('asks', [])

    # Calculate cumulative sizes
    bid_prices = [b['price'] / 100 for b in bids]
    bid_sizes = [b['size'] for b in bids]
    bid_cumulative = np.cumsum(bid_sizes)

    ask_prices = [a['price'] / 100 for a in asks]
    ask_sizes = [a['size'] for a in asks]
    ask_cumulative = np.cumsum(ask_sizes)

    fig = go.Figure()

    # Bid depth
    fig.add_trace(go.Scatter(
        x=bid_prices,
        y=bid_cumulative,
        name='Bids',
        fill='tozeroy',
        line=dict(color=COLORS['green']),
        fillcolor='rgba(0,210,106,0.3)'
    ))

    # Ask depth
    fig.add_trace(go.Scatter(
        x=ask_prices,
        y=ask_cumulative,
        name='Asks',
        fill='tozeroy',
        line=dict(color=COLORS['red']),
        fillcolor='rgba(255,75,75,0.3)'
    ))

    fig.update_layout(
        title='Order Book Depth',
        xaxis_title='Price',
        yaxis_title='Cumulative Size',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x'
    )

    return apply_dark_theme(fig)


def create_feature_importance_chart(importance_df: pd.DataFrame,
                                   top_n: int = 10) -> go.Figure:
    """
    Create horizontal bar chart of feature importance.

    Args:
        importance_df: DataFrame with 'importance' column
        top_n: Number of top features to show

    Returns:
        Plotly figure
    """
    # Get top features
    top_features = importance_df.nlargest(top_n, 'importance')

    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features.index,
        orientation='h',
        marker_color=COLORS['primary']
    ))

    fig.update_layout(
        title=f'Top {top_n} Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        yaxis=dict(autorange='reversed')
    )

    return apply_dark_theme(fig)


def create_positions_chart(positions: List[Dict]) -> go.Figure:
    """
    Create positions visualization.

    Args:
        positions: List of position dicts

    Returns:
        Plotly figure
    """
    if not positions:
        fig = go.Figure()
        fig.add_annotation(
            text="No open positions",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=COLORS['gray'])
        )
        return apply_dark_theme(fig)

    markets = [p['title'][:30] for p in positions]
    sizes = [p['size'] * (1 if p['side'] == 'yes' else -1) for p in positions]
    pnls = [p.get('unrealized_pnl', 0) for p in positions]

    colors = [COLORS['green'] if s > 0 else COLORS['red'] for s in sizes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=markets,
        y=sizes,
        name='Position Size',
        marker_color=colors,
        text=[f"${p:.2f}" for p in pnls],
        textposition='outside'
    ))

    fig.update_layout(
        title='Open Positions',
        xaxis_title='Market',
        yaxis_title='Position (+ = YES, - = NO)',
        showlegend=False
    )

    return apply_dark_theme(fig)


def create_risk_gauge(value: float, max_value: float,
                     title: str = "Risk Utilization") -> go.Figure:
    """
    Create a gauge chart for risk metrics.

    Args:
        value: Current value
        max_value: Maximum allowed value
        title: Gauge title

    Returns:
        Plotly figure
    """
    pct = (value / max_value * 100) if max_value > 0 else 0

    # Determine color based on utilization
    if pct < 50:
        color = COLORS['green']
    elif pct < 75:
        color = COLORS['orange']
    else:
        color = COLORS['red']

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0,210,106,0.2)'},
                {'range': [50, 75], 'color': 'rgba(245,158,11,0.2)'},
                {'range': [75, 100], 'color': 'rgba(255,75,75,0.2)'}
            ],
            'threshold': {
                'line': {'color': COLORS['red'], 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        },
        number={'suffix': '%'}
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return apply_dark_theme(fig)


def create_equity_curve_chart(equity_curve: List) -> go.Figure:
    """
    Create equity curve line chart for backtest results.

    Args:
        equity_curve: List of (date, equity) tuples or dicts

    Returns:
        Plotly figure
    """
    if not equity_curve:
        fig = go.Figure()
        fig.add_annotation(
            text="No backtest data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS['gray'])
        )
        return apply_dark_theme(fig)

    # Handle both tuple and dict formats
    if isinstance(equity_curve[0], (list, tuple)):
        dates = [item[0] for item in equity_curve]
        equities = [item[1] for item in equity_curve]
    else:
        dates = [item.get('date') for item in equity_curve]
        equities = [item.get('equity') for item in equity_curve]

    df = pd.DataFrame({'date': dates, 'equity': equities})

    # Calculate if we're up or down from start
    if len(equities) > 0:
        start_equity = equities[0]
        final_equity = equities[-1]
        is_profitable = final_equity >= start_equity
    else:
        is_profitable = True

    fig = go.Figure()

    # Main equity line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['equity'],
        mode='lines',
        name='Equity',
        line=dict(color=COLORS['green'] if is_profitable else COLORS['red'], width=2),
        fill='tozeroy',
        fillcolor='rgba(0,210,106,0.1)' if is_profitable else 'rgba(255,75,75,0.1)'
    ))

    # Add starting equity reference line
    if len(equities) > 0:
        fig.add_hline(
            y=equities[0],
            line_dash='dash',
            line_color=COLORS['gray'],
            annotation_text='Start'
        )

    fig.update_layout(
        title='Cumulative Profit Over Time',
        xaxis_title='Date',
        yaxis_title='Equity ($)',
        hovermode='x unified',
        showlegend=False
    )

    return apply_dark_theme(fig)


def create_weekly_profits_chart(weekly_profits: List[Dict]) -> go.Figure:
    """
    Create bar chart of weekly profits/losses.

    Args:
        weekly_profits: List of dicts with 'date' and 'profit' keys

    Returns:
        Plotly figure
    """
    if not weekly_profits:
        fig = go.Figure()
        fig.add_annotation(
            text="No weekly data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS['gray'])
        )
        return apply_dark_theme(fig)

    df = pd.DataFrame(weekly_profits)

    # Color bars based on profit/loss
    colors = [COLORS['green'] if p >= 0 else COLORS['red'] for p in df['profit']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['profit'],
        marker_color=colors,
        name='Weekly P&L'
    ))

    fig.add_hline(y=0, line_color=COLORS['gray'], line_width=1)

    fig.update_layout(
        title='Weekly Profit/Loss',
        xaxis_title='Week',
        yaxis_title='Profit ($)',
        showlegend=False,
        hovermode='x'
    )

    return apply_dark_theme(fig)
