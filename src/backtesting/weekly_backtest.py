"""
Weekly Monday-only backtesting engine from 2022.
Simulates trading TSA prediction contracts every Monday.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

from src.trading.contract_selector import ContractSelector

logger = logging.getLogger(__name__)


class WeeklyBacktestEngine:
    """Weekly Monday-only backtesting from 2022."""

    def __init__(self, initial_capital: float = 100, bet_size: float = 100):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital (default $100)
            bet_size: Amount to bet per week (default $100)
        """
        self.initial_capital = initial_capital
        self.bet_size = bet_size
        self.contract_selector = ContractSelector(min_ev_threshold=0.02)

    def run(
        self,
        tsa_data: pd.DataFrame,
        start_date: str = '2022-01-03',
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Run backtest from start_date to end_date (or today).

        Args:
            tsa_data: DataFrame with daily TSA passenger data (index=date, column='passengers' or 'current_year')
            start_date: First Monday to start backtesting (default: Jan 3, 2022)
            end_date: Last date to include (default: today)

        Returns:
            Dict with:
                - equity_curve: List of (date, equity) tuples
                - weekly_profits: List of (date, profit) dicts
                - total_profit: float
                - win_rate: float
                - sharpe_ratio: float
                - num_weeks: int
                - avg_profit: float
        """
        if tsa_data.empty:
            return self._empty_results()

        # Get passenger column
        col = 'passengers' if 'passengers' in tsa_data.columns else 'current_year'
        if col not in tsa_data.columns:
            logger.error("No passenger data column found")
            return self._empty_results()

        # Generate list of Mondays
        mondays = self._get_mondays(start_date, end_date)
        if not mondays:
            return self._empty_results()

        results = []
        equity = self.initial_capital
        weekly_returns = []

        for monday in mondays:
            try:
                # Get data available on Monday (only prior data for prediction)
                train_data = tsa_data[tsa_data.index < monday]
                if len(train_data) < 365:  # Need at least 1 year of data
                    continue

                # Predict this week's total using simple model
                prediction, uncertainty = self._predict_week(train_data, col, monday)

                # Get actual week total (Mon-Sun)
                week_end = monday + timedelta(days=6)
                week_mask = (tsa_data.index >= monday) & (tsa_data.index <= week_end)
                week_data = tsa_data.loc[week_mask, col]

                if len(week_data) < 7:
                    continue  # Skip incomplete weeks

                actual = week_data.sum()

                # Simulate contract prices for this week
                simulated_contracts = self._simulate_contracts(actual, prediction, uncertainty)

                # Use contract selector to pick best contract
                recommendation = self.contract_selector.select_contract(
                    prediction=prediction,
                    prediction_std=uncertainty,
                    contracts=simulated_contracts
                )

                # Calculate profit based on contract outcome
                week_profit = self._calculate_week_profit(
                    recommendation, actual, bet_size=self.bet_size
                )

                equity += week_profit
                week_return = week_profit / self.bet_size if self.bet_size > 0 else 0
                weekly_returns.append(week_return)

                results.append({
                    'date': monday,
                    'prediction': prediction,
                    'actual': actual,
                    'contract': recommendation.get('contract'),
                    'side': recommendation.get('side'),
                    'confidence': recommendation.get('confidence', 0),
                    'profit': week_profit,
                    'equity': equity
                })

            except Exception as e:
                logger.debug(f"Error processing week {monday}: {e}")
                continue

        return self._compile_results(results, weekly_returns)

    def _get_mondays(self, start_date: str, end_date: Optional[str]) -> List[datetime]:
        """Generate list of Mondays between start and end dates."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) if end_date else datetime.now() - timedelta(days=7)

        # Adjust start to Monday if not already
        days_until_monday = (7 - start.weekday()) % 7
        if start.weekday() != 0:
            start = start + timedelta(days=days_until_monday)

        mondays = []
        current = start
        while current <= end:
            mondays.append(current)
            current += timedelta(days=7)

        return mondays

    def _predict_week(
        self,
        train_data: pd.DataFrame,
        col: str,
        target_monday: datetime
    ) -> Tuple[float, float]:
        """
        Generate weekly prediction using simple forecasting.

        Uses weighted combination of:
        - Same week last year
        - Recent 4-week average
        - Trend adjustment

        Returns:
            Tuple of (prediction, uncertainty)
        """
        # Calculate weekly totals from training data
        weekly_totals = train_data[col].resample('W-SUN').sum()

        if len(weekly_totals) < 52:
            # Fallback: use recent average
            recent_mean = train_data[col].iloc[-28:].sum() if len(train_data) >= 28 else train_data[col].sum()
            recent_std = train_data[col].iloc[-28:].std() * np.sqrt(7) if len(train_data) >= 28 else recent_mean * 0.1
            return recent_mean, recent_std

        # Find same week last year
        target_week = target_monday.isocalendar()[1]
        last_year_weeks = weekly_totals.iloc[-52:]

        # Get same week last year (approximately)
        same_week_ly = weekly_totals.iloc[-52] if len(weekly_totals) >= 52 else weekly_totals.mean()

        # Recent 4-week average
        recent_avg = weekly_totals.iloc[-4:].mean()

        # Calculate trend (YoY growth rate)
        if len(weekly_totals) >= 104:
            last_year_avg = weekly_totals.iloc[-104:-52].mean()
            this_year_avg = weekly_totals.iloc[-52:].mean()
            yoy_growth = (this_year_avg / last_year_avg) - 1 if last_year_avg > 0 else 0
        else:
            yoy_growth = 0

        # Weighted prediction: 30% same week LY (adjusted for trend), 70% recent trend
        prediction = 0.3 * same_week_ly * (1 + yoy_growth) + 0.7 * recent_avg

        # Uncertainty based on recent standard deviation
        uncertainty = weekly_totals.iloc[-12:].std() if len(weekly_totals) >= 12 else prediction * 0.1

        return prediction, max(uncertainty, prediction * 0.03)  # Minimum 3% uncertainty

    def _simulate_contracts(
        self,
        actual_volume: float,
        predicted_volume: float,
        uncertainty: float
    ) -> List[Dict]:
        """
        Simulate realistic contract thresholds and prices for historical backtest.

        Contract thresholds are based on expected volume.
        Prices are simulated with market-like uncertainty.
        """
        # Round to nearest 500K for threshold base
        base = round(predicted_volume / 500000) * 500000

        # Generate thresholds around expected volume
        thresholds = [
            base - 1000000,
            base - 500000,
            base,
            base + 500000,
            base + 1000000
        ]

        contracts = []
        for threshold in thresholds:
            if threshold <= 0:
                continue

            # Calculate "true" probability based on actual (which we know in backtest)
            # In reality, we'd use prediction uncertainty
            prob_above = 1.0 if actual_volume > threshold else 0.0

            # Add noise to simulate market price uncertainty (markets aren't perfect)
            # Use prediction-based probability for realistic pricing
            pred_prob = 1 - self._normal_cdf(threshold, predicted_volume, uncertainty)
            market_noise = np.random.normal(0, 0.08)  # 8% noise
            market_price = np.clip(pred_prob + market_noise, 0.05, 0.95)

            contracts.append({
                'ticker': f'TSA-W{int(threshold/1000000)}M',
                'threshold': threshold,
                'yes_price': market_price,
                'no_price': 1 - market_price
            })

        return contracts

    def _normal_cdf(self, x: float, mean: float, std: float) -> float:
        """Calculate normal CDF."""
        from scipy.stats import norm
        return norm.cdf(x, loc=mean, scale=std)

    def _calculate_week_profit(
        self,
        recommendation: Dict,
        actual: float,
        bet_size: float
    ) -> float:
        """
        Calculate profit/loss for a week's trade.

        Args:
            recommendation: Contract selection result
            actual: Actual weekly passenger volume
            bet_size: Amount bet

        Returns:
            Profit (positive) or loss (negative)
        """
        if recommendation.get('contract') is None:
            return 0  # No trade

        # Parse threshold from ticker
        ticker = recommendation['contract']
        try:
            # Extract threshold from ticker like 'TSA-W17M' -> 17000000
            threshold_str = ticker.split('W')[1].replace('M', '')
            threshold = float(threshold_str) * 1000000
        except (ValueError, IndexError):
            threshold = 0

        side = recommendation.get('side', 'yes')
        outcome_above = actual > threshold

        # Determine win/loss
        if side == 'yes':
            won = outcome_above
        else:
            won = not outcome_above

        if won:
            # Profit = bet_size * (1 - price) / price for full payout
            # Simplified: profit = bet_size * (1/price - 1) when you win
            # Or just: profit = bet_size for binary outcome
            return bet_size
        else:
            return -bet_size

    def _compile_results(self, results: List[Dict], weekly_returns: List[float]) -> Dict:
        """Compile backtest results into summary statistics."""
        if not results:
            return self._empty_results()

        df = pd.DataFrame(results)

        # Calculate statistics
        total_profit = df['profit'].sum()
        num_weeks = len(df)
        wins = (df['profit'] > 0).sum()
        win_rate = wins / num_weeks if num_weeks > 0 else 0
        avg_profit = df['profit'].mean()

        # Sharpe ratio (annualized, assuming weekly returns)
        if len(weekly_returns) > 1 and np.std(weekly_returns) > 0:
            sharpe_ratio = (np.mean(weekly_returns) / np.std(weekly_returns)) * np.sqrt(52)
        else:
            sharpe_ratio = 0

        # Max drawdown
        equity_curve = df['equity'].values
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

        return {
            'equity_curve': list(zip(df['date'].tolist(), df['equity'].tolist())),
            'weekly_profits': df[['date', 'prediction', 'actual', 'contract', 'side', 'profit']].to_dict('records'),
            'total_profit': total_profit,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_weeks': num_weeks,
            'avg_profit': avg_profit,
            'initial_capital': self.initial_capital,
            'final_equity': df['equity'].iloc[-1] if len(df) > 0 else self.initial_capital
        }

    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            'equity_curve': [],
            'weekly_profits': [],
            'total_profit': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'num_weeks': 0,
            'avg_profit': 0,
            'initial_capital': self.initial_capital,
            'final_equity': self.initial_capital
        }
