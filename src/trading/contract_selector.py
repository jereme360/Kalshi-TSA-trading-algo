"""
Contract selection algorithm for TSA prediction markets.
Selects optimal contracts based on prediction confidence and prices.
"""
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ContractSelector:
    """Select optimal contract based on prediction confidence and prices."""

    def __init__(self, min_ev_threshold: float = 0.02):
        """
        Initialize contract selector.

        Args:
            min_ev_threshold: Minimum expected value threshold for trade (default 2%)
        """
        self.min_ev_threshold = min_ev_threshold

    def select_contract(
        self,
        prediction: float,
        prediction_std: float,
        contracts: List[Dict]
    ) -> Dict:
        """
        Select the optimal contract to trade.

        Args:
            prediction: Predicted weekly passengers
            prediction_std: Prediction uncertainty (standard deviation)
            contracts: List of contracts with keys:
                - ticker: Contract identifier
                - threshold: Passenger threshold for YES outcome
                - yes_price: Current YES price (0-1)
                - no_price: Current NO price (0-1)

        Returns:
            Dict with keys:
                - contract: Selected contract ticker (None if no trade)
                - side: 'yes' or 'no'
                - confidence: Probability of winning
                - expected_value: Expected value per $1 bet
                - reasoning: Explanation string
        """
        if not contracts or prediction is None or prediction_std is None:
            return self._no_trade("No contracts or prediction data available")

        if prediction_std <= 0:
            prediction_std = prediction * 0.05  # Default 5% uncertainty

        # Calculate P(actual > threshold) for each contract
        probabilities = {}
        for c in contracts:
            threshold = c.get('threshold', 0)
            # Probability that actual volume exceeds threshold
            prob = 1 - norm.cdf(threshold, loc=prediction, scale=prediction_std)
            probabilities[c['ticker']] = prob

        # Sort contracts by threshold (ascending)
        sorted_contracts = sorted(contracts, key=lambda x: x.get('threshold', 0))

        if not sorted_contracts:
            return self._no_trade("No valid contracts found")

        highest = sorted_contracts[-1]
        highest_prob = probabilities.get(highest['ticker'], 0)

        # HIGH CONFIDENCE CASE: 90%+ confident above highest threshold -> BUY YES
        if highest_prob >= 0.90:
            yes_price = highest.get('yes_price', 0.5)
            ev = self._calc_ev_yes(highest_prob, yes_price)

            if ev >= self.min_ev_threshold:
                return {
                    'contract': highest['ticker'],
                    'side': 'yes',
                    'confidence': highest_prob,
                    'expected_value': ev,
                    'reasoning': f"High confidence ({highest_prob:.0%}) above highest bracket ({highest.get('threshold', 0):,})"
                }

        # MODERATE CONFIDENCE: Find contract with best expected value
        best = self._find_best_ev_contract(probabilities, contracts)

        if best['expected_value'] >= self.min_ev_threshold:
            return best

        return self._no_trade(f"No positive EV contracts (best EV: {best['expected_value']:.2%})")

    def _calc_ev_yes(self, prob: float, price: float) -> float:
        """
        Calculate expected value for YES bet.

        EV = P(win) * (1 - price) - P(lose) * price
           = prob * (1 - price) - (1 - prob) * price
           = prob - price
        """
        return prob - price

    def _calc_ev_no(self, prob_above: float, price: float) -> float:
        """
        Calculate expected value for NO bet.

        P(NO wins) = 1 - P(actual > threshold) = 1 - prob_above
        EV = (1 - prob_above) * (1 - price) - prob_above * price
           = (1 - prob_above) - price
        """
        prob_below = 1 - prob_above
        return prob_below - price

    def _find_best_ev_contract(
        self,
        probabilities: Dict[str, float],
        contracts: List[Dict]
    ) -> Dict:
        """Find the contract with the best expected value."""
        best_ev = -float('inf')
        best_result = self._no_trade("No contracts evaluated")

        for contract in contracts:
            ticker = contract['ticker']
            prob_above = probabilities.get(ticker, 0.5)
            yes_price = contract.get('yes_price', 0.5)
            no_price = contract.get('no_price', 0.5)

            # Evaluate YES bet
            ev_yes = self._calc_ev_yes(prob_above, yes_price)
            if ev_yes > best_ev:
                best_ev = ev_yes
                best_result = {
                    'contract': ticker,
                    'side': 'yes',
                    'confidence': prob_above,
                    'expected_value': ev_yes,
                    'reasoning': f"Best EV on YES ({ev_yes:.1%}) at threshold {contract.get('threshold', 0):,}"
                }

            # Evaluate NO bet
            ev_no = self._calc_ev_no(prob_above, no_price)
            if ev_no > best_ev:
                best_ev = ev_no
                best_result = {
                    'contract': ticker,
                    'side': 'no',
                    'confidence': 1 - prob_above,
                    'expected_value': ev_no,
                    'reasoning': f"Best EV on NO ({ev_no:.1%}) at threshold {contract.get('threshold', 0):,}"
                }

        return best_result

    def _no_trade(self, reason: str) -> Dict:
        """Return a no-trade recommendation."""
        return {
            'contract': None,
            'side': None,
            'confidence': 0,
            'expected_value': 0,
            'reasoning': reason
        }

    def calculate_position_size(
        self,
        bankroll: float,
        confidence: float,
        price: float,
        max_risk_pct: float = 0.05
    ) -> int:
        """
        Calculate optimal position size using Kelly Criterion (fractional).

        Args:
            bankroll: Available capital
            confidence: Probability of winning
            price: Contract price
            max_risk_pct: Maximum percentage of bankroll to risk

        Returns:
            Number of contracts to buy
        """
        if confidence <= 0 or price <= 0 or price >= 1:
            return 0

        # Kelly fraction: f = (p*b - q) / b where b = odds, p = prob win, q = prob lose
        # For binary contracts: b = (1-price)/price, so f = (p - price) / (1 - price)
        edge = confidence - price

        if edge <= 0:
            return 0

        kelly_fraction = edge / (1 - price)

        # Use fractional Kelly (25%) for safety
        fractional_kelly = kelly_fraction * 0.25

        # Cap at max risk percentage
        risk_fraction = min(fractional_kelly, max_risk_pct)

        # Calculate number of contracts
        max_position_value = bankroll * risk_fraction
        contracts = int(max_position_value / price)

        return max(0, contracts)
