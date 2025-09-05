"""Portfolio optimization for betting strategies."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import scipy.optimize as opt
from scipy.linalg import LinAlgError
import cvxpy as cp

from config.settings import settings

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Portfolio optimizer for betting strategies using Modern Portfolio Theory."""
    
    def __init__(self, max_kelly_fraction: float = None, risk_aversion: float = 2.0):
        self.max_kelly_fraction = max_kelly_fraction or settings.max_kelly_fraction
        self.risk_aversion = risk_aversion
        
        # Portfolio constraints
        self.min_bet_size = 0.01  # 1% of bankroll minimum
        self.max_bet_size = 0.25  # 25% of bankroll maximum
        self.max_total_exposure = 0.8  # 80% of bankroll maximum total exposure
        
    def optimize_portfolio(self, predictions: List[Dict], bankroll: float = 10000, 
                          odds_data: Optional[List[Dict]] = None) -> Dict:
        """
        Optimize betting portfolio using Modern Portfolio Theory.
        
        Args:
            predictions: List of match predictions with probabilities
            bankroll: Total available bankroll
            odds_data: Optional odds data for each prediction
        
        Returns:
            Dictionary with optimized portfolio allocation
        """
        logger.info(f"Optimizing portfolio for {len(predictions)} predictions")
        
        if not predictions:
            return {'allocations': [], 'total_allocation': 0, 'expected_return': 0}
        
        # Filter profitable bets
        profitable_bets = self._filter_profitable_bets(predictions, odds_data)
        
        if not profitable_bets:
            logger.warning("No profitable bets found")
            return {'allocations': [], 'total_allocation': 0, 'expected_return': 0}
        
        # Calculate expected returns and covariance matrix
        expected_returns, covariance_matrix = self._calculate_risk_return_metrics(profitable_bets)
        
        # Optimize portfolio
        optimal_weights = self._optimize_mean_variance(expected_returns, covariance_matrix)
        
        # Convert weights to actual bet sizes
        allocations = self._weights_to_allocations(
            profitable_bets, optimal_weights, bankroll
        )
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            allocations, expected_returns, covariance_matrix, bankroll
        )
        
        return {
            'allocations': allocations,
            'portfolio_metrics': portfolio_metrics,
            'total_allocation': sum(alloc['stake'] for alloc in allocations),
            'expected_return': portfolio_metrics['expected_return'],
            'expected_volatility': portfolio_metrics['expected_volatility'],
            'sharpe_ratio': portfolio_metrics['sharpe_ratio']
        }
    
    def kelly_criterion_allocation(self, predictions: List[Dict], 
                                 odds_data: List[Dict], bankroll: float) -> List[Dict]:
        """
        Calculate bet sizes using Kelly Criterion.
        
        Args:
            predictions: List of predictions with probabilities
            odds_data: Odds data for each prediction
            bankroll: Total bankroll
        
        Returns:
            List of Kelly-optimized allocations
        """
        allocations = []
        
        for i, (pred, odds) in enumerate(zip(predictions, odds_data)):
            try:
                # Find the best betting market
                best_bet = self._find_best_bet_opportunity(pred, odds)
                
                if best_bet:
                    # Calculate Kelly fraction
                    kelly_fraction = self._calculate_kelly_fraction(
                        best_bet['probability'],
                        best_bet['odds']
                    )
                    
                    # Apply Kelly fraction cap
                    capped_fraction = min(kelly_fraction, self.max_kelly_fraction)
                    
                    if capped_fraction > 0.01:  # Minimum 1% to be worth betting
                        stake = bankroll * capped_fraction
                        
                        allocations.append({
                            'match_id': pred.get('match_id'),
                            'match_name': f"{pred.get('home_team', 'Home')} vs {pred.get('away_team', 'Away')}",
                            'bet_type': best_bet['bet_type'],
                            'selection': best_bet['selection'],
                            'odds': best_bet['odds'],
                            'probability': best_bet['probability'],
                            'stake': stake,
                            'fraction_of_bankroll': capped_fraction,
                            'kelly_fraction': kelly_fraction,
                            'expected_value': stake * (best_bet['odds'] * best_bet['probability'] - 1),
                            'confidence': pred.get('confidence', 0)
                        })
            
            except Exception as e:
                logger.error(f"Error calculating Kelly allocation for prediction {i}: {e}")
                continue
        
        # Sort by expected value
        allocations.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return allocations
    
    def fractional_kelly_allocation(self, predictions: List[Dict], odds_data: List[Dict],
                                  bankroll: float, fraction: float = 0.25) -> List[Dict]:
        """
        Calculate fractional Kelly allocations (more conservative).
        
        Args:
            predictions: List of predictions
            odds_data: Odds data
            bankroll: Total bankroll
            fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly)
        
        Returns:
            List of fractional Kelly allocations
        """
        full_kelly_allocations = self.kelly_criterion_allocation(predictions, odds_data, bankroll)
        
        # Apply fractional scaling
        for allocation in full_kelly_allocations:
            allocation['stake'] *= fraction
            allocation['fraction_of_bankroll'] *= fraction
            allocation['expected_value'] *= fraction
        
        return full_kelly_allocations
    
    def _filter_profitable_bets(self, predictions: List[Dict], 
                               odds_data: Optional[List[Dict]]) -> List[Dict]:
        """Filter predictions to only include profitable betting opportunities."""
        profitable_bets = []
        
        for i, pred in enumerate(predictions):
            try:
                # Use dummy odds if not provided
                if odds_data and i < len(odds_data):
                    odds = odds_data[i]
                else:
                    odds = self._generate_dummy_odds(pred)
                
                # Find best betting opportunity
                best_bet = self._find_best_bet_opportunity(pred, odds)
                
                if best_bet and best_bet['expected_value'] > 0:
                    bet_info = {
                        **pred,
                        'best_bet': best_bet,
                        'odds_data': odds
                    }
                    profitable_bets.append(bet_info)
            
            except Exception as e:
                logger.error(f"Error filtering prediction {i}: {e}")
                continue
        
        logger.info(f"Found {len(profitable_bets)} profitable opportunities out of {len(predictions)}")
        return profitable_bets
    
    def _find_best_bet_opportunity(self, prediction: Dict, odds: Dict) -> Optional[Dict]:
        """Find the best betting opportunity for a prediction."""
        opportunities = []
        
        # Check 1X2 market
        if 'h2h' in odds:
            h2h_odds = odds['h2h']
            
            # Home win
            if 'home' in h2h_odds:
                prob = prediction['home_win_prob']
                bet_odds = h2h_odds['home']
                ev = prob * bet_odds - 1
                
                opportunities.append({
                    'bet_type': '1X2',
                    'selection': 'Home Win',
                    'odds': bet_odds,
                    'probability': prob,
                    'expected_value': ev
                })
            
            # Draw
            if 'draw' in h2h_odds:
                prob = prediction['draw_prob']
                bet_odds = h2h_odds['draw']
                ev = prob * bet_odds - 1
                
                opportunities.append({
                    'bet_type': '1X2',
                    'selection': 'Draw',
                    'odds': bet_odds,
                    'probability': prob,
                    'expected_value': ev
                })
            
            # Away win
            if 'away' in h2h_odds:
                prob = prediction['away_win_prob']
                bet_odds = h2h_odds['away']
                ev = prob * bet_odds - 1
                
                opportunities.append({
                    'bet_type': '1X2',
                    'selection': 'Away Win',
                    'odds': bet_odds,
                    'probability': prob,
                    'expected_value': ev
                })
        
        # Check Over/Under market
        if 'totals' in odds:
            totals_odds = odds['totals']
            
            for market, bet_odds in totals_odds.items():
                if 'over_2_5' in market:
                    prob = prediction.get('over_2_5_prob', 0.5)  # Default assumption
                    ev = prob * bet_odds - 1
                    
                    opportunities.append({
                        'bet_type': 'Over/Under',
                        'selection': 'Over 2.5',
                        'odds': bet_odds,
                        'probability': prob,
                        'expected_value': ev
                    })
        
        # Check BTTS market
        if 'btts' in odds:
            btts_odds = odds['btts']
            
            if 'yes' in btts_odds:
                prob = prediction.get('btts_yes_prob', 0.5)
                bet_odds = btts_odds['yes']
                ev = prob * bet_odds - 1
                
                opportunities.append({
                    'bet_type': 'BTTS',
                    'selection': 'Yes',
                    'odds': bet_odds,
                    'probability': prob,
                    'expected_value': ev
                })
        
        # Return best opportunity
        profitable_opportunities = [opp for opp in opportunities if opp['expected_value'] > 0]
        
        if profitable_opportunities:
            return max(profitable_opportunities, key=lambda x: x['expected_value'])
        
        return None
    
    def _calculate_kelly_fraction(self, probability: float, odds: float) -> float:
        """Calculate Kelly Criterion fraction for a bet."""
        if odds <= 1.0 or probability <= 0:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = probability, q = 1 - p
        b = odds - 1
        p = probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        return max(0, kelly_fraction)
    
    def _generate_dummy_odds(self, prediction: Dict) -> Dict:
        """Generate dummy odds when real odds are not available."""
        # Convert probabilities to fair odds
        home_odds = 1 / prediction['home_win_prob'] if prediction['home_win_prob'] > 0 else 10.0
        draw_odds = 1 / prediction['draw_prob'] if prediction['draw_prob'] > 0 else 10.0
        away_odds = 1 / prediction['away_win_prob'] if prediction['away_win_prob'] > 0 else 10.0
        
        # Add bookmaker margin (reduce odds by ~5%)
        margin = 0.05
        home_odds *= (1 - margin)
        draw_odds *= (1 - margin)
        away_odds *= (1 - margin)
        
        return {
            'h2h': {
                'home': home_odds,
                'draw': draw_odds,
                'away': away_odds
            }
        }
    
    def _calculate_risk_return_metrics(self, profitable_bets: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate expected returns and covariance matrix for bets."""
        n_bets = len(profitable_bets)
        expected_returns = np.zeros(n_bets)
        
        # Calculate expected returns
        for i, bet in enumerate(profitable_bets):
            best_bet = bet['best_bet']
            expected_returns[i] = best_bet['expected_value']
        
        # Calculate covariance matrix
        # For simplicity, assume correlations based on league and timing
        covariance_matrix = np.eye(n_bets)
        
        for i in range(n_bets):
            for j in range(i + 1, n_bets):
                # Calculate correlation between bets
                correlation = self._calculate_bet_correlation(profitable_bets[i], profitable_bets[j])
                covariance_matrix[i, j] = correlation
                covariance_matrix[j, i] = correlation
        
        # Apply variance estimates (simplified)
        variances = np.array([max(0.01, abs(ret) * 0.5) for ret in expected_returns])
        covariance_matrix = np.outer(np.sqrt(variances), np.sqrt(variances)) * covariance_matrix
        
        return expected_returns, covariance_matrix
    
    def _calculate_bet_correlation(self, bet1: Dict, bet2: Dict) -> float:
        """Calculate correlation between two bets."""
        correlation = 0.0
        
        # Same match = high correlation
        if bet1.get('match_id') == bet2.get('match_id'):
            correlation = 0.8
        
        # Same league = moderate correlation
        elif bet1.get('league') == bet2.get('league'):
            correlation = 0.3
        
        # Same date = low correlation
        elif bet1.get('match_date') == bet2.get('match_date'):
            correlation = 0.1
        
        return correlation
    
    def _optimize_mean_variance(self, expected_returns: np.ndarray, 
                              covariance_matrix: np.ndarray) -> np.ndarray:
        """Optimize portfolio using mean-variance optimization."""
        n_assets = len(expected_returns)
        
        try:
            # Use CVXPY for convex optimization
            weights = cp.Variable(n_assets)
            
            # Objective: maximize return - risk_aversion * risk
            portfolio_return = expected_returns.T @ weights
            portfolio_risk = cp.quad_form(weights, covariance_matrix)
            
            objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_risk)
            
            # Constraints
            constraints = [
                cp.sum(weights) <= self.max_total_exposure,  # Total exposure limit
                weights >= 0,  # No short selling
                weights <= self.max_bet_size  # Individual bet size limit
            ]
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if weights.value is not None:
                return weights.value
            else:
                logger.warning("Optimization failed, using equal weights")
                return np.ones(n_assets) / n_assets * 0.1
                
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            # Fallback to equal weights
            return np.ones(n_assets) / n_assets * 0.1
    
    def _weights_to_allocations(self, profitable_bets: List[Dict], 
                               weights: np.ndarray, bankroll: float) -> List[Dict]:
        """Convert portfolio weights to actual bet allocations."""
        allocations = []
        
        for i, (bet, weight) in enumerate(zip(profitable_bets, weights)):
            if weight > 0.001:  # Only include meaningful allocations
                best_bet = bet['best_bet']
                stake = bankroll * weight
                
                allocations.append({
                    'match_id': bet.get('match_id'),
                    'match_name': f"{bet.get('home_team', 'Home')} vs {bet.get('away_team', 'Away')}",
                    'bet_type': best_bet['bet_type'],
                    'selection': best_bet['selection'],
                    'odds': best_bet['odds'],
                    'probability': best_bet['probability'],
                    'stake': stake,
                    'fraction_of_bankroll': weight,
                    'expected_value': stake * best_bet['expected_value'],
                    'confidence': bet.get('confidence', 0),
                    'league': bet.get('league', ''),
                    'match_date': bet.get('match_date', '')
                })
        
        return allocations
    
    def _calculate_portfolio_metrics(self, allocations: List[Dict], 
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray, 
                                   bankroll: float) -> Dict:
        """Calculate portfolio performance metrics."""
        if not allocations:
            return {
                'expected_return': 0,
                'expected_volatility': 0,
                'sharpe_ratio': 0,
                'var_95': 0,
                'max_drawdown_estimate': 0
            }
        
        # Calculate portfolio expected return
        total_expected_value = sum(alloc['expected_value'] for alloc in allocations)
        expected_return = total_expected_value / bankroll
        
        # Calculate portfolio risk (simplified)
        weights = np.array([alloc['stake'] / bankroll for alloc in allocations])
        
        try:
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            expected_volatility = np.sqrt(portfolio_variance)
        except (LinAlgError, ValueError):
            expected_volatility = 0.1  # Default estimate
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = -1.645 * expected_volatility * bankroll  # Assuming normal distribution
        
        # Maximum drawdown estimate
        max_drawdown_estimate = min(-0.05 * bankroll, 2 * var_95)
        
        return {
            'expected_return': expected_return,
            'expected_return_absolute': total_expected_value,
            'expected_volatility': expected_volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'max_drawdown_estimate': max_drawdown_estimate,
            'number_of_bets': len(allocations),
            'total_stake': sum(alloc['stake'] for alloc in allocations),
            'average_odds': np.mean([alloc['odds'] for alloc in allocations])
        }
    
    def risk_parity_allocation(self, profitable_bets: List[Dict], 
                             bankroll: float) -> List[Dict]:
        """
        Allocate capital using risk parity approach.
        Each bet contributes equally to portfolio risk.
        """
        if not profitable_bets:
            return []
        
        n_bets = len(profitable_bets)
        
        # Estimate individual bet risks
        bet_risks = []
        for bet in profitable_bets:
            best_bet = bet['best_bet']
            # Risk proxy: volatility of bet outcome
            prob = best_bet['probability']
            odds = best_bet['odds']
            
            # Variance of bet outcome
            win_outcome = odds - 1  # Profit if win
            lose_outcome = -1       # Loss if lose
            
            expected_outcome = prob * win_outcome + (1 - prob) * lose_outcome
            variance = prob * (win_outcome - expected_outcome)**2 + (1 - prob) * (lose_outcome - expected_outcome)**2
            risk = np.sqrt(variance)
            
            bet_risks.append(risk)
        
        bet_risks = np.array(bet_risks)
        
        # Inverse risk weights (higher risk = lower weight)
        inverse_risks = 1 / (bet_risks + 1e-8)  # Add small epsilon to avoid division by zero
        risk_parity_weights = inverse_risks / np.sum(inverse_risks)
        
        # Scale by maximum total exposure
        risk_parity_weights *= self.max_total_exposure
        
        # Create allocations
        allocations = self._weights_to_allocations(profitable_bets, risk_parity_weights, bankroll)
        
        return allocations


class DynamicPositionSizing:
    """Dynamic position sizing based on recent performance and market conditions."""
    
    def __init__(self, base_optimizer: PortfolioOptimizer):
        self.base_optimizer = base_optimizer
        self.performance_history = []
        
    def update_performance(self, bet_results: List[Dict]):
        """Update performance history with recent bet results."""
        self.performance_history.extend(bet_results)
        
        # Keep only recent history (last 100 bets)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def adaptive_allocation(self, predictions: List[Dict], odds_data: List[Dict],
                          bankroll: float) -> List[Dict]:
        """
        Calculate adaptive position sizing based on recent performance.
        """
        # Calculate performance metrics
        performance_factor = self._calculate_performance_factor()
        
        # Get base allocation
        base_allocation = self.base_optimizer.kelly_criterion_allocation(
            predictions, odds_data, bankroll
        )
        
        # Adjust stakes based on performance
        for allocation in base_allocation:
            allocation['stake'] *= performance_factor
            allocation['fraction_of_bankroll'] *= performance_factor
            allocation['expected_value'] *= performance_factor
        
        return base_allocation
    
    def _calculate_performance_factor(self) -> float:
        """Calculate performance adjustment factor based on recent results."""
        if len(self.performance_history) < 10:
            return 1.0  # Not enough history
        
        recent_results = self.performance_history[-20:]  # Last 20 bets
        
        # Calculate win rate and ROI
        wins = sum(1 for result in recent_results if result.get('profit', 0) > 0)
        win_rate = wins / len(recent_results)
        
        total_stake = sum(result.get('stake', 0) for result in recent_results)
        total_profit = sum(result.get('profit', 0) for result in recent_results)
        roi = total_profit / total_stake if total_stake > 0 else 0
        
        # Adjustment factor based on performance
        # Good performance (>55% win rate, >5% ROI) = increase stakes
        # Poor performance (<45% win rate, <-5% ROI) = decrease stakes
        
        if win_rate > 0.55 and roi > 0.05:
            performance_factor = 1.2  # Increase stakes by 20%
        elif win_rate < 0.45 or roi < -0.05:
            performance_factor = 0.8  # Decrease stakes by 20%
        else:
            performance_factor = 1.0  # Keep current stakes
        
        # Cap the adjustment
        return max(0.5, min(1.5, performance_factor))