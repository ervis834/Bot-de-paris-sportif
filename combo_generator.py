"""Betting combination generator for Bot Quantum Max."""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import itertools
from datetime import datetime
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class ComboGenerator:
    """Generate optimal betting combinations (accumulators, systems, etc.)."""
    
    def __init__(self):
        self.min_odds = settings.min_odds
        self.max_odds = settings.max_odds
        self.max_correlation = settings.max_correlation
        
        # Combination types and their risk profiles
        self.combo_types = {
            'conservative': {
                'min_selections': 2,
                'max_selections': 4,
                'min_confidence': 0.7,
                'min_individual_odds': 1.4,
                'max_individual_odds': 2.5
            },
            'balanced': {
                'min_selections': 3,
                'max_selections': 6,
                'min_confidence': 0.6,
                'min_individual_odds': 1.3,
                'max_individual_odds': 4.0
            },
            'aggressive': {
                'min_selections': 4,
                'max_selections': 8,
                'min_confidence': 0.55,
                'min_individual_odds': 1.2,
                'max_individual_odds': 6.0
            }
        }
    
    def generate_conservative_combos(self, predictions: List[Dict], 
                                   max_combos: int = 10) -> List[Dict]:
        """Generate conservative betting combinations."""
        return self._generate_combos_by_type(predictions, 'conservative', max_combos)
    
    def generate_balanced_combos(self, predictions: List[Dict], 
                               max_combos: int = 15) -> List[Dict]:
        """Generate balanced betting combinations."""
        return self._generate_combos_by_type(predictions, 'balanced', max_combos)
    
    def generate_aggressive_combos(self, predictions: List[Dict], 
                                 max_combos: int = 20) -> List[Dict]:
        """Generate aggressive betting combinations."""
        return self._generate_combos_by_type(predictions, 'aggressive', max_combos)
    
    def _generate_combos_by_type(self, predictions: List[Dict], combo_type: str, 
                                max_combos: int) -> List[Dict]:
        """Generate combinations based on specified type."""
        config = self.combo_types[combo_type]
        
        # Filter predictions based on criteria
        suitable_predictions = self._filter_predictions_for_combos(predictions, config)
        
        if len(suitable_predictions) < config['min_selections']:
            logger.warning(f"Not enough suitable predictions for {combo_type} combos")
            return []
        
        # Generate all possible combinations
        all_combos = []
        
        for size in range(config['min_selections'], 
                         min(config['max_selections'] + 1, len(suitable_predictions) + 1)):
            
            for combo_predictions in itertools.combinations(suitable_predictions, size):
                combo = self._create_combination(list(combo_predictions), combo_type)
                
                if combo and self._validate_combination(combo, config):
                    all_combos.append(combo)
        
        # Sort by expected value and return top combinations
        all_combos.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return all_combos[:max_combos]
    
    def _filter_predictions_for_combos(self, predictions: List[Dict], 
                                     config: Dict) -> List[Dict]:
        """Filter predictions suitable for combinations."""
        suitable = []
        
        for pred in predictions:
            # Check confidence
            if pred.get('confidence', 0) < config['min_confidence']:
                continue
            
            # Estimate fair odds from probability
            predicted_outcome = pred.get('predicted_outcome', '')
            if predicted_outcome == 'Home Win':
                prob = pred.get('home_win_prob', 0)
            elif predicted_outcome == 'Away Win':
                prob = pred.get('away_win_prob', 0)
            elif predicted_outcome == 'Draw':
                prob = pred.get('draw_prob', 0)
            else:
                continue
            
            if prob <= 0:
                continue
            
            fair_odds = 1 / prob
            
            # Check odds range
            if not (config['min_individual_odds'] <= fair_odds <= config['max_individual_odds']):
                continue
            
            # Add estimated odds to prediction
            pred_copy = pred.copy()
            pred_copy['estimated_odds'] = fair_odds
            suitable.append(pred_copy)
        
        return suitable
    
    def _create_combination(self, predictions: List[Dict], combo_type: str) -> Optional[Dict]:
        """Create a betting combination from predictions."""
        if not predictions:
            return None
        
        # Calculate combination odds and probability
        total_odds = 1.0
        total_probability = 1.0
        
        selections = []
        total_confidence = 0
        
        for pred in predictions:
            odds = pred.get('estimated_odds', 2.0)
            
            # Get probability for the predicted outcome
            predicted_outcome = pred.get('predicted_outcome', '')
            if predicted_outcome == 'Home Win':
                prob = pred.get('home_win_prob', 0)
            elif predicted_outcome == 'Away Win':
                prob = pred.get('away_win_prob', 0)
            elif predicted_outcome == 'Draw':
                prob = pred.get('draw_prob', 0)
            else:
                prob = 0.5  # Fallback
            
            total_odds *= odds
            total_probability *= prob
            total_confidence += pred.get('confidence', 0)
            
            selections.append({
                'match': f"{pred.get('home_team', 'Home')} vs {pred.get('away_team', 'Away')}",
                'selection': predicted_outcome,
                'odds': odds,
                'probability': prob,
                'confidence': pred.get('confidence', 0),
                'match_date': pred.get('match_date', ''),
                'league': pred.get('league', '')
            })
        
        avg_confidence = total_confidence / len(predictions)
        
        # Calculate expected value (assuming 5% bookmaker margin)
        bookmaker_margin = 0.05
        adjusted_odds = total_odds * (1 - bookmaker_margin)
        expected_value = total_probability * adjusted_odds - 1
        
        # Calculate risk metrics
        variance = self._calculate_combo_variance(predictions)
        
        combination = {
            'type': combo_type,
            'selections': selections,
            'num_selections': len(selections),
            'total_odds': total_odds,
            'adjusted_odds': adjusted_odds,
            'probability': total_probability,
            'expected_value': expected_value,
            'avg_confidence': avg_confidence,
            'variance': variance,
            'risk_level': self._calculate_risk_level(total_odds, total_probability, variance),
            'recommended_stake_pct': self._calculate_recommended_stake(
                expected_value, variance, combo_type
            )
        }
        
        return combination
    
    def _validate_combination(self, combo: Dict, config: Dict) -> bool:
        """Validate if combination meets criteria."""
        # Must have positive expected value
        if combo['expected_value'] <= 0:
            return False
        
        # Check correlation between matches (simplified)
        if self._check_correlation(combo['selections']) > self.max_correlation:
            return False
        
        # Check minimum probability
        if combo['probability'] < 0.05:  # Less than 5% chance is too risky
            return False
        
        return True
    
    def _check_correlation(self, selections: List[Dict]) -> float:
        """Check correlation between selections (simplified)."""
        if len(selections) < 2:
            return 0
        
        # Simple correlation based on same league/date
        same_league_pairs = 0
        same_date_pairs = 0
        total_pairs = len(selections) * (len(selections) - 1) / 2
        
        for i in range(len(selections)):
            for j in range(i + 1, len(selections)):
                if selections[i]['league'] == selections[j]['league']:
                    same_league_pairs += 1
                
                if selections[i]['match_date'] == selections[j]['match_date']:
                    same_date_pairs += 1
        
        if total_pairs == 0:
            return 0
        
        # Correlation proxy
        league_correlation = same_league_pairs / total_pairs * 0.3
        date_correlation = same_date_pairs / total_pairs * 0.1
        
        return min(1.0, league_correlation + date_correlation)
    
    def _calculate_combo_variance(self, predictions: List[Dict]) -> float:
        """Calculate variance of combination outcome."""
        # Simplified variance calculation
        total_variance = 0
        
        for pred in predictions:
            # Get probability for predicted outcome
            predicted_outcome = pred.get('predicted_outcome', '')
            if predicted_outcome == 'Home Win':
                prob = pred.get('home_win_prob', 0)
            elif predicted_outcome == 'Away Win':
                prob = pred.get('away_win_prob', 0)
            elif predicted_outcome == 'Draw':
                prob = pred.get('draw_prob', 0)
            else:
                prob = 0.5
            
            # Variance of Bernoulli random variable
            variance = prob * (1 - prob)
            total_variance += variance
        
        return total_variance
    
    def _calculate_risk_level(self, odds: float, probability: float, variance: float) -> str:
        """Calculate risk level of combination."""
        # Risk factors
        odds_risk = min(1.0, odds / 50)  # Higher odds = higher risk
        prob_risk = max(0, (0.5 - probability) / 0.5)  # Lower probability = higher risk
        variance_risk = min(1.0, variance / 2)  # Higher variance = higher risk
        
        combined_risk = (odds_risk + prob_risk + variance_risk) / 3
        
        if combined_risk < 0.3:
            return 'low'
        elif combined_risk < 0.6:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_recommended_stake(self, expected_value: float, variance: float, 
                                   combo_type: str) -> float:
        """Calculate recommended stake percentage of bankroll."""
        if expected_value <= 0:
            return 0
        
        # Kelly Criterion adjusted for combination betting
        # f = (bp - q) / b, but modified for combinations
        
        # Base stake percentages by combo type
        base_stakes = {
            'conservative': 0.05,  # 5% max
            'balanced': 0.03,      # 3% max
            'aggressive': 0.02     # 2% max
        }
        
        base_stake = base_stakes.get(combo_type, 0.02)
        
        # Adjust based on expected value and risk
        ev_multiplier = min(2.0, expected_value * 5)  # Higher EV = higher stake
        risk_divisor = max(1.0, variance * 2)         # Higher risk = lower stake
        
        recommended_stake = base_stake * ev_multiplier / risk_divisor
        
        # Cap at reasonable limits
        return min(base_stake, max(0.005, recommended_stake))
    
    def generate_system_bets(self, predictions: List[Dict], system_type: str = 'yankee') -> List[Dict]:
        """
        Generate system bets (Yankees, Lucky 15s, etc.).
        
        Args:
            predictions: List of predictions
            system_type: Type of system bet ('yankee', 'lucky15', 'heinz', etc.)
        
        Returns:
            List of system bet combinations
        """
        system_configs = {
            'yankee': {  # 4 selections, 11 bets (6 doubles, 4 trebles, 1 four-fold)
                'selections': 4,
                'combinations': [(2, 6), (3, 4), (4, 1)]
            },
            'lucky15': {  # 4 selections, 15 bets (4 singles, 6 doubles, 4 trebles, 1 four-fold)
                'selections': 4,
                'combinations': [(1, 4), (2, 6), (3, 4), (4, 1)]
            },
            'heinz': {   # 6 selections, 57 bets
                'selections': 6,
                'combinations': [(2, 15), (3, 20), (4, 15), (5, 6), (6, 1)]
            }
        }
        
        if system_type not in system_configs:
            logger.error(f"Unknown system type: {system_type}")
            return []
        
        config = system_configs[system_type]
        required_selections = config['selections']
        
        if len(predictions) < required_selections:
            logger.warning(f"Need at least {required_selections} predictions for {system_type}")
            return []
        
        # Filter suitable predictions
        suitable_predictions = self._filter_predictions_for_combos(
            predictions, self.combo_types['balanced']
        )
        
        if len(suitable_predictions) < required_selections:
            return []
        
        system_bets = []
        
        # Generate all possible system combinations
        for combination in itertools.combinations(suitable_predictions, required_selections):
            system_bet = self._create_system_bet(list(combination), system_type, config)
            if system_bet:
                system_bets.append(system_bet)
        
        # Sort by expected value
        system_bets.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return system_bets[:5]  # Return top 5 system bets
    
    def _create_system_bet(self, predictions: List[Dict], system_type: str, 
                          config: Dict) -> Optional[Dict]:
        """Create a system bet from predictions."""
        selections = []
        for pred in predictions:
            odds = pred.get('estimated_odds', 2.0)
            predicted_outcome = pred.get('predicted_outcome', '')
            
            if predicted_outcome == 'Home Win':
                prob = pred.get('home_win_prob', 0)
            elif predicted_outcome == 'Away Win':
                prob = pred.get('away_win_prob', 0)
            elif predicted_outcome == 'Draw':
                prob = pred.get('draw_prob', 0)
            else:
                prob = 0.5
            
            selections.append({
                'match': f"{pred.get('home_team', 'Home')} vs {pred.get('away_team', 'Away')}",
                'selection': predicted_outcome,
                'odds': odds,
                'probability': prob
            })
        
        # Calculate system bet metrics
        total_combinations = sum(combo[1] for combo in config['combinations'])
        expected_return = 0
        
        # Calculate expected return for each combination type
        for combo_size, combo_count in config['combinations']:
            for combo in itertools.combinations(selections, combo_size):
                combo_odds = np.prod([sel['odds'] for sel in combo])
                combo_prob = np.prod([sel['probability'] for sel in combo])
                expected_return += combo_prob * combo_odds
        
        expected_value = (expected_return / total_combinations) - 1
        
        return {
            'type': f'system_{system_type}',
            'system_type': system_type,
            'selections': selections,
            'total_combinations': total_combinations,
            'expected_value': expected_value,
            'recommended_stake_pct': min(0.02, max(0.005, expected_value * 0.1))
        }
    
    def optimize_combination_portfolio(self, all_combos: List[Dict], 
                                     bankroll: float = 1000) -> Dict:
        """
        Optimize portfolio of combinations to maximize expected return while managing risk.
        """
        if not all_combos:
            return {'selected_combos': [], 'total_stake': 0, 'expected_return': 0}
        
        # Sort combinations by risk-adjusted return
        risk_adjusted_combos = []
        
        for combo in all_combos:
            risk_adjustment = 1.0
            
            # Adjust for risk level
            if combo.get('risk_level') == 'high':
                risk_adjustment = 0.7
            elif combo.get('risk_level') == 'medium':
                risk_adjustment = 0.85
            
            # Adjust for correlation
            if len(combo.get('selections', [])) > 5:
                risk_adjustment *= 0.9  # Penalize highly correlated bets
            
            risk_adjusted_ev = combo['expected_value'] * risk_adjustment
            combo['risk_adjusted_ev'] = risk_adjusted_ev
            
            risk_adjusted_combos.append(combo)
        
        # Sort by risk-adjusted expected value
        risk_adjusted_combos.sort(key=lambda x: x['risk_adjusted_ev'], reverse=True)
        
        # Select combinations with portfolio constraints
        selected_combos = []
        total_stake = 0
        max_total_stake = bankroll * 0.3  # Don't risk more than 30% on combinations
        
        for combo in risk_adjusted_combos:
            stake_pct = combo.get('recommended_stake_pct', 0.01)
            stake = bankroll * stake_pct
            
            # Check if we can afford this combination
            if total_stake + stake <= max_total_stake:
                combo['actual_stake'] = stake
                selected_combos.append(combo)
                total_stake += stake
                
                # Don't select too many combinations
                if len(selected_combos) >= 10:
                    break
        
        # Calculate portfolio metrics
        total_expected_value = sum(combo['expected_value'] * combo['actual_stake'] 
                                 for combo in selected_combos)
        
        return {
            'selected_combos': selected_combos,
            'total_stake': total_stake,
            'expected_return': total_expected_value,
            'roi': total_expected_value / total_stake if total_stake > 0 else 0,
            'num_combinations': len(selected_combos)
        }