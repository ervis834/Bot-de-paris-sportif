"""Monte Carlo simulation for match outcomes and betting scenarios."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import scipy.stats as stats
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Monte Carlo simulator for football matches and betting scenarios."""
    
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)
        
        # Score distribution parameters based on historical data
        self.score_distributions = {
            'lambda_home': 1.5,  # Average home goals
            'lambda_away': 1.2,  # Average away goals
            'max_goals': 8       # Maximum goals to consider
        }
        
        # Correlation between home and away goals (typically negative)
        self.goal_correlation = -0.15
    
    def simulate_match(self, home_win_prob: float, draw_prob: float, away_win_prob: float,
                      runs: int = 10000, return_detailed: bool = True) -> Dict:
        """
        Simulate a single match multiple times.
        
        Args:
            home_win_prob: Probability of home team winning
            draw_prob: Probability of draw
            away_win_prob: Probability of away team winning
            runs: Number of simulation runs
            return_detailed: Whether to return detailed score distributions
        
        Returns:
            Dictionary with simulation results
        """
        # Validate probabilities
        total_prob = home_win_prob + draw_prob + away_win_prob
        if abs(total_prob - 1.0) > 0.01:
            # Normalize if needed
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob
        
        # Convert outcome probabilities to goal expectation
        lambda_home, lambda_away = self._probabilities_to_poisson_params(
            home_win_prob, draw_prob, away_win_prob
        )
        
        # Run simulations
        results = []
        score_counts = defaultdict(int)
        outcome_counts = {'home_win': 0, 'draw': 0, 'away_win': 0}
        
        for _ in range(runs):
            home_goals, away_goals = self._simulate_single_match(lambda_home, lambda_away)
            
            # Record result
            results.append((home_goals, away_goals))
            score_counts[(home_goals, away_goals)] += 1
            
            # Count outcome
            if home_goals > away_goals:
                outcome_counts['home_win'] += 1
            elif home_goals < away_goals:
                outcome_counts['away_win'] += 1
            else:
                outcome_counts['draw'] += 1
        
        # Calculate statistics
        home_goals_array = np.array([r[0] for r in results])
        away_goals_array = np.array([r[1] for r in results])
        total_goals_array = home_goals_array + away_goals_array
        
        simulation_result = {
            'runs': runs,
            'outcome_probabilities': {
                'home_win': outcome_counts['home_win'] / runs,
                'draw': outcome_counts['draw'] / runs,
                'away_win': outcome_counts['away_win'] / runs
            },
            'goal_statistics': {
                'avg_home_goals': float(np.mean(home_goals_array)),
                'avg_away_goals': float(np.mean(away_goals_array)),
                'avg_total_goals': float(np.mean(total_goals_array)),
                'std_home_goals': float(np.std(home_goals_array)),
                'std_away_goals': float(np.std(away_goals_array)),
                'std_total_goals': float(np.std(total_goals_array))
            },
            'over_under_probabilities': self._calculate_over_under_probs(total_goals_array),
            'btts_probability': np.mean((home_goals_array > 0) & (away_goals_array > 0)),
            'clean_sheet_probabilities': {
                'home': np.mean(away_goals_array == 0),
                'away': np.mean(home_goals_array == 0)
            }
        }
        
        if return_detailed:
            # Most likely scores
            most_common_scores = Counter(results).most_common(10)
            simulation_result['most_likely_scores'] = [
                {'score': f"{score[0]}-{score[1]}", 'probability': count/runs}
                for score, count in most_common_scores
            ]
            
            # Score distribution
            simulation_result['score_distribution'] = {
                f"{score[0]}-{score[1]}": count/runs
                for score, count in score_counts.items()
            }
        
        return simulation_result
    
    def simulate_multiple_matches(self, match_predictions: List[Dict], 
                                 runs: int = 10000) -> List[Dict]:
        """
        Simulate multiple matches.
        
        Args:
            match_predictions: List of match predictions with probabilities
            runs: Number of simulation runs per match
        
        Returns:
            List of simulation results for each match
        """
        results = []
        
        for i, prediction in enumerate(match_predictions):
            logger.info(f"Simulating match {i+1}/{len(match_predictions)}")
            
            simulation = self.simulate_match(
                home_win_prob=prediction['home_win_prob'],
                draw_prob=prediction['draw_prob'],
                away_win_prob=prediction['away_win_prob'],
                runs=runs
            )
            
            simulation['match_info'] = prediction.get('match_info', {})
            results.append(simulation)
        
        return results
    
    def simulate_betting_scenario(self, bets: List[Dict], runs: int = 10000) -> Dict:
        """
        Simulate betting scenario with multiple bets.
        
        Args:
            bets: List of bet dictionaries with odds and predictions
            runs: Number of simulation runs
        
        Returns:
            Dictionary with betting simulation results
        """
        total_returns = []
        bet_outcomes = {i: [] for i in range(len(bets))}
        
        for run in range(runs):
            total_return = 0
            
            for i, bet in enumerate(bets):
                # Simulate match outcome
                home_win_prob = bet['home_win_prob']
                draw_prob = bet['draw_prob'] 
                away_win_prob = bet['away_win_prob']
                
                # Determine outcome
                rand = self.rng.random()
                if rand < away_win_prob:
                    outcome = 'away_win'
                elif rand < away_win_prob + draw_prob:
                    outcome = 'draw'
                else:
                    outcome = 'home_win'
                
                # Check if bet wins
                bet_selection = bet['selection']  # 'home_win', 'draw', 'away_win', etc.
                bet_odds = bet['odds']
                stake = bet['stake']
                
                if bet_selection == outcome:
                    # Bet wins
                    bet_return = stake * bet_odds
                    bet_profit = bet_return - stake
                    total_return += bet_profit
                    bet_outcomes[i].append(bet_profit)
                else:
                    # Bet loses
                    total_return -= stake
                    bet_outcomes[i].append(-stake)
            
            total_returns.append(total_return)
        
        # Calculate statistics
        total_returns = np.array(total_returns)
        
        return {
            'runs': runs,
            'expected_return': float(np.mean(total_returns)),
            'std_return': float(np.std(total_returns)),
            'max_return': float(np.max(total_returns)),
            'min_return': float(np.min(total_returns)),
            'probability_profit': float(np.mean(total_returns > 0)),
            'var_95': float(np.percentile(total_returns, 5)),  # Value at Risk
            'var_99': float(np.percentile(total_returns, 1)),
            'sharpe_ratio': float(np.mean(total_returns) / np.std(total_returns)) if np.std(total_returns) > 0 else 0,
            'individual_bet_performance': {
                i: {
                    'expected_return': float(np.mean(outcomes)),
                    'win_rate': float(np.mean(np.array(outcomes) > 0)),
                    'std_return': float(np.std(outcomes))
                }
                for i, outcomes in bet_outcomes.items()
            },
            'return_distribution': {
                'percentile_10': float(np.percentile(total_returns, 10)),
                'percentile_25': float(np.percentile(total_returns, 25)),
                'percentile_50': float(np.percentile(total_returns, 50)),
                'percentile_75': float(np.percentile(total_returns, 75)),
                'percentile_90': float(np.percentile(total_returns, 90))
            }
        }
    
    def simulate_season(self, matches: List[Dict], runs: int = 1000) -> Dict:
        """
        Simulate entire season or tournament.
        
        Args:
            matches: List of all matches with predictions
            runs: Number of season simulations
        
        Returns:
            Season simulation results
        """
        team_points = defaultdict(list)
        final_standings = []
        
        for run in range(runs):
            # Initialize points for this simulation
            points = defaultdict(int)
            
            # Simulate each match
            for match in matches:
                home_team = match['home_team']
                away_team = match['away_team']
                
                # Simulate match result
                simulation = self.simulate_match(
                    match['home_win_prob'],
                    match['draw_prob'], 
                    match['away_win_prob'],
                    runs=1,
                    return_detailed=False
                )
                
                # Determine outcome and award points
                outcome_probs = simulation['outcome_probabilities']
                rand = self.rng.random()
                
                if rand < outcome_probs['away_win']:
                    # Away win
                    points[away_team] += 3
                elif rand < outcome_probs['away_win'] + outcome_probs['draw']:
                    # Draw
                    points[home_team] += 1
                    points[away_team] += 1
                else:
                    # Home win
                    points[home_team] += 3
            
            # Store points for each team
            for team, pts in points.items():
                team_points[team].append(pts)
            
            # Create final table for this simulation
            standings = sorted(points.items(), key=lambda x: x[1], reverse=True)
            final_standings.append(standings)
        
        # Calculate statistics
        season_stats = {}
        for team in team_points:
            points_array = np.array(team_points[team])
            season_stats[team] = {
                'avg_points': float(np.mean(points_array)),
                'std_points': float(np.std(points_array)),
                'min_points': float(np.min(points_array)),
                'max_points': float(np.max(points_array)),
                'top_4_probability': 0,  # Will calculate below
                'relegation_probability': 0  # Will calculate below
            }
        
        # Calculate position probabilities
        position_counts = defaultdict(lambda: defaultdict(int))
        
        for standings in final_standings:
            for position, (team, points) in enumerate(standings, 1):
                position_counts[team][position] += 1
        
        for team in season_stats:
            total_simulations = runs
            top_4_count = sum(position_counts[team][pos] for pos in range(1, 5))
            bottom_3_count = sum(position_counts[team][pos] for pos in range(len(season_stats)-2, len(season_stats)+1))
            
            season_stats[team]['top_4_probability'] = top_4_count / total_simulations
            season_stats[team]['relegation_probability'] = bottom_3_count / total_simulations
        
        return {
            'runs': runs,
            'team_statistics': season_stats,
            'position_probabilities': dict(position_counts)
        }
    
    def _simulate_single_match(self, lambda_home: float, lambda_away: float) -> Tuple[int, int]:
        """Simulate a single match using Poisson distribution with correlation."""
        
        # Generate correlated Poisson variables
        # Method: Use bivariate normal and transform to Poisson
        
        # Generate correlated normal variables
        mean = [0, 0]
        cov = [[1, self.goal_correlation], [self.goal_correlation, 1]]
        normal_vars = self.rng.multivariate_normal(mean, cov)
        
        # Transform to uniform [0,1]
        uniform_vars = stats.norm.cdf(normal_vars)
        
        # Transform to Poisson
        home_goals = stats.poisson.ppf(uniform_vars[0], lambda_home)
        away_goals = stats.poisson.ppf(uniform_vars[1], lambda_away)
        
        # Ensure non-negative integers
        home_goals = max(0, int(home_goals))
        away_goals = max(0, int(away_goals))
        
        # Cap at maximum goals
        home_goals = min(home_goals, self.score_distributions['max_goals'])
        away_goals = min(away_goals, self.score_distributions['max_goals'])
        
        return home_goals, away_goals
    
    def _probabilities_to_poisson_params(self, home_win_prob: float, draw_prob: float, 
                                       away_win_prob: float) -> Tuple[float, float]:
        """
        Convert match outcome probabilities to Poisson parameters.
        Uses optimization to find lambda values that best match the given probabilities.
        """
        from scipy.optimize import minimize
        
        def objective(params):
            lambda_home, lambda_away = params
            
            # Calculate theoretical probabilities
            theory_probs = self._calculate_outcome_probabilities(lambda_home, lambda_away)
            
            # Mean squared error
            error = ((theory_probs['home_win'] - home_win_prob) ** 2 +
                    (theory_probs['draw'] - draw_prob) ** 2 +
                    (theory_probs['away_win'] - away_win_prob) ** 2)
            
            return error
        
        # Initial guess
        initial_params = [self.score_distributions['lambda_home'], 
                         self.score_distributions['lambda_away']]
        
        # Bounds (lambda must be positive)
        bounds = [(0.1, 5.0), (0.1, 5.0)]
        
        # Optimize
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            return result.x[0], result.x[1]
        else:
            # Fallback to default parameters
            logger.warning("Failed to optimize Poisson parameters, using defaults")
            return initial_params[0], initial_params[1]
    
    def _calculate_outcome_probabilities(self, lambda_home: float, lambda_away: float) -> Dict:
        """Calculate match outcome probabilities from Poisson parameters."""
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        max_goals = self.score_distributions['max_goals']
        
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = stats.poisson.pmf(h, lambda_home) * stats.poisson.pmf(a, lambda_away)
                
                if h > a:
                    home_win_prob += prob
                elif h == a:
                    draw_prob += prob
                else:
                    away_win_prob += prob
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob
        }
    
    def _calculate_over_under_probs(self, total_goals_array: np.ndarray) -> Dict:
        """Calculate over/under probabilities for different thresholds."""
        thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        
        over_under_probs = {}
        for threshold in thresholds:
            over_prob = float(np.mean(total_goals_array > threshold))
            under_prob = 1 - over_prob
            
            over_under_probs[f'over_{threshold}'] = over_prob
            over_under_probs[f'under_{threshold}'] = under_prob
        
        return over_under_probs


class BayesianSimulator:
    """Bayesian approach to match simulation with uncertainty quantification."""
    
    def __init__(self, prior_alpha: float = 2.0, prior_beta: float = 2.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.rng = np.random.RandomState(42)
    
    def simulate_with_uncertainty(self, predictions: Dict, uncertainty: float = 0.1, 
                                runs: int = 10000) -> Dict:
        """
        Simulate match with prediction uncertainty.
        
        Args:
            predictions: Dictionary with match predictions
            uncertainty: Uncertainty level (0-1)
            runs: Number of simulation runs
        
        Returns:
            Simulation results with uncertainty bands
        """
        # Add noise to predictions based on uncertainty
        results = []
        
        for run in range(runs):
            # Sample from Beta distribution around predictions
            noise_scale = uncertainty
            
            home_prob = predictions['home_win_prob']
            draw_prob = predictions['draw_prob']
            away_prob = predictions['away_win_prob']
            
            # Add Dirichlet noise to maintain probability constraints
            alpha = np.array([home_prob, draw_prob, away_prob]) / noise_scale
            noisy_probs = self.rng.dirichlet(alpha)
            
            # Simulate with noisy probabilities
            rand = self.rng.random()
            if rand < noisy_probs[2]:  # away
                outcome = 'away_win'
            elif rand < noisy_probs[2] + noisy_probs[1]:  # draw
                outcome = 'draw'
            else:  # home
                outcome = 'home_win'
            
            results.append(outcome)
        
        # Calculate statistics
        outcome_counts = Counter(results)
        total_runs = len(results)
        
        return {
            'mean_probabilities': {
                'home_win': outcome_counts['home_win'] / total_runs,
                'draw': outcome_counts['draw'] / total_runs,
                'away_win': outcome_counts['away_win'] / total_runs
            },
            'uncertainty_bands': {
                'home_win_ci': self._calculate_confidence_interval(outcome_counts['home_win'], total_runs),
                'draw_ci': self._calculate_confidence_interval(outcome_counts['draw'], total_runs),
                'away_win_ci': self._calculate_confidence_interval(outcome_counts['away_win'], total_runs)
            },
            'runs': runs,
            'uncertainty_level': uncertainty
        }
    
    def _calculate_confidence_interval(self, successes: int, trials: int, 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for binomial proportion."""
        p = successes / trials
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        margin_of_error = z * np.sqrt(p * (1 - p) / trials)
        
        return (max(0, p - margin_of_error), min(1, p + margin_of_error))