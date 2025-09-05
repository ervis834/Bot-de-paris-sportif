#!/usr/bin/env python3
"""
Prediction script for Bot Quantum Max.
Generates predictions for today's matches and creates betting combinations.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import argparse
import json

# Add src to path
sys.path.append('src')

from config.settings import settings, LOGGING_CONFIG
from src.data.database import db_manager, match_queries
from src.models.supervised import SupervisedMatchPredictor
from src.models.base import model_registry
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.combos import ComboGenerator
from src.sim.monte_carlo import MonteCarloSimulator

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def load_trained_models():
    """Load all trained models."""
    logger.info("Loading trained models")
    
    model_files = {
        'xgboost': 'data/models/xgboost_model.pkl',
        'lightgbm': 'data/models/lightgbm_model.pkl',
        'random_forest': 'data/models/random_forest_model.pkl',
        'ensemble': 'data/models/ensemble_model.pkl'
    }
    
    loaded_models = {}
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            try:
                if model_name == 'ensemble':
                    # Ensemble model would need special loading logic
                    logger.info(f"Ensemble model loading not yet implemented")
                    continue
                
                model = SupervisedMatchPredictor(model_name)
                model.load_model(model_path)
                model_registry.register_model(model, is_active=True)
                loaded_models[model_name] = model
                
                logger.info(f"Loaded {model_name} model from {model_path}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_name} model: {e}")
                continue
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    if not loaded_models:
        raise ValueError("No models could be loaded. Run train.py first.")
    
    logger.info(f"Loaded {len(loaded_models)} models")
    return loaded_models


def get_todays_matches():
    """Get today's scheduled matches."""
    logger.info("Fetching today's matches")
    
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    
    query = """
    SELECT 
        m.id as match_id,
        m.api_id,
        m.match_date,
        ht.name as home_team,
        at.name as away_team,
        m.league,
        m.venue
    FROM matches m
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    WHERE DATE(m.match_date) BETWEEN :today AND :tomorrow
    AND m.status = 'SCHEDULED'
    ORDER BY m.match_date
    """
    
    matches = db_manager.execute_query(query, {
        'today': today,
        'tomorrow': tomorrow
    })
    
    logger.info(f"Found {len(matches)} matches for today")
    
    return matches


def generate_predictions(matches, models):
    """Generate predictions for all matches using all models."""
    logger.info("Generating predictions")
    
    if not matches:
        logger.warning("No matches to predict")
        return []
    
    match_ids = [match['match_id'] for match in matches]
    all_predictions = []
    
    # Get predictions from each model
    for model_name, model in models.items():
        try:
            logger.info(f"Getting predictions from {model_name}")
            predictions = model.predict(match_ids)
            
            for i, prediction in enumerate(predictions):
                match_info = matches[i]
                prediction.update({
                    'home_team': match_info['home_team'],
                    'away_team': match_info['away_team'],
                    'match_date': match_info['match_date'],
                    'league': match_info['league']
                })
            
            all_predictions.extend(predictions)
            
            # Save predictions to database
            for prediction in predictions:
                model.save_prediction_to_db(prediction)
            
        except Exception as e:
            logger.error(f"Failed to get predictions from {model_name}: {e}")
            continue
    
    # Generate ensemble predictions
    try:
        logger.info("Generating ensemble predictions")
        ensemble_predictions = model_registry.make_ensemble_prediction(match_ids)
        
        for i, prediction in enumerate(ensemble_predictions):
            match_info = matches[i]
            prediction.update({
                'home_team': match_info['home_team'],
                'away_team': match_info['away_team'],
                'match_date': match_info['match_date'],
                'league': match_info['league']
            })
        
        all_predictions.extend(ensemble_predictions)
        
    except Exception as e:
        logger.error(f"Failed to generate ensemble predictions: {e}")
    
    logger.info(f"Generated {len(all_predictions)} total predictions")
    
    return all_predictions


def run_monte_carlo_simulation(matches, predictions):
    """Run Monte Carlo simulations for matches."""
    logger.info("Running Monte Carlo simulations")
    
    simulator = MonteCarloSimulator()
    simulation_results = []
    
    # Group predictions by match
    predictions_by_match = {}
    for pred in predictions:
        if pred['model_name'] == 'ensemble':  # Use ensemble predictions for simulation
            match_id = pred['match_id']
            predictions_by_match[match_id] = pred
    
    for match in matches:
        match_id = match['match_id']
        if match_id in predictions_by_match:
            prediction = predictions_by_match[match_id]
            
            try:
                sim_result = simulator.simulate_match(
                    home_win_prob=prediction['home_win_prob'],
                    draw_prob=prediction['draw_prob'],
                    away_win_prob=prediction['away_win_prob'],
                    runs=settings.max_monte_carlo_runs
                )
                
                sim_result.update({
                    'match_id': match_id,
                    'home_team': match['home_team'],
                    'away_team': match['away_team']
                })
                
                simulation_results.append(sim_result)
                
            except Exception as e:
                logger.error(f"Simulation failed for match {match_id}: {e}")
                continue
    
    logger.info(f"Completed simulations for {len(simulation_results)} matches")
    return simulation_results


def generate_betting_combinations(predictions, simulations):
    """Generate optimal betting combinations."""
    logger.info("Generating betting combinations")
    
    try:
        # Portfolio optimizer
        optimizer = PortfolioOptimizer()
        
        # Combo generator
        combo_generator = ComboGenerator()
        
        # Filter high-confidence predictions
        high_confidence_preds = [
            pred for pred in predictions 
            if pred.get('confidence', 0) >= settings.confidence_threshold
            and pred['model_name'] == 'ensemble'
        ]
        
        if not high_confidence_preds:
            logger.warning("No high-confidence predictions found")
            return {}
        
        # Generate different types of combinations
        combinations = {
            'conservative': combo_generator.generate_conservative_combos(high_confidence_preds),
            'balanced': combo_generator.generate_balanced_combos(high_confidence_preds),
            'aggressive': combo_generator.generate_aggressive_combos(high_confidence_preds)
        }
        
        # Optimize portfolio allocation
        portfolio_allocation = optimizer.optimize_portfolio(
            high_confidence_preds, 
            bankroll=10000  # Example bankroll
        )
        
        combinations['portfolio_allocation'] = portfolio_allocation
        
        logger.info("Betting combinations generated successfully")
        
        return combinations
        
    except Exception as e:
        logger.error(f"Failed to generate betting combinations: {e}")
        return {}


def save_results(matches, predictions, simulations, combinations):
    """Save all results to files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'generated_at': datetime.now().isoformat(),
        'matches_count': len(matches),
        'predictions_count': len(predictions),
        'matches': matches,
        'predictions': predictions,
        'simulations': simulations,
        'combinations': combinations
    }
    
    # Save complete results
    results_file = f"data/processed/predictions_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    # Save today's best bets (simplified format)
    best_bets = []
    for pred in predictions:
        if (pred.get('confidence', 0) >= settings.confidence_threshold and 
            pred['model_name'] == 'ensemble'):
            
            best_bets.append({
                'match': f"{pred['home_team']} vs {pred['away_team']}",
                'prediction': pred['predicted_outcome'],
                'confidence': f"{pred['confidence']:.3f}",
                'odds': {
                    'home': f"{pred['home_win_prob']:.3f}",
                    'draw': f"{pred['draw_prob']:.3f}",
                    'away': f"{pred['away_win_prob']:.3f}"
                }
            })
    
    best_bets_file = f"data/processed/best_bets_today.json"
    with open(best_bets_file, 'w') as f:
        json.dump(best_bets, f, indent=2)
    
    logger.info(f"Best bets saved to {best_bets_file}")
    
    return results_file, best_bets_file


def print_summary(matches, predictions, combinations):
    """Print prediction summary."""
    print("\n" + "="*60)
    print("BOT QUANTUM MAX - TODAY'S PREDICTIONS")
    print("="*60)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Matches analyzed: {len(matches)}")
    print()
    
    # High confidence predictions
    high_conf_preds = [
        p for p in predictions 
        if p.get('confidence', 0) >= settings.confidence_threshold 
        and p['model_name'] == 'ensemble'
    ]
    
    if high_conf_preds:
        print(f"HIGH CONFIDENCE PREDICTIONS (≥{settings.confidence_threshold:.2f}):")
        print("-" * 50)
        
        for pred in high_conf_preds:
            print(f"{pred['home_team']} vs {pred['away_team']}")
            print(f"  Prediction: {pred['predicted_outcome']}")
            print(f"  Confidence: {pred['confidence']:.3f}")
            print(f"  Probabilities: H:{pred['home_win_prob']:.2f} D:{pred['draw_prob']:.2f} A:{pred['away_win_prob']:.2f}")
            print()
    else:
        print("No high confidence predictions found.")
        print()
    
    # Betting combinations summary
    if combinations:
        print("RECOMMENDED BETTING COMBINATIONS:")
        print("-" * 35)
        
        for combo_type, combos in combinations.items():
            if combo_type != 'portfolio_allocation' and combos:
                print(f"{combo_type.upper()}: {len(combos)} combinations")
        
        print()
    
    print("="*60)


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(description='Generate predictions for today\'s matches')
    parser.add_argument('--save-only', action='store_true', help='Only save results, don\'t print summary')
    parser.add_argument('--skip-simulation', action='store_true', help='Skip Monte Carlo simulation')
    parser.add_argument('--skip-combinations', action='store_true', help='Skip betting combinations')
    
    args = parser.parse_args()
    
    logger.info("Starting Bot Quantum Max prediction pipeline")
    
    try:
        # Load models
        models = load_trained_models()
        
        # Get today's matches
        matches = get_todays_matches()
        
        if not matches:
            print("No matches scheduled for today.")
            return
        
        # Generate predictions
        predictions = generate_predictions(matches, models)
        
        # Run simulations
        simulations = []
        if not args.skip_simulation:
            simulations = run_monte_carlo_simulation(matches, predictions)
        
        # Generate combinations
        combinations = {}
        if not args.skip_combinations:
            combinations = generate_betting_combinations(predictions, simulations)
        
        # Save results
        results_file, best_bets_file = save_results(matches, predictions, simulations, combinations)
        
        # Print summary
        if not args.save_only:
            print_summary(matches, predictions, combinations)
        
        logger.info("Prediction pipeline completed successfully")
        
        return {
            'results_file': results_file,
            'best_bets_file': best_bets_file,
            'matches_count': len(matches),
            'predictions_count': len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}")
        print(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()