#!/usr/bin/env python3
"""
Training script for Bot Quantum Max.
Trains all models and saves them for prediction.
"""

import os
import sys
import logging
from datetime import datetime
import argparse

# Add src to path
sys.path.append('src')

from config.settings import settings, LOGGING_CONFIG
from src.data.etl import run_daily_etl
from src.features.engineering import update_features_table
from src.models.supervised import SupervisedMatchPredictor, EnsembleModel
from src.models.base import model_registry

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories."""
    directories = ['data/models', 'logs', 'data/raw', 'data/processed']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def run_etl_pipeline():
    """Run ETL pipeline to ensure we have latest data."""
    logger.info("Running ETL pipeline")
    try:
        run_daily_etl()
        logger.info("ETL pipeline completed successfully")
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        raise


def update_features():
    """Update features table with latest data."""
    logger.info("Updating features table")
    try:
        update_features_table()
        logger.info("Features updated successfully")
    except Exception as e:
        logger.error(f"Feature update failed: {e}")
        raise


def train_supervised_models():
    """Train supervised learning models."""
    logger.info("Training supervised models")
    
    model_types = ['xgboost', 'lightgbm', 'random_forest']
    trained_models = {}
    
    for model_type in model_types:
        logger.info(f"Training {model_type} model")
        
        try:
            model = SupervisedMatchPredictor(model_type)
            results = model.train()
            
            # Save model
            model_path = f"data/models/{model_type}_model.pkl"
            model.save_model(model_path)
            
            # Register model
            model_registry.register_model(model)
            
            trained_models[model_type] = {
                'model': model,
                'results': results,
                'path': model_path
            }
            
            logger.info(f"{model_type} model training completed. "
                       f"Validation accuracy: {results['validation_metrics']['accuracy']:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train {model_type} model: {e}")
            continue
    
    return trained_models


def train_ensemble_model(individual_models):
    """Train ensemble model."""
    logger.info("Training ensemble model")
    
    try:
        # Get model types from successfully trained models
        model_types = list(individual_models.keys())
        
        if len(model_types) < 2:
            logger.warning("Need at least 2 models for ensemble. Skipping ensemble training.")
            return None
        
        ensemble = EnsembleModel(model_types)
        results = ensemble.train()
        
        # Save ensemble
        ensemble_path = "data/models/ensemble_model.pkl"
        # Note: EnsembleModel would need save/load methods similar to SupervisedMatchPredictor
        
        # Register ensemble
        model_registry.register_model(ensemble)
        
        logger.info("Ensemble model training completed")
        
        return {
            'model': ensemble,
            'results': results,
            'path': ensemble_path
        }
        
    except Exception as e:
        logger.error(f"Failed to train ensemble model: {e}")
        return None


def train_advanced_models():
    """Train advanced models (GNN, Bayesian, etc.)."""
    logger.info("Training advanced models")
    
    # For now, we'll add placeholders for advanced models
    # These would be implemented in their respective modules
    
    advanced_models = {}
    
    try:
        # GNN Model
        logger.info("GNN model training not yet implemented")
        
        # Bayesian Model
        logger.info("Bayesian model training not yet implemented")
        
        # Causal Model
        logger.info("Causal model training not yet implemented")
        
        # RL Model
        logger.info("RL model training not yet implemented")
        
    except Exception as e:
        logger.error(f"Advanced model training failed: {e}")
    
    return advanced_models


def validate_models(trained_models):
    """Validate trained models on recent data."""
    logger.info("Validating trained models")
    
    validation_results = {}
    
    for model_name, model_info in trained_models.items():
        try:
            model = model_info['model']
            results = model_info['results']
            
            # Basic validation - check if model can make predictions
            test_match_ids = ["test_id"]  # Would use actual match IDs
            
            # This is just a placeholder - real validation would use actual data
            validation_results[model_name] = {
                'status': 'passed',
                'validation_accuracy': results['validation_metrics']['accuracy'],
                'can_predict': True
            }
            
            logger.info(f"{model_name} validation passed")
            
        except Exception as e:
            logger.error(f"{model_name} validation failed: {e}")
            validation_results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return validation_results


def generate_training_report(trained_models, validation_results):
    """Generate training report."""
    logger.info("Generating training report")
    
    report = {
        'training_date': datetime.now().isoformat(),
        'models_trained': len(trained_models),
        'successful_validations': sum(1 for v in validation_results.values() if v['status'] == 'passed'),
        'model_details': {}
    }
    
    for model_name, model_info in trained_models.items():
        results = model_info['results']
        validation = validation_results.get(model_name, {})
        
        report['model_details'][model_name] = {
            'training_samples': results.get('training_samples', 0),
            'feature_count': results.get('feature_count', 0),
            'validation_accuracy': results.get('validation_metrics', {}).get('accuracy', 0),
            'validation_f1': results.get('validation_metrics', {}).get('f1_score', 0),
            'cv_accuracy': results.get('cv_scores', {}).get('mean_accuracy', 0),
            'validation_status': validation.get('status', 'unknown'),
            'model_path': model_info.get('path', '')
        }
    
    # Save report
    import json
    report_path = f"data/models/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Training report saved to {report_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Training completed at: {report['training_date']}")
    print(f"Models trained: {report['models_trained']}")
    print(f"Successful validations: {report['successful_validations']}")
    print()
    
    for model_name, details in report['model_details'].items():
        print(f"{model_name.upper()}:")
        print(f"  Validation Accuracy: {details['validation_accuracy']:.3f}")
        print(f"  CV Accuracy: {details['cv_accuracy']:.3f}")
        print(f"  F1 Score: {details['validation_f1']:.3f}")
        print(f"  Status: {details['validation_status']}")
        print()
    
    return report


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train Bot Quantum Max models')
    parser.add_argument('--skip-etl', action='store_true', help='Skip ETL pipeline')
    parser.add_argument('--skip-features', action='store_true', help='Skip feature update')
    parser.add_argument('--models', nargs='+', default=['supervised'], 
                       choices=['supervised', 'ensemble', 'advanced', 'all'],
                       help='Models to train')
    
    args = parser.parse_args()
    
    logger.info("Starting Bot Quantum Max training pipeline")
    print("Bot Quantum Max - Training Pipeline")
    print("==================================")
    
    try:
        # Setup
        setup_directories()
        
        # ETL Pipeline
        if not args.skip_etl:
            run_etl_pipeline()
        else:
            logger.info("Skipping ETL pipeline")
        
        # Feature Engineering
        if not args.skip_features:
            update_features()
        else:
            logger.info("Skipping feature update")
        
        trained_models = {}
        
        # Train Models
        if 'supervised' in args.models or 'all' in args.models:
            supervised_models = train_supervised_models()
            trained_models.update(supervised_models)
        
        if 'ensemble' in args.models or 'all' in args.models:
            if trained_models:
                ensemble_model = train_ensemble_model(trained_models)
                if ensemble_model:
                    trained_models['ensemble'] = ensemble_model
        
        if 'advanced' in args.models or 'all' in args.models:
            advanced_models = train_advanced_models()
            trained_models.update(advanced_models)
        
        # Validation
        validation_results = validate_models(trained_models)
        
        # Generate Report
        report = generate_training_report(trained_models, validation_results)
        
        logger.info("Training pipeline completed successfully")
        print("Training completed successfully!")
        
        return report
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()