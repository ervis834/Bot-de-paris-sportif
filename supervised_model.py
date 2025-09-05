"""Supervised learning models for match outcome prediction."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from datetime import datetime
import uuid

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import shap

from src.models.base import BaseModel
from src.data.database import db_manager
from src.features.engineering import FeaturePipeline
from config.settings import settings

logger = logging.getLogger(__name__)


class SupervisedMatchPredictor(BaseModel):
    """Supervised learning models for match outcome prediction."""
    
    def __init__(self, model_type: str = 'xgboost'):
        super().__init__(name=f"supervised_{model_type}")
        self.model_type = model_type
        self.model = None
        self.feature_pipeline = FeaturePipeline()
        self.feature_importance = None
        self.shap_explainer = None
        
        # Model configurations
        self.model_configs = {
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss'
            },
            'lightgbm': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'verbose': -1
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42
            },
            'logistic': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42,
                'multi_class': 'multinomial',
                'solver': 'lbfgs'
            }
        }
    
    def _create_model(self):
        """Create the specified model type."""
        config = self.model_configs.get(self.model_type, {})
        
        if self.model_type == 'xgboost':
            return xgb.XGBClassifier(**config)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(**config)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(**config)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(**config)
        elif self.model_type == 'logistic':
            return LogisticRegression(**config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, train_data: pd.DataFrame = None, validation_split: float = 0.2) -> Dict:
        """Train the supervised model."""
        logger.info(f"Training {self.name} model")
        
        if train_data is None:
            train_data = self._get_training_data()
        
        if train_data.empty:
            raise ValueError("No training data available")
        
        # Prepare features and targets
        X, y = self._prepare_training_data(train_data)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        
        # Create and train model
        self.model = self._create_model()
        
        # Train with early stopping for gradient boosting models
        if self.model_type in ['xgboost', 'lightgbm']:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Calibrate probabilities
        self.model = CalibratedClassifierCV(self.model, method='isotonic', cv=3)
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self._calculate_feature_importance(X_train)
        
        # Initialize SHAP explainer
        self._initialize_shap_explainer(X_train.sample(100))
        
        # Evaluate model
        train_metrics = self._evaluate_model(X_train, y_train, 'train')
        val_metrics = self._evaluate_model(X_val, y_val, 'validation')
        
        # Cross-validation
        cv_scores = self._cross_validate(X, y)
        
        training_results = {
            'model_type': self.model_type,
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'cv_scores': cv_scores,
            'feature_count': X.shape[1],
            'training_samples': X.shape[0]
        }
        
        # Save model performance to database
        self._save_model_performance(training_results)
        
        logger.info(f"Model training completed. Validation accuracy: {val_metrics['accuracy']:.3f}")
        
        return training_results
    
    def _get_training_data(self) -> pd.DataFrame:
        """Get training data from database."""
        query = """
        SELECT 
            m.id as match_id,
            m.match_date,
            m.home_score,
            m.away_score,
            m.home_team_id,
            m.away_team_id,
            m.league
        FROM matches m
        WHERE m.status = 'FINISHED'
        AND m.match_date >= CURRENT_DATE - INTERVAL '2 years'
        AND m.home_score IS NOT NULL
        AND m.away_score IS NOT NULL
        ORDER BY m.match_date DESC
        """
        
        return db_manager.get_dataframe(query)
    
    def _prepare_training_data(self, match_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and targets from match data."""
        match_ids = match_data['match_id'].tolist()
        match_dates = match_data['match_date'].tolist()
        
        # Generate features
        X, y = self.feature_pipeline.fit_transform(match_ids, match_dates)
        
        # Filter out matches with insufficient data
        valid_indices = y >= 0
        X = X[valid_indices]
        y = y[valid_indices]
        
        logger.info(f"Prepared {len(X)} training samples with {X.shape[1]} features")
        
        return X, y
    
    def _calculate_feature_importance(self, X: pd.DataFrame):
        """Calculate and store feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            base_estimator = self.model.base_estimator if hasattr(self.model, 'base_estimator') else self.model
            importances = base_estimator.feature_importances_
        else:
            # For linear models, use absolute coefficients
            base_estimator = self.model.base_estimator if hasattr(self.model, 'base_estimator') else self.model
            if hasattr(base_estimator, 'coef_'):
                importances = np.abs(base_estimator.coef_).mean(axis=0)
            else:
                importances = np.ones(X.shape[1]) / X.shape[1]  # Equal importance fallback
        
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def _initialize_shap_explainer(self, sample_data: pd.DataFrame):
        """Initialize SHAP explainer for model interpretability."""
        try:
            if self.model_type in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']:
                base_estimator = self.model.base_estimator if hasattr(self.model, 'base_estimator') else self.model
                self.shap_explainer = shap.TreeExplainer(base_estimator)
            else:
                self.shap_explainer = shap.Explainer(self.model, sample_data)
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Dict:
        """Evaluate model performance."""
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted'),
            'log_loss': log_loss(y, y_pred_proba),
            'dataset': dataset_name,
            'sample_size': len(y)
        }
        
        # Add per-class metrics
        for class_idx, class_name in enumerate(['Away Win', 'Draw', 'Home Win']):
            y_binary = (y == class_idx).astype(int)
            if len(np.unique(y_binary)) > 1:
                metrics[f'auc_roc_{class_name}'] = roc_auc_score(y_binary, y_pred_proba[:, class_idx])
        
        return metrics
    
    def _cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict:
        """Perform cross-validation."""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Create fresh model for CV (without calibration)
        cv_model = self._create_model()
        
        cv_scores = {
            'accuracy': cross_val_score(cv_model, X, y, cv=cv, scoring='accuracy'),
            'f1_weighted': cross_val_score(cv_model, X, y, cv=cv, scoring='f1_weighted'),
            'neg_log_loss': cross_val_score(cv_model, X, y, cv=cv, scoring='neg_log_loss')
        }
        
        return {
            'mean_accuracy': cv_scores['accuracy'].mean(),
            'std_accuracy': cv_scores['accuracy'].std(),
            'mean_f1': cv_scores['f1_weighted'].mean(),
            'std_f1': cv_scores['f1_weighted'].std(),
            'mean_log_loss': -cv_scores['neg_log_loss'].mean(),
            'std_log_loss': cv_scores['neg_log_loss'].std()
        }
    
    def predict(self, match_ids: List[str], return_probabilities: bool = True) -> List[Dict]:
        """Make predictions for matches."""
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        # Generate features
        X = self.feature_pipeline.transform(match_ids)
        
        predictions = []
        
        for i, match_id in enumerate(match_ids):
            features = X.iloc[i:i+1]
            
            # Get predictions
            pred_proba = self.model.predict_proba(features)[0]
            pred_class = self.model.predict(features)[0]
            
            # Calculate prediction confidence
            confidence = float(np.max(pred_proba))
            
            # Get feature explanation if SHAP is available
            explanation = None
            if self.shap_explainer:
                try:
                    shap_values = self.shap_explainer.shap_values(features)
                    if isinstance(shap_values, list):
                        # For multi-class, take the values for predicted class
                        explanation = {
                            'shap_values': shap_values[pred_class][0].tolist(),
                            'feature_names': features.columns.tolist()
                        }
                    else:
                        explanation = {
                            'shap_values': shap_values[0].tolist(),
                            'feature_names': features.columns.tolist()
                        }
                except Exception as e:
                    logger.warning(f"Could not generate SHAP explanation: {e}")
            
            prediction = {
                'match_id': match_id,
                'model_name': self.name,
                'away_win_prob': float(pred_proba[0]),
                'draw_prob': float(pred_proba[1]),
                'home_win_prob': float(pred_proba[2]),
                'predicted_outcome': ['Away Win', 'Draw', 'Home Win'][pred_class],
                'confidence': confidence,
                'explanation': explanation
            }
            
            predictions.append(prediction)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.model:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'feature_pipeline': self.feature_pipeline,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'name': self.name
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_pipeline = model_data['feature_pipeline']
        self.feature_importance = model_data.get('feature_importance')
        self.model_type = model_data.get('model_type', self.model_type)
        self.name = model_data.get('name', self.name)
        
        # Reinitialize SHAP explainer
        if self.feature_pipeline.is_fitted:
            sample_data = pd.DataFrame(
                np.zeros((10, len(self.feature_pipeline.feature_columns))),
                columns=self.feature_pipeline.feature_columns
            )
            self._initialize_shap_explainer(sample_data)
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importances."""
        if self.feature_importance is None:
            return pd.DataFrame()
        
        return self.feature_importance.head(top_n)
    
    def explain_prediction(self, match_id: str) -> Dict:
        """Get detailed explanation for a prediction."""
        predictions = self.predict([match_id])
        if not predictions:
            return {}
        
        prediction = predictions[0]
        explanation = prediction.get('explanation', {})
        
        if explanation and 'shap_values' in explanation:
            # Create explanation summary
            shap_values = explanation['shap_values']
            feature_names = explanation['feature_names']
            
            # Get top contributing features
            feature_contributions = [
                {'feature': name, 'contribution': value}
                for name, value in zip(feature_names, shap_values)
            ]
            
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
            
            return {
                'match_id': match_id,
                'prediction': prediction,
                'top_features': feature_contributions[:10],
                'model_confidence': prediction['confidence']
            }
        
        return {'match_id': match_id, 'prediction': prediction}
    
    def _save_model_performance(self, results: Dict):
        """Save model performance to database."""
        for dataset_type in ['train', 'validation']:
            metrics_key = f'{dataset_type}_metrics'
            if metrics_key in results:
                metrics = results[metrics_key]
                
                query = """
                INSERT INTO model_performance (
                    id, model_name, evaluation_date, dataset_type,
                    accuracy, precision, recall, f1_score, log_loss,
                    sample_size
                ) VALUES (
                    :id, :model_name, :evaluation_date, :dataset_type,
                    :accuracy, :precision, :recall, :f1_score, :log_loss,
                    :sample_size
                )
                """
                
                db_manager.execute_insert(query, {
                    'id': str(uuid.uuid4()),
                    'model_name': self.name,
                    'evaluation_date': datetime.now().date(),
                    'dataset_type': dataset_type.upper(),
                    'accuracy': metrics.get('accuracy'),
                    'precision': metrics.get('precision'),
                    'recall': metrics.get('recall'),
                    'f1_score': metrics.get('f1_score'),
                    'log_loss': metrics.get('log_loss'),
                    'sample_size': metrics.get('sample_size')
                })


class EnsembleModel(BaseModel):
    """Ensemble of multiple supervised models."""
    
    def __init__(self, model_types: List[str] = None):
        super().__init__(name="supervised_ensemble")
        
        if model_types is None:
            model_types = ['xgboost', 'lightgbm', 'random_forest']
        
        self.models = {}
        self.model_weights = {}
        
        # Create individual models
        for model_type in model_types:
            self.models[model_type] = SupervisedMatchPredictor(model_type)
    
    def train(self, train_data: pd.DataFrame = None) -> Dict:
        """Train all models in the ensemble."""
        logger.info("Training ensemble models")
        
        training_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}")
            results = model.train(train_data)
            training_results[model_name] = results
            
            # Set weight based on validation accuracy
            val_accuracy = results['validation_metrics']['accuracy']
            self.model_weights[model_name] = val_accuracy
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        for model_name in self.model_weights:
            self.model_weights[model_name] /= total_weight
        
        logger.info(f"Ensemble weights: {self.model_weights}")
        
        return {
            'ensemble_results': training_results,
            'model_weights': self.model_weights
        }
    
    def predict(self, match_ids: List[str]) -> List[Dict]:
        """Make ensemble predictions."""
        all_predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            predictions = model.predict(match_ids)
            all_predictions[model_name] = predictions
        
        ensemble_predictions = []
        
        for i, match_id in enumerate(match_ids):
            # Weighted average of probabilities
            away_prob = sum(
                all_predictions[model_name][i]['away_win_prob'] * self.model_weights[model_name]
                for model_name in self.models
            )
            draw_prob = sum(
                all_predictions[model_name][i]['draw_prob'] * self.model_weights[model_name]
                for model_name in self.models
            )
            home_prob = sum(
                all_predictions[model_name][i]['home_win_prob'] * self.model_weights[model_name]
                for model_name in self.models
            )
            
            # Normalize probabilities
            total_prob = away_prob + draw_prob + home_prob
            away_prob /= total_prob
            draw_prob /= total_prob
            home_prob /= total_prob
            
            # Determine prediction
            probs = [away_prob, draw_prob, home_prob]
            predicted_class = np.argmax(probs)
            confidence = max(probs)
            
            ensemble_prediction = {
                'match_id': match_id,
                'model_name': self.name,
                'away_win_prob': away_prob,
                'draw_prob': draw_prob,
                'home_win_prob': home_prob,
                'predicted_outcome': ['Away Win', 'Draw', 'Home Win'][predicted_class],
                'confidence': confidence,
                'individual_predictions': {
                    model_name: all_predictions[model_name][i]
                    for model_name in self.models
                }
            }
            
            ensemble_predictions.append(ensemble_prediction)
        
        return ensemble_predictions