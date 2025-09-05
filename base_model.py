"""Base model class for Bot Quantum Max."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, name: str):
        self.name = name
        self.version = "1.0"
        self.trained_at = None
        self.model_metadata = {}
    
    @abstractmethod
    def train(self, train_data: pd.DataFrame = None, **kwargs) -> Dict:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, match_ids: List[str], **kwargs) -> List[Dict]:
        """Make predictions for given matches."""
        pass
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'name': self.name,
            'version': self.version,
            'trained_at': self.trained_at,
            'metadata': self.model_metadata
        }
    
    def set_metadata(self, key: str, value: Any):
        """Set model metadata."""
        self.model_metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get model metadata."""
        return self.model_metadata.get(key, default)
    
    def validate_input(self, match_ids: List[str]) -> bool:
        """Validate input match IDs."""
        if not match_ids:
            return False
        
        if not all(isinstance(mid, str) for mid in match_ids):
            return False
        
        return True
    
    def save_prediction_to_db(self, prediction: Dict):
        """Save prediction to database."""
        from src.data.database import db_manager
        import uuid
        
        query = """
        INSERT INTO predictions (
            id, match_id, model_name, model_version, prediction_date,
            home_win_prob, draw_prob, away_win_prob, confidence_score
        ) VALUES (
            :id, :match_id, :model_name, :model_version, :prediction_date,
            :home_win_prob, :draw_prob, :away_win_prob, :confidence_score
        )
        ON CONFLICT (match_id, model_name) 
        DO UPDATE SET
            home_win_prob = EXCLUDED.home_win_prob,
            draw_prob = EXCLUDED.draw_prob,
            away_win_prob = EXCLUDED.away_win_prob,
            confidence_score = EXCLUDED.confidence_score,
            prediction_date = EXCLUDED.prediction_date
        """
        
        db_manager.execute_insert(query, {
            'id': str(uuid.uuid4()),
            'match_id': prediction['match_id'],
            'model_name': self.name,
            'model_version': self.version,
            'prediction_date': datetime.now(),
            'home_win_prob': prediction.get('home_win_prob', 0),
            'draw_prob': prediction.get('draw_prob', 0),
            'away_win_prob': prediction.get('away_win_prob', 0),
            'confidence_score': prediction.get('confidence', 0)
        })


class ModelRegistry:
    """Registry for managing multiple models."""
    
    def __init__(self):
        self.models = {}
        self.active_models = []
    
    def register_model(self, model: BaseModel, is_active: bool = True):
        """Register a model."""
        self.models[model.name] = model
        if is_active and model.name not in self.active_models:
            self.active_models.append(model.name)
        
        logger.info(f"Registered model: {model.name}")
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """Get a registered model by name."""
        return self.models.get(name)
    
    def get_active_models(self) -> List[BaseModel]:
        """Get all active models."""
        return [self.models[name] for name in self.active_models if name in self.models]
    
    def make_ensemble_prediction(self, match_ids: List[str]) -> List[Dict]:
        """Make predictions using all active models and combine them."""
        if not self.active_models:
            logger.warning("No active models for ensemble prediction")
            return []
        
        all_predictions = {}
        model_weights = {}
        
        # Get predictions from each active model
        for model_name in self.active_models:
            model = self.models[model_name]
            try:
                predictions = model.predict(match_ids)
                all_predictions[model_name] = predictions
                # Weight based on model confidence or use equal weights
                model_weights[model_name] = 1.0 / len(self.active_models)
            except Exception as e:
                logger.error(f"Error getting predictions from {model_name}: {e}")
                continue
        
        if not all_predictions:
            return []
        
        # Combine predictions
        ensemble_predictions = []
        for i, match_id in enumerate(match_ids):
            combined_probs = {'home_win_prob': 0, 'draw_prob': 0, 'away_win_prob': 0}
            total_weight = 0
            individual_preds = {}
            
            for model_name, predictions in all_predictions.items():
                if i < len(predictions):
                    pred = predictions[i]
                    weight = model_weights[model_name]
                    
                    combined_probs['home_win_prob'] += pred.get('home_win_prob', 0) * weight
                    combined_probs['draw_prob'] += pred.get('draw_prob', 0) * weight
                    combined_probs['away_win_prob'] += pred.get('away_win_prob', 0) * weight
                    total_weight += weight
                    
                    individual_preds[model_name] = pred
            
            # Normalize probabilities
            if total_weight > 0:
                for key in combined_probs:
                    combined_probs[key] /= total_weight
            
            # Determine predicted outcome
            max_prob = max(combined_probs.values())
            if combined_probs['home_win_prob'] == max_prob:
                predicted_outcome = 'Home Win'
            elif combined_probs['away_win_prob'] == max_prob:
                predicted_outcome = 'Away Win'
            else:
                predicted_outcome = 'Draw'
            
            ensemble_prediction = {
                'match_id': match_id,
                'model_name': 'ensemble',
                'predicted_outcome': predicted_outcome,
                'confidence': max_prob,
                'individual_predictions': individual_preds,
                **combined_probs
            }
            
            ensemble_predictions.append(ensemble_prediction)
        
        return ensemble_predictions


# Global model registry instance
model_registry = ModelRegistry()
            '