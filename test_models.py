"""Tests for machine learning models."""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.models.supervised import SupervisedMatchPredictor, EnsembleModel
from src.models.gnn import GraphMatchPredictor
from src.models.base import BaseModel, model_registry


class TestSupervisedMatchPredictor:
    """Test cases for supervised learning models."""
    
    @pytest.fixture
    def mock_training_data(self):
        """Mock training data."""
        return pd.DataFrame({
            'match_id': ['match_1', 'match_2', 'match_3', 'match_4'],
            'match_date': [datetime.now()] * 4,
            'home_score': [2, 1, 0, 3],
            'away_score': [1, 1, 2, 0],
            'home_team_id': ['team_1', 'team_2', 'team_3', 'team_4'],
            'away_team_id': ['team_2', 'team_3', 'team_4', 'team_1'],
            'league': ['PL'] * 4
        })
    
    @pytest.fixture
    def mock_features_and_targets(self):
        """Mock features and targets."""
        features = pd.DataFrame(np.random.rand(100, 50))
        targets = pd.Series(np.random.randint(0, 3, 100))  # 0=Away, 1=Draw, 2=Home
        return features, targets
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = SupervisedMatchPredictor('xgboost')
        
        assert model.model_type == 'xgboost'
        assert model.name == 'supervised_xgboost'
        assert model.model is None  # Not trained yet
        assert model.feature_pipeline is not None
    
    def test_model_configurations(self):
        """Test different model type configurations."""
        model_types = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting', 'logistic']
        
        for model_type in model_types:
            model = SupervisedMatchPredictor(model_type)
            assert model.model_type == model_type
            assert model_type in model.model_configs
    
    @patch('src.models.supervised.db_manager')
    def test_get_training_data(self, mock_db, mock_training_data):
        """Test training data retrieval."""
        mock_db.get_dataframe.return_value = mock_training_data
        
        model = SupervisedMatchPredictor('xgboost')
        data = model._get_training_data()
        
        assert not data.empty
        assert 'match_id' in data.columns
        assert 'home_score' in data.columns
        mock_db.get_dataframe.assert_called_once()
    
    @patch.object(SupervisedMatchPredictor, '_get_training_data')
    @patch('src.models.supervised.FeaturePipeline')
    def test_prepare_training_data(self, mock_pipeline_class, mock_get_data, 
                                 mock_training_data, mock_features_and_targets):
        """Test training data preparation."""
        # Setup mocks
        mock_get_data.return_value = mock_training_data
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.fit_transform.return_value = mock_features_and_targets
        
        model = SupervisedMatchPredictor('xgboost')
        X, y = model._prepare_training_data(mock_training_data)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        mock_pipeline.fit_transform.assert_called_once()
    
    @patch.object(SupervisedMatchPredictor, '_prepare_training_data')
    @patch.object(SupervisedMatchPredictor, '_get_training_data')
    def test_train_xgboost_model(self, mock_get_data, mock_prepare_data, 
                                mock_training_data, mock_features_and_targets):
        """Test XGBoost model training."""
        # Setup mocks
        mock_get_data.return_value = mock_training_data
        mock_prepare_data.return_value = mock_features_and_targets
        
        model = SupervisedMatchPredictor('xgboost')
        
        # Mock the actual training to avoid heavy computation
        with patch.object(model, '_create_model') as mock_create:
            mock_xgb_model = Mock()
            mock_xgb_model.fit = Mock()
            mock_create.return_value = mock_xgb_model
            
            with patch('src.models.supervised.CalibratedClassifierCV') as mock_calibrated:
                mock_calibrated.return_value = mock_xgb_model
                
                results = model.train()
        
        assert 'model_type' in results
        assert results['model_type'] == 'xgboost'
        assert 'train_metrics' in results
        assert 'validation_metrics' in results
        assert model.model is not None
    
    @patch.object(SupervisedMatchPredictor, 'train')
    def test_predict_without_training_raises_error(self, mock_train):
        """Test that prediction without training raises error."""
        model = SupervisedMatchPredictor('xgboost')
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(['match_1'])
    
    @patch('src.models.supervised.FeaturePipeline')
    def test_predict_with_trained_model(self, mock_pipeline_class):
        """Test prediction with trained model."""
        # Setup mock model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.3, 0.5]])
        mock_model.predict.return_value = np.array([2])  # Home win
        
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_features = pd.DataFrame(np.random.rand(1, 10))
        mock_pipeline.transform.return_value = mock_features
        
        model = SupervisedMatchPredictor('xgboost')
        model.model = mock_model
        model.feature_pipeline = mock_pipeline
        
        predictions = model.predict(['match_1'])
        
        assert len(predictions) == 1
        assert predictions[0]['match_id'] == 'match_1'
        assert predictions[0]['model_name'] == 'supervised_xgboost'
        assert 'home_win_prob' in predictions[0]
        assert 'away_win_prob' in predictions[0]
        assert 'draw_prob' in predictions[0]
        assert predictions[0]['predicted_outcome'] == 'Home Win'
    
    def test_feature_importance_calculation(self):
        """Test feature importance calculation."""
        model = SupervisedMatchPredictor('xgboost')
        
        # Mock model with feature importances
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.5])
        model.model = mock_model
        
        # Mock features
        mock_features = pd.DataFrame(np.random.rand(10, 3), columns=['feat1', 'feat2', 'feat3'])
        
        model._calculate_feature_importance(mock_features)
        
        assert model.feature_importance is not None
        assert len(model.feature_importance) == 3
        assert 'feature' in model.feature_importance.columns
        assert 'importance' in model.feature_importance.columns


class TestEnsembleModel:
    """Test cases for ensemble model."""
    
    def test_ensemble_initialization(self):
        """Test ensemble model initialization."""
        ensemble = EnsembleModel(['xgboost', 'lightgbm'])
        
        assert ensemble.name == 'supervised_ensemble'
        assert len(ensemble.models) == 2
        assert 'xgboost' in ensemble.models
        assert 'lightgbm' in ensemble.models
    
    def test_ensemble_default_models(self):
        """Test ensemble with default models."""
        ensemble = EnsembleModel()
        
        assert len(ensemble.models) == 3  # Default: xgboost, lightgbm, random_forest
        assert 'xgboost' in ensemble.models
        assert 'lightgbm' in ensemble.models
        assert 'random_forest' in ensemble.models
    
    @patch.object(SupervisedMatchPredictor, 'train')
    def test_ensemble_training(self, mock_train):
        """Test ensemble training."""
        # Mock individual model training results
        mock_train.return_value = {
            'validation_metrics': {'accuracy': 0.65},
            'model_type': 'xgboost'
        }
        
        ensemble = EnsembleModel(['xgboost', 'lightgbm'])
        results = ensemble.train()
        
        assert 'ensemble_results' in results
        assert 'model_weights' in results
        assert len(results['model_weights']) == 2
        
        # Check that weights sum to 1
        total_weight = sum(results['model_weights'].values())
        assert abs(total_weight - 1.0) < 1e-6
    
    @patch.object(SupervisedMatchPredictor, 'predict')
    def test_ensemble_prediction(self, mock_predict):
        """Test ensemble prediction."""
        # Mock individual model predictions
        mock_predict.return_value = [{
            'match_id': 'match_1',
            'away_win_prob': 0.3,
            'draw_prob': 0.2,
            'home_win_prob': 0.5,
            'predicted_outcome': 'Home Win',
            'confidence': 0.5
        }]
        
        ensemble = EnsembleModel(['xgboost', 'lightgbm'])
        ensemble.model_weights = {'xgboost': 0.6, 'lightgbm': 0.4}
        
        predictions = ensemble.predict(['match_1'])
        
        assert len(predictions) == 1
        assert predictions[0]['match_id'] == 'match_1'
        assert predictions[0]['model_name'] == 'supervised_ensemble'
        assert 'individual_predictions' in predictions[0]


class TestGraphMatchPredictor:
    """Test cases for Graph Neural Network model."""
    
    @pytest.fixture
    def mock_player_data(self):
        """Mock player data for testing."""
        return [
            {'id': 'p1', 'name': 'Player 1', 'position': 'Forward', 'age': 25, 'market_value': 50000000},
            {'id': 'p2', 'name': 'Player 2', 'position': 'Midfielder', 'age': 27, 'market_value': 30000000},
            {'id': 'p3', 'name': 'Player 3', 'position': 'Defender', 'age': 29, 'market_value': 20000000},
        ]
    
    def test_gnn_initialization(self):
        """Test GNN model initialization."""
        model = GraphMatchPredictor(input_dim=32, hidden_dim=64)
        
        assert model.name == 'gnn_tactical'
        assert model.input_dim == 32
        assert model.hidden_dim == 64
        assert model.model is not None
        assert isinstance(model.model, torch.nn.Module)
    
    def test_player_features_creation(self, mock_player_data):
        """Test player feature vector creation."""
        model = GraphMatchPredictor()
        
        player = mock_player_data[0]
        features = model._create_player_features(player, 'home')
        
        assert len(features) == model.input_dim
        assert all(isinstance(f, float) for f in features)
    
    def test_tactical_edge_creation(self, mock_player_data):
        """Test tactical edge creation between players."""
        model = GraphMatchPredictor()
        
        edges = model._create_tactical_edges(mock_player_data, mock_player_data)
        
        assert isinstance(edges, list)
        assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in edges)
    
    @patch('src.models.gnn.db_manager')
    def test_get_team_lineup(self, mock_db, mock_player_data):
        """Test team lineup retrieval."""
        mock_db.execute_query.return_value = mock_player_data
        
        model = GraphMatchPredictor()
        lineup = model._get_team_lineup('team_1', datetime.now())
        
        assert lineup == mock_player_data
        mock_db.execute_query.assert_called_once()
    
    def test_should_connect_players(self):
        """Test player connection logic."""
        model = GraphMatchPredictor()
        
        # Test position-based connections
        midfielder = {'position': 'Midfielder'}
        forward = {'position': 'Forward'}
        goalkeeper = {'position': 'Goalkeeper'}
        
        assert model._should_connect_players(midfielder, forward)
        assert not model._should_connect_players(goalkeeper, forward)
    
    def test_should_connect_opponents(self):
        """Test opponent connection logic."""
        model = GraphMatchPredictor()
        
        home_forward = {'position': 'Forward'}
        away_forward = {'position': 'Forward'}
        away_defender = {'position': 'Defender'}
        
        assert model._should_connect_opponents(home_forward, away_forward)
        assert not model._should_connect_opponents(home_forward, away_defender)


class TestModelRegistry:
    """Test cases for model registry."""
    
    def test_model_registration(self):
        """Test model registration."""
        registry = model_registry
        
        # Clear registry for clean test
        registry.models.clear()
        registry.active_models.clear()
        
        model = SupervisedMatchPredictor('xgboost')
        registry.register_model(model, is_active=True)
        
        assert model.name in registry.models
        assert model.name in registry.active_models
    
    def test_get_registered_model(self):
        """Test retrieving registered model."""
        registry = model_registry
        registry.models.clear()
        
        model = SupervisedMatchPredictor('lightgbm')
        registry.register_model(model)
        
        retrieved_model = registry.get_model(model.name)
        assert retrieved_model is model
    
    def test_get_active_models(self):
        """Test retrieving active models."""
        registry = model_registry
        registry.models.clear()
        registry.active_models.clear()
        
        model1 = SupervisedMatchPredictor('xgboost')
        model2 = SupervisedMatchPredictor('lightgbm')
        
        registry.register_model(model1, is_active=True)
        registry.register_model(model2, is_active=False)
        
        active_models = registry.get_active_models()
        assert len(active_models) == 1
        assert active_models[0] is model1
    
    @patch.object(SupervisedMatchPredictor, 'predict')
    def test_ensemble_prediction_registry(self, mock_predict):
        """Test ensemble prediction through registry."""
        registry = model_registry
        registry.models.clear()
        registry.active_models.clear()
        
        # Mock predictions
        mock_predict.return_value = [{
            'match_id': 'match_1',
            'away_win_prob': 0.2,
            'draw_prob': 0.3,
            'home_win_prob': 0.5,
            'predicted_outcome': 'Home Win',
            'confidence': 0.6
        }]
        
        # Register models
        model1 = SupervisedMatchPredictor('xgboost')
        model2 = SupervisedMatchPredictor('lightgbm')
        registry.register_model(model1, is_active=True)
        registry.register_model(model2, is_active=True)
        
        # Test ensemble prediction
        ensemble_preds = registry.make_ensemble_prediction(['match_1'])
        
        assert len(ensemble_preds) == 1
        assert ensemble_preds[0]['model_name'] == 'ensemble'
        assert 'individual_predictions' in ensemble_preds[0]


class TestBaseModel:
    """Test cases for base model functionality."""
    
    def test_base_model_abstract(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel("test_model")
    
    def test_model_info(self):
        """Test model info retrieval."""
        model = SupervisedMatchPredictor('xgboost')
        model.trained_at = datetime.now()
        model.set_metadata('test_key', 'test_value')
        
        info = model.get_model_info()
        
        assert info['name'] == 'supervised_xgboost'
        assert info['version'] == '1.0'
        assert 'trained_at' in info
        assert info['metadata']['test_key'] == 'test_value'
    
    def test_input_validation(self):
        """Test input validation."""
        model = SupervisedMatchPredictor('xgboost')
        
        # Valid inputs
        assert model.validate_input(['match_1', 'match_2'])
        
        # Invalid inputs
        assert not model.validate_input([])  # Empty list
        assert not model.validate_input([1, 2, 3])  # Non-string IDs
        assert not model.validate_input(None)  # None input


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for complete model workflows."""
    
    @patch('src.models.supervised.db_manager')
    @patch('src.models.supervised.FeaturePipeline')
    def test_full_training_prediction_workflow(self, mock_pipeline_class, mock_db):
        """Test complete workflow from training to prediction."""
        # Mock training data
        training_data = pd.DataFrame({
            'match_id': ['m1', 'm2', 'm3', 'm4'],
            'match_date': [datetime.now()] * 4,
            'home_score': [2, 1, 0, 3],
            'away_score': [1, 1, 2, 0],
            'home_team_id': ['t1', 't2', 't3', 't4'],
            'away_team_id': ['t2', 't3', 't4', 't1'],
            'league': ['PL'] * 4
        })
        
        # Mock features and targets
        features = pd.DataFrame(np.random.rand(4, 10))
        targets = pd.Series([2, 1, 0, 2])  # Match outcomes
        
        # Setup mocks
        mock_db.get_dataframe.return_value = training_data
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.fit_transform.return_value = (features, targets)
        mock_pipeline.transform.return_value = features.iloc[:1]  # For prediction
        
        # Train model
        model = SupervisedMatchPredictor('xgboost')
        
        with patch.object(model, '_create_model') as mock_create:
            mock_xgb = Mock()
            mock_xgb.fit = Mock()
            mock_create.return_value = mock_xgb
            
            with patch('src.models.supervised.CalibratedClassifierCV') as mock_cal:
                mock_cal_model = Mock()
                mock_cal_model.predict_proba.return_value = np.array([[0.2, 0.3, 0.5]])
                mock_cal_model.predict.return_value = np.array([2])
                mock_cal.return_value = mock_cal_model
                
                # Train
                training_results = model.train(training_data)
                
                # Predict
                predictions = model.predict(['test_match'])
        
        # Verify results
        assert 'validation_metrics' in training_results
        assert len(predictions) == 1
        assert predictions[0]['match_id'] == 'test_match'
        assert predictions[0]['predicted_outcome'] in ['Home Win', 'Draw', 'Away Win']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])