"""Graph Neural Network for tactical analysis and match prediction."""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from typing import Dict, List, Tuple, Optional
import networkx as nx
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from src.models.base import BaseModel
from src.data.database import db_manager
from config.settings import settings

logger = logging.getLogger(__name__)


class TacticalGNN(nn.Module):
    """Graph Neural Network for tactical pattern analysis."""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 3, 
                 num_layers: int = 3, dropout: float = 0.1):
        super(TacticalGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Attention mechanism for important nodes
        self.attention = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # Final prediction layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for home + away
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch):
        """Forward pass through the GNN."""
        # Apply graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Apply attention
        x = self.attention(x, edge_index)
        x = F.relu(x)
        
        # Global pooling to get graph-level representations
        graph_repr = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch)
        ], dim=1)
        
        # Final classification
        output = self.classifier(graph_repr)
        
        return F.softmax(output, dim=1)


class GraphMatchPredictor(BaseModel):
    """GNN-based match outcome predictor using tactical graphs."""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128):
        super().__init__(name="gnn_tactical")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = TacticalGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=3,  # Home win, Draw, Away win
            num_layers=3
        ).to(self.device)
        
        # Training parameters
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = StandardScaler()
        
        # Graph construction parameters
        self.max_players_per_team = 11
        self.position_mapping = {
            'Goalkeeper': 0, 'Defender': 1, 'Midfielder': 2, 'Forward': 3
        }
    
    def train(self, train_data: pd.DataFrame = None, epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train the GNN model."""
        logger.info("Training GNN model")
        
        if train_data is None:
            train_data = self._get_training_data()
        
        # Create graph data
        train_graphs, train_labels = self._create_graph_dataset(train_data)
        
        # Split into train/validation
        split_idx = int(0.8 * len(train_graphs))
        train_loader = DataLoader(train_graphs[:split_idx], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(train_graphs[split_idx:], batch_size=batch_size, shuffle=False)
        val_labels = train_labels[split_idx:]
        
        # Training loop
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(output, batch.y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation
            if epoch % 10 == 0:
                val_accuracy = self._evaluate_model(val_loader, val_labels)
                val_accuracies.append(val_accuracy)
                logger.info(f"Epoch {epoch}: Loss {avg_loss:.4f}, Val Accuracy {val_accuracy:.4f}")
        
        # Final evaluation
        final_val_accuracy = self._evaluate_model(val_loader, val_labels)
        
        training_results = {
            'model_type': 'gnn',
            'final_val_accuracy': final_val_accuracy,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'epochs': epochs,
            'device': str(self.device)
        }
        
        logger.info(f"GNN training completed. Final validation accuracy: {final_val_accuracy:.3f}")
        
        return training_results
    
    def _get_training_data(self) -> pd.DataFrame:
        """Get training data with tactical information."""
        query = """
        SELECT 
            m.id as match_id,
            m.home_team_id,
            m.away_team_id,
            m.match_date,
            m.home_score,
            m.away_score,
            m.league
        FROM matches m
        WHERE m.status = 'FINISHED'
        AND m.match_date >= CURRENT_DATE - INTERVAL '2 years'
        AND m.home_score IS NOT NULL
        AND m.away_score IS NOT NULL
        ORDER BY m.match_date DESC
        LIMIT 1000
        """
        
        return db_manager.get_dataframe(query)
    
    def _create_graph_dataset(self, match_data: pd.DataFrame) -> Tuple[List[Data], torch.Tensor]:
        """Create graph dataset from match data."""
        graphs = []
        labels = []
        
        for _, match in match_data.iterrows():
            try:
                # Create tactical graph for this match
                graph_data = self._create_match_graph(match)
                
                if graph_data is not None:
                    graphs.append(graph_data)
                    
                    # Create label
                    if match['home_score'] > match['away_score']:
                        label = 2  # Home win
                    elif match['away_score'] > match['home_score']:
                        label = 0  # Away win
                    else:
                        label = 1  # Draw
                    
                    labels.append(label)
                    
            except Exception as e:
                logger.error(f"Error creating graph for match {match['match_id']}: {e}")
                continue
        
        return graphs, torch.tensor(labels, dtype=torch.long)
    
    def _create_match_graph(self, match: pd.Series) -> Optional[Data]:
        """Create a tactical graph representation for a match."""
        # Get team lineups and tactical data
        home_lineup = self._get_team_lineup(match['home_team_id'], match['match_date'])
        away_lineup = self._get_team_lineup(match['away_team_id'], match['match_date'])
        
        if not home_lineup or not away_lineup:
            return None
        
        # Create node features
        node_features = []
        edge_indices = []
        
        # Add home team nodes (0-10)
        for i, player in enumerate(home_lineup[:self.max_players_per_team]):
            features = self._create_player_features(player, team='home')
            node_features.append(features)
        
        # Add away team nodes (11-21)
        for i, player in enumerate(away_lineup[:self.max_players_per_team]):
            features = self._create_player_features(player, team='away')
            node_features.append(features)
        
        # Create edges (tactical connections)
        edges = self._create_tactical_edges(home_lineup, away_lineup)
        
        if not edges:
            return None
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    def _get_team_lineup(self, team_id: str, match_date: datetime) -> List[Dict]:
        """Get team lineup for a specific match."""
        # Simplified - in reality would get actual lineup data
        # For now, create a mock lineup based on team data
        query = """
        SELECT p.id, p.name, p.position, p.age, p.market_value
        FROM players p
        WHERE p.team_id = :team_id
        ORDER BY p.market_value DESC NULLS LAST
        LIMIT 11
        """
        
        result = db_manager.execute_query(query, {"team_id": team_id})
        
        return result if result else []
    
    def _create_player_features(self, player: Dict, team: str) -> List[float]:
        """Create feature vector for a player."""
        features = []
        
        # Position encoding (one-hot)
        position = player.get('position', 'Midfielder')
        pos_vector = [0] * len(self.position_mapping)
        if position in self.position_mapping:
            pos_vector[self.position_mapping[position]] = 1
        features.extend(pos_vector)
        
        # Age (normalized)
        age = min(max(player.get('age', 25), 16), 45)  # Clamp age
        features.append((age - 25) / 10)  # Normalize around 25
        
        # Market value (log-normalized)
        market_value = player.get('market_value', 1000000)
        features.append(np.log(max(market_value, 1000)) / 20)  # Log normalize
        
        # Team indicator
        features.append(1.0 if team == 'home' else 0.0)
        
        # Pad to reach input_dim
        while len(features) < self.input_dim:
            features.append(0.0)
        
        return features[:self.input_dim]
    
    def _create_tactical_edges(self, home_lineup: List[Dict], away_lineup: List[Dict]) -> List[Tuple[int, int]]:
        """Create edges representing tactical connections."""
        edges = []
        
        # Within-team connections (tactical formations)
        # Home team edges (0-10)
        for i in range(min(len(home_lineup), self.max_players_per_team)):
            for j in range(i + 1, min(len(home_lineup), self.max_players_per_team)):
                # Connect players based on positional proximity
                if self._should_connect_players(home_lineup[i], home_lineup[j]):
                    edges.append((i, j))
                    edges.append((j, i))  # Undirected
        
        # Away team edges (11-21)
        offset = self.max_players_per_team
        for i in range(min(len(away_lineup), self.max_players_per_team)):
            for j in range(i + 1, min(len(away_lineup), self.max_players_per_team)):
                if self._should_connect_players(away_lineup[i], away_lineup[j]):
                    edges.append((offset + i, offset + j))
                    edges.append((offset + j, offset + i))  # Undirected
        
        # Cross-team connections (marking/duels)
        for i, home_player in enumerate(home_lineup[:self.max_players_per_team]):
            for j, away_player in enumerate(away_lineup[:self.max_players_per_team]):
                if self._should_connect_opponents(home_player, away_player):
                    edges.append((i, offset + j))
                    edges.append((offset + j, i))
        
        return edges
    
    def _should_connect_players(self, player1: Dict, player2: Dict) -> bool:
        """Determine if two teammates should be connected."""
        pos1 = player1.get('position', '')
        pos2 = player2.get('position', '')
        
        # Connect players in similar positions or adjacent lines
        connections = {
            'Goalkeeper': ['Defender'],
            'Defender': ['Goalkeeper', 'Defender', 'Midfielder'],
            'Midfielder': ['Defender', 'Midfielder', 'Forward'],
            'Forward': ['Midfielder', 'Forward']
        }
        
        return pos2 in connections.get(pos1, [])
    
    def _should_connect_opponents(self, home_player: Dict, away_player: Dict) -> bool:
        """Determine if opposing players should be connected (marking/duels)."""
        # Connect players in similar positions (marking)
        return home_player.get('position') == away_player.get('position')
    
    def _evaluate_model(self, data_loader: DataLoader, true_labels: torch.Tensor) -> float:
        """Evaluate model accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            pred_labels = []
            
            for batch in data_loader:
                batch = batch.to(self.device)
                output = self.model(batch.x, batch.edge_index, batch.batch)
                predicted = torch.argmax(output, dim=1)
                pred_labels.extend(predicted.cpu().numpy())
            
            pred_labels = torch.tensor(pred_labels)
            correct = (pred_labels == true_labels).sum().item()
            total = len(true_labels)
        
        return correct / total if total > 0 else 0.0
    
    def predict(self, match_ids: List[str]) -> List[Dict]:
        """Make predictions using the trained GNN."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = []
        
        for match_id in match_ids:
            try:
                # Get match info
                match_info = self._get_match_info(match_id)
                if not match_info:
                    continue
                
                # Create graph
                graph_data = self._create_match_graph(match_info)
                if graph_data is None:
                    continue
                
                # Make prediction
                self.model.eval()
                with torch.no_grad():
                    graph_data = graph_data.to(self.device)
                    output = self.model(
                        graph_data.x.unsqueeze(0), 
                        graph_data.edge_index,
                        torch.zeros(graph_data.x.size(0), dtype=torch.long, device=self.device)
                    )
                    
                    probabilities = output.cpu().numpy()[0]
                    confidence = float(np.max(probabilities))
                    predicted_class = int(np.argmax(probabilities))
                
                prediction = {
                    'match_id': match_id,
                    'model_name': self.name,
                    'away_win_prob': float(probabilities[0]),
                    'draw_prob': float(probabilities[1]),
                    'home_win_prob': float(probabilities[2]),
                    'predicted_outcome': ['Away Win', 'Draw', 'Home Win'][predicted_class],
                    'confidence': confidence,
                    'graph_nodes': graph_data.x.size(0),
                    'graph_edges': graph_data.edge_index.size(1)
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error predicting match {match_id}: {e}")
                continue
        
        return predictions
    
    def _get_match_info(self, match_id: str) -> Optional[Dict]:
        """Get match information for graph construction."""
        query = """
        SELECT 
            m.id as match_id,
            m.home_team_id,
            m.away_team_id,
            m.match_date
        FROM matches m
        WHERE m.id = :match_id
        """
        
        result = db_manager.execute_query(query, {"match_id": match_id})
        return result[0] if result else None
    
    def save_model(self, filepath: str):
        """Save the trained GNN model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'scaler': self.scaler
        }, filepath)
        
        logger.info(f"GNN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained GNN model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        
        # Recreate model with loaded dimensions
        self.model = TacticalGNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=3
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler = checkpoint.get('scaler', StandardScaler())
        
        logger.info(f"GNN model loaded from {filepath}")
    
    def get_graph_explanation(self, match_id: str) -> Dict:
        """Get explanation of graph structure for a match."""
        match_info = self._get_match_info(match_id)
        if not match_info:
            return {}
        
        graph_data = self._create_match_graph(match_info)
        if graph_data is None:
            return {}
        
        return {
            'num_nodes': graph_data.x.size(0),
            'num_edges': graph_data.edge_index.size(1),
            'node_features_dim': graph_data.x.size(1),
            'graph_density': graph_data.edge_index.size(1) / (graph_data.x.size(0) * (graph_data.x.size(0) - 1)),
            'home_team_nodes': list(range(self.max_players_per_team)),
            'away_team_nodes': list(range(self.max_players_per_team, 2 * self.max_players_per_team))
        }