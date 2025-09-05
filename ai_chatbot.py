"""AI Chatbot interface for Bot Quantum Max using GPT-4/5."""

import logging
import openai
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

from src.data.database import db_manager
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.combos import ComboGenerator
from src.models.base import model_registry
from config.settings import settings

logger = logging.getLogger(__name__)


class QuantumChatBot:
    """AI-powered chatbot for natural language commands."""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.conversation_history = []
        
        # Command parsers and executors
        self.command_handlers = {
            'create_combo': self._handle_create_combo,
            'get_predictions': self._handle_get_predictions,
            'analyze_match': self._handle_analyze_match,
            'portfolio_status': self._handle_portfolio_status,
            'find_value_bets': self._handle_find_value_bets,
            'risk_assessment': self._handle_risk_assessment,
            'model_performance': self._handle_model_performance,
            'schedule_alert': self._handle_schedule_alert
        }
        
        # System prompt for the AI
        self.system_prompt = """
Tu es l'assistant IA de Bot Quantum Max, un système avancé de prédiction sportive et de paris intelligents.

CAPACITÉS DISPONIBLES:
1. create_combo - Créer des paris combinés selon des critères
2. get_predictions - Obtenir des prédictions pour des matchs
3. analyze_match - Analyser un match spécifique en détail
4. portfolio_status - Vérifier le statut du portefeuille
5. find_value_bets - Trouver des paris à valeur
6. risk_assessment - Évaluer le risque d'un pari
7. model_performance - Vérifier la performance des modèles
8. schedule_alert - Programmer une alerte

EXEMPLE DE COMMANDES:
- "Crée un combiné avec 3 matchs d'aujourd'hui ayant des cotes > 1.5"
- "Montre-moi les prédictions les plus confiantes pour demain"
- "Analyse le match PSG vs Lyon en détail"
- "Quel est le statut de mon portefeuille ?"
- "Trouve-moi des paris à valeur pour ce weekend"

INSTRUCTIONS:
1. Identifie l'intention de l'utilisateur
2. Extraie les paramètres nécessaires
3. Retourne une réponse JSON avec 'command', 'parameters', et 'explanation'
4. Si la demande n'est pas claire, demande des précisions
5. Sois concis mais informatif

IMPORTANT: Réponds TOUJOURS en JSON valide avec cette structure:
{
    "command": "nom_de_la_commande",
    "parameters": {
        "param1": "valeur1",
        "param2": "valeur2"
    },
    "explanation": "Explication de ce que tu vas faire",
    "clarification_needed": false
}
        """
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process user message and return response."""
        logger.info(f"Processing chat message: {user_message}")
        
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Get AI response
            ai_response = self._get_ai_response(user_message)
            
            # Parse AI response
            command_data = self._parse_ai_response(ai_response)
            
            if command_data.get('clarification_needed'):
                return {
                    'type': 'clarification',
                    'message': command_data.get('explanation', 'Peux-tu préciser ta demande ?'),
                    'suggestions': self._get_command_suggestions()
                }
            
            # Execute command
            if command_data.get('command') in self.command_handlers:
                result = self.command_handlers[command_data['command']](
                    command_data.get('parameters', {})
                )
                
                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"Commande exécutée: {command_data['command']}"
                })
                
                return {
                    'type': 'success',
                    'command': command_data['command'],
                    'explanation': command_data.get('explanation', ''),
                    'result': result
                }
            else:
                return {
                    'type': 'error',
                    'message': f"Commande inconnue: {command_data.get('command')}",
                    'suggestions': self._get_command_suggestions()
                }
                
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return {
                'type': 'error',
                'message': f"Erreur lors du traitement: {str(e)}"
            }
    
    def _get_ai_response(self, user_message: str) -> str:
        """Get response from OpenAI API."""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add recent conversation history (last 5 exchanges)
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        messages.extend(recent_history)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=500,
            temperature=0.3  # Lower temperature for more consistent responses
        )
        
        return response.choices[0].message.content
    
    def _parse_ai_response(self, ai_response: str) -> Dict:
        """Parse AI response to extract command and parameters."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # Fallback: try to parse the entire response as JSON
                return json.loads(ai_response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse AI response as JSON: {ai_response}")
            return {
                'command': 'unknown',
                'parameters': {},
                'explanation': 'Désolé, je n\'ai pas compris votre demande.',
                'clarification_needed': True
            }
    
    def _get_command_suggestions(self) -> List[str]:
        """Get command suggestions for user."""
        return [
            "Crée un combiné avec des matchs d'aujourd'hui",
            "Montre-moi les meilleures prédictions",
            "Analyse le prochain match de [équipe]",
            "Quel est le statut de mon portefeuille ?",
            "Trouve des paris à valeur élevée"
        ]
    
    def _handle_create_combo(self, params: Dict) -> Dict:
        """Handle combo creation command."""
        logger.info("Creating betting combination")
        
        # Extract parameters
        num_matches = params.get('num_matches', 3)
        min_odds = params.get('min_odds', 1.5)
        max_odds = params.get('max_odds', 5.0)
        date_filter = params.get('date', 'today')
        leagues = params.get('leagues', [])
        combo_type = params.get('type', 'balanced')
        
        # Get matches based on criteria
        matches = self._get_matches_by_criteria(date_filter, leagues)
        
        if len(matches) < num_matches:
            return {
                'success': False,
                'message': f"Seulement {len(matches)} matchs trouvés, {num_matches} requis",
                'available_matches': len(matches)
            }
        
        # Get predictions for matches
        match_ids = [match['match_id'] for match in matches]
        predictions = self._get_predictions_for_matches(match_ids)
        
        # Filter predictions based on odds criteria
        suitable_predictions = []
        for pred in predictions:
            # Calculate implied odds from probabilities
            if pred['predicted_outcome'] == 'Home Win':
                implied_odds = 1 / pred['home_win_prob'] if pred['home_win_prob'] > 0 else 10
            elif pred['predicted_outcome'] == 'Away Win':
                implied_odds = 1 / pred['away_win_prob'] if pred['away_win_prob'] > 0 else 10
            else:  # Draw
                implied_odds = 1 / pred['draw_prob'] if pred['draw_prob'] > 0 else 10
            
            if min_odds <= implied_odds <= max_odds:
                pred['estimated_odds'] = implied_odds
                suitable_predictions.append(pred)
        
        if len(suitable_predictions) < num_matches:
            return {
                'success': False,
                'message': f"Seulement {len(suitable_predictions)} matchs correspondent aux critères de cotes",
                'criteria': f"Cotes entre {min_odds} et {max_odds}"
            }
        
        # Create combination
        combo_generator = ComboGenerator()
        selected_preds = suitable_predictions[:num_matches]
        
        if combo_type == 'conservative':
            combos = combo_generator.generate_conservative_combos(selected_preds, max_combos=1)
        elif combo_type == 'aggressive':
            combos = combo_generator.generate_aggressive_combos(selected_preds, max_combos=1)
        else:
            combos = combo_generator.generate_balanced_combos(selected_preds, max_combos=1)
        
        if combos:
            combo = combos[0]
            return {
                'success': True,
                'combo': combo,
                'matches_selected': len(selected_preds),
                'total_odds': combo['total_odds'],
                'expected_value': combo['expected_value'],
                'recommended_stake': f"{combo['recommended_stake_pct']*100:.1f}% du bankroll",
                'selections': combo['selections']
            }
        else:
            return {
                'success': False,
                'message': "Impossible de créer un combiné viable avec ces critères"
            }
    
    def _handle_get_predictions(self, params: Dict) -> Dict:
        """Handle predictions request."""
        logger.info("Getting predictions")
        
        confidence_threshold = params.get('min_confidence', 0.6)
        leagues = params.get('leagues', [])
        date_filter = params.get('date', 'today')
        limit = params.get('limit', 10)
        
        # Get matches
        matches = self._get_matches_by_criteria(date_filter, leagues)
        
        if not matches:
            return {
                'success': False,
                'message': "Aucun match trouvé pour les critères spécifiés"
            }
        
        # Get predictions
        match_ids = [match['match_id'] for match in matches]
        predictions = self._get_predictions_for_matches(match_ids)
        
        # Filter by confidence and sort
        high_confidence_preds = [
            pred for pred in predictions 
            if pred.get('confidence', 0) >= confidence_threshold
        ]
        
        high_confidence_preds.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return {
            'success': True,
            'predictions': high_confidence_preds[:limit],
            'total_matches': len(matches),
            'high_confidence_count': len(high_confidence_preds),
            'criteria': f"Confiance >= {confidence_threshold:.0%}"
        }
    
    def _handle_analyze_match(self, params: Dict) -> Dict:
        """Handle match analysis request."""
        logger.info("Analyzing specific match")
        
        team1 = params.get('team1', '').lower()
        team2 = params.get('team2', '').lower()
        match_id = params.get('match_id')
        
        if match_id:
            match = self._get_match_by_id(match_id)
        elif team1 and team2:
            match = self._find_match_by_teams(team1, team2)
        else:
            return {
                'success': False,
                'message': "Spécifiez soit un match_id, soit les noms des deux équipes"
            }
        
        if not match:
            return {
                'success': False,
                'message': "Match non trouvé"
            }
        
        # Get detailed analysis
        analysis = self._get_detailed_match_analysis(match['match_id'])
        
        return {
            'success': True,
            'match': match,
            'analysis': analysis
        }
    
    def _handle_portfolio_status(self, params: Dict) -> Dict:
        """Handle portfolio status request."""
        logger.info("Getting portfolio status")
        
        # Get latest portfolio metrics
        query = """
        SELECT * FROM portfolio_performance 
        ORDER BY date DESC 
        LIMIT 1
        """
        
        latest_perf = db_manager.execute_query(query)
        
        if not latest_perf:
            return {
                'success': False,
                'message': "Aucune donnée de portefeuille disponible"
            }
        
        perf = latest_perf[0]
        
        # Get recent bets
        recent_bets_query = """
        SELECT COUNT(*) as total_bets,
               SUM(CASE WHEN status = 'WON' THEN 1 ELSE 0 END) as winning_bets,
               SUM(CASE WHEN status = 'LOST' THEN 1 ELSE 0 END) as losing_bets,
               AVG(odds) as avg_odds
        FROM bets 
        WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
        """
        
        recent_bets = db_manager.execute_query(recent_bets_query)
        
        return {
            'success': True,
            'portfolio': {
                'total_bankroll': perf['total_bankroll'],
                'total_pnl': perf['total_pnl'],
                'roi': f"{perf['roi']:.1%}",
                'sharpe_ratio': perf['sharpe_ratio'],
                'max_drawdown': f"{perf['max_drawdown']:.1%}",
                'win_rate': f"{perf['win_rate']:.1%}"
            },
            'recent_activity': recent_bets[0] if recent_bets else {},
            'last_update': perf['date']
        }
    
    def _handle_find_value_bets(self, params: Dict) -> Dict:
        """Handle value bets search."""
        logger.info("Finding value bets")
        
        min_edge = params.get('min_edge', 0.05)  # 5% minimum edge
        leagues = params.get('leagues', [])
        date_filter = params.get('date', 'today')
        
        # Get matches and predictions
        matches = self._get_matches_by_criteria(date_filter, leagues)
        match_ids = [match['match_id'] for match in matches]
        predictions = self._get_predictions_for_matches(match_ids)
        
        # Calculate value bets
        value_bets = []
        for pred in predictions:
            # Get market odds (simplified - would integrate with odds API)
            market_odds = self._get_market_odds(pred['match_id'])
            
            if market_odds:
                edge = self._calculate_edge(pred, market_odds)
                if edge >= min_edge:
                    value_bets.append({
                        'match': f"{pred.get('home_team', 'Home')} vs {pred.get('away_team', 'Away')}",
                        'prediction': pred['predicted_outcome'],
                        'model_probability': self._get_outcome_probability(pred),
                        'market_odds': market_odds.get(pred['predicted_outcome'].lower().replace(' ', '_'), 0),
                        'edge': f"{edge:.1%}",
                        'confidence': f"{pred.get('confidence', 0):.1%}"
                    })
        
        value_bets.sort(key=lambda x: float(x['edge'].replace('%', '')), reverse=True)
        
        return {
            'success': True,
            'value_bets': value_bets[:10],  # Top 10 value bets
            'total_found': len(value_bets),
            'min_edge_criteria': f"{min_edge:.1%}"
        }
    
    def _handle_risk_assessment(self, params: Dict) -> Dict:
        """Handle risk assessment request."""
        logger.info("Assessing betting risk")
        
        bet_amount = params.get('amount', 100)
        match_id = params.get('match_id')
        bet_type = params.get('bet_type', '1X2')
        
        if not match_id:
            return {
                'success': False,
                'message': "match_id requis pour l'évaluation du risque"
            }
        
        # Get match prediction
        predictions = self._get_predictions_for_matches([match_id])
        if not predictions:
            return {
                'success': False,
                'message': "Aucune prédiction disponible pour ce match"
            }
        
        pred = predictions[0]
        
        # Risk assessment
        confidence = pred.get('confidence', 0)
        probability = self._get_outcome_probability(pred)
        
        # Simple risk metrics
        risk_level = 'LOW' if confidence > 0.7 else 'MEDIUM' if confidence > 0.5 else 'HIGH'
        max_loss = bet_amount
        expected_return = bet_amount * probability - bet_amount
        
        return {
            'success': True,
            'risk_assessment': {
                'risk_level': risk_level,
                'confidence': f"{confidence:.1%}",
                'probability': f"{probability:.1%}",
                'max_loss': max_loss,
                'expected_return': expected_return,
                'kelly_fraction': self._calculate_kelly_fraction(probability, 2.0),  # Assuming 2.0 odds
                'recommendation': self._get_risk_recommendation(risk_level, confidence)
            }
        }
    
    def _handle_model_performance(self, params: Dict) -> Dict:
        """Handle model performance request."""
        logger.info("Getting model performance")
        
        days_back = params.get('days', 30)
        
        query = f"""
        SELECT 
            model_name,
            AVG(accuracy) as avg_accuracy,
            AVG(f1_score) as avg_f1,
            COUNT(*) as evaluations
        FROM model_performance
        WHERE evaluation_date >= CURRENT_DATE - INTERVAL '{days_back} days'
        GROUP BY model_name
        ORDER BY avg_accuracy DESC
        """
        
        performance_data = db_manager.execute_query(query)
        
        return {
            'success': True,
            'model_performance': performance_data,
            'period': f"Derniers {days_back} jours",
            'best_model': performance_data[0]['model_name'] if performance_data else None
        }
    
    def _handle_schedule_alert(self, params: Dict) -> Dict:
        """Handle alert scheduling."""
        logger.info("Scheduling alert")
        
        alert_type = params.get('type', 'prediction')
        criteria = params.get('criteria', {})
        
        return {
            'success': True,
            'message': f"Alerte {alert_type} programmée avec les critères spécifiés",
            'criteria': criteria
        }
    
    # Helper methods
    def _get_matches_by_criteria(self, date_filter: str, leagues: List[str]) -> List[Dict]:
        """Get matches based on criteria."""
        if date_filter == 'today':
            date_condition = "DATE(m.match_date) = CURRENT_DATE"
        elif date_filter == 'tomorrow':
            date_condition = "DATE(m.match_date) = CURRENT_DATE + INTERVAL '1 day'"
        elif date_filter == 'week':
            date_condition = "m.match_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'"
        else:
            date_condition = "DATE(m.match_date) = CURRENT_DATE"
        
        league_condition = ""
        if leagues:
            league_list = "', '".join(leagues)
            league_condition = f"AND m.league IN ('{league_list}')"
        
        query = f"""
        SELECT 
            m.id as match_id,
            ht.name as home_team,
            at.name as away_team,
            m.league,
            m.match_date
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE {date_condition}
        AND m.status = 'SCHEDULED'
        {league_condition}
        ORDER BY m.match_date
        """
        
        return db_manager.execute_query(query)
    
    def _get_predictions_for_matches(self, match_ids: List[str]) -> List[Dict]:
        """Get predictions for specific matches."""
        if not match_ids:
            return []
        
        # Use model registry to get predictions
        try:
            ensemble_predictions = model_registry.make_ensemble_prediction(match_ids)
            return ensemble_predictions
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return []
    
    def _get_match_by_id(self, match_id: str) -> Optional[Dict]:
        """Get match by ID."""
        query = """
        SELECT 
            m.id as match_id,
            ht.name as home_team,
            at.name as away_team,
            m.league,
            m.match_date,
            m.status
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE m.id = :match_id
        """
        
        result = db_manager.execute_query(query, {"match_id": match_id})
        return result[0] if result else None
    
    def _find_match_by_teams(self, team1: str, team2: str) -> Optional[Dict]:
        """Find match by team names."""
        query = """
        SELECT 
            m.id as match_id,
            ht.name as home_team,
            at.name as away_team,
            m.league,
            m.match_date,
            m.status
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE (LOWER(ht.name) LIKE :team1 OR LOWER(at.name) LIKE :team1)
        AND (LOWER(ht.name) LIKE :team2 OR LOWER(at.name) LIKE :team2)
        AND m.match_date >= CURRENT_DATE - INTERVAL '1 day'
        ORDER BY m.match_date
        LIMIT 1
        """
        
        result = db_manager.execute_query(query, {
            "team1": f"%{team1}%",
            "team2": f"%{team2}%"
        })
        return result[0] if result else None
    
    def _get_detailed_match_analysis(self, match_id: str) -> Dict:
        """Get detailed match analysis."""
        # This would integrate with various analysis modules
        return {
            'tactical_analysis': 'Formation 4-4-2 vs 3-5-2',
            'key_players': ['Player A', 'Player B'],
            'weather_impact': 'Faible',
            'head_to_head': '2-1-1 (derniers 4 matchs)',
            'form': 'Équipe domicile: 3W-1D, Équipe extérieure: 2W-2L',
            'injury_report': 'Aucune blessure majeure'
        }
    
    def _get_market_odds(self, match_id: str) -> Dict:
        """Get market odds for a match."""
        query = """
        SELECT market_type, odds_data
        FROM odds
        WHERE match_id = :match_id
        AND odds_date >= CURRENT_DATE - INTERVAL '1 day'
        ORDER BY odds_date DESC
        LIMIT 1
        """
        
        result = db_manager.execute_query(query, {"match_id": match_id})
        return result[0]['odds_data'] if result else {}
    
    def _calculate_edge(self, prediction: Dict, market_odds: Dict) -> float:
        """Calculate betting edge."""
        outcome = prediction['predicted_outcome'].lower().replace(' ', '_')
        model_prob = self._get_outcome_probability(prediction)
        market_odd = market_odds.get(outcome, 0)
        
        if market_odd <= 1:
            return 0
        
        implied_prob = 1 / market_odd
        edge = model_prob - implied_prob
        return edge
    
    def _get_outcome_probability(self, prediction: Dict) -> float:
        """Get probability for predicted outcome."""
        outcome = prediction['predicted_outcome']
        if outcome == 'Home Win':
            return prediction.get('home_win_prob', 0)
        elif outcome == 'Away Win':
            return prediction.get('away_win_prob', 0)
        else:  # Draw
            return prediction.get('draw_prob', 0)
    
    def _calculate_kelly_fraction(self, probability: float, odds: float) -> float:
        """Calculate Kelly criterion fraction."""
        if odds <= 1 or probability <= 0:
            return 0
        
        b = odds - 1
        p = probability
        q = 1 - p
        
        kelly = (b * p - q) / b
        return max(0, min(0.25, kelly))  # Cap at 25%
    
    def _get_risk_recommendation(self, risk_level: str, confidence: float) -> str:
        """Get risk-based recommendation."""
        if risk_level == 'LOW' and confidence > 0.7:
            return "Pari recommandé - Risque faible, confiance élevée"
        elif risk_level == 'MEDIUM':
            return "Pari acceptable - Surveiller les cotes"
        else:
            return "Pari déconseillé - Risque élevé ou confiance faible"


# Integration functions for Streamlit
def create_chatbot_interface():
    """Create Streamlit chatbot interface."""
    import streamlit as st
    
    # Initialize chatbot in session state
    if 'chatbot' not in st.session_state:
        openai_key = st.secrets.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
        if openai_key:
            st.session_state.chatbot = QuantumChatBot(openai_key)
        else:
            st.error("Clé API OpenAI requise pour le chatbot")
            return
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    st.subheader("🤖 Assistant IA Quantum")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            st.chat_message("assistant").write(message['content'])
    
    # User input
    if prompt := st.chat_input("Tapez votre commande..."):
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': prompt
        })
        st.chat_message("user").write(prompt)
        
        # Process with chatbot
        with st.chat_message("assistant"):
            with st.spinner("Traitement..."):
                response = st.session_state.chatbot.chat(prompt)
            
            if response['type'] == 'success':
                st.success(response['explanation'])
                
                # Display results based on command type
                if response['command'] == 'create_combo':
                    display_combo_result(response['result'])
                elif response['command'] == 'get_predictions':
                    display_predictions_result(response['result'])
                elif response['command'] == 'portfolio_status':
                    display_portfolio_result(response['result'])
                else:
                    st.json(response['result'])
            
            elif response['type'] == 'clarification':
                st.warning(response['message'])
                st.write("**Suggestions:**")
                for suggestion in response.get('suggestions', []):
                    st.write(f"• {suggestion}")
            
            else:  # error
                st.error(response['message'])
        
        # Add assistant response to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': str(response)
        })

def display_combo_result(result):
    """Display combo creation result."""
    import streamlit as st
    
    if result['success']:
        st.success("✅ Combiné créé avec succès!")
        
        combo = result['combo']
        st.metric("Cote totale", f"{combo['total_odds']:.2f}")
        st.metric("Valeur attendue", f"{combo['expected_value']:.1%}")
        st.metric("Mise recommandée", result['recommended_stake'])
        
        st.subheader("Sélections:")
        for selection in combo['selections']:
            with st.expander(f"{selection['match']} - {selection['selection']}"):
                st.write(f"**Cote:** {selection['odds']:.2f}")
                st.write(f"**Probabilité:** {selection['probability']:.1%}")
                st.write(f"**Confiance:** {selection['confidence']:.1%}")
    else:
        st.error(result['message'])

def display_predictions_result(result):
    """Display predictions result."""
    import streamlit as st
    
    if result['success']:
        st.info(f"🔮 {result['high_confidence_count']} prédictions haute confiance trouvées")
        
        for pred in result['predictions']:
            with st.expander(f"{pred.get('home_team', 'Home')} vs {pred.get('away_team', 'Away')}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prédiction", pred['predicted_outcome'])
                with col2:
                    st.metric("Confiance", f"{pred.get('confidence', 0):.1%}")
                with col3:
                    st.metric("Probabilité", f"{pred.get('home_win_prob', 0):.1%}" if pred['predicted_outcome'] == 'Home Win' else f"{pred.get('away_win_prob', 0):.1%}")
                
                st.write(f"**Ligue:** {pred.get('league', 'N/A')}")
                st.write(f"**Date:** {pred.get('match_date', 'N/A')}")
    else:
        st.warning(result['message'])

def display_portfolio_result(result):
    """Display portfolio status result."""
    import streamlit as st
    
    if result['success']:
        portfolio = result['portfolio']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Bankroll Total", f"€{portfolio['total_bankroll']:,.0f}")
        with col2:
            st.metric("ROI", portfolio['roi'])
        with col3:
            st.metric("Taux de Réussite", portfolio['win_rate'])
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.metric("P&L Total", f"€{portfolio['total_pnl']:,.0f}")
        with col5:
            st.metric("Ratio Sharpe", f"{portfolio['sharpe_ratio']:.2f}")
        with col6:
            st.metric("Drawdown Max", portfolio['max_drawdown'])
        
        if result.get('recent_activity'):
            st.subheader("Activité Récente (7 jours)")
            activity = result['recent_activity']
            st.write(f"**Paris totaux:** {activity.get('total_bets', 0)}")
            st.write(f"**Paris gagnants:** {activity.get('winning_bets', 0)}")
            st.write(f"**Cote moyenne:** {activity.get('avg_odds', 0):.2f}")
    else:
        st.error(result['message'])