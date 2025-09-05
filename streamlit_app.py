"""Streamlit dashboard for Bot Quantum Max."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append('src')
sys.path.append('.')

from src.data.database import db_manager, match_queries
from src.models.supervised import SupervisedMatchPredictor
from src.models.base import model_registry
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.combos import ComboGenerator
from config.settings import settings

# Page configuration
st.set_page_config(
    page_title="Bot Quantum Max",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #2e8b57;
    }
    
    .high-confidence {
        border-left-color: #228b22 !important;
    }
    
    .medium-confidence {
        border-left-color: #ffa500 !important;
    }
    
    .low-confidence {
        border-left-color: #ff6b6b !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data with caching."""
    try:
        # Get today's matches
        today_matches = get_todays_matches()
        
        # Get recent predictions
        recent_predictions = get_recent_predictions()
        
        # Get model performance
        model_performance = get_model_performance()
        
        # Get portfolio performance
        portfolio_performance = get_portfolio_performance()
        
        return {
            'matches': today_matches,
            'predictions': recent_predictions,
            'model_performance': model_performance,
            'portfolio_performance': portfolio_performance
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def get_todays_matches():
    """Get today's matches."""
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    
    query = """
    SELECT 
        m.id as match_id,
        m.match_date,
        ht.name as home_team,
        at.name as away_team,
        m.league,
        m.status
    FROM matches m
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    WHERE DATE(m.match_date) BETWEEN %s AND %s
    ORDER BY m.match_date
    """
    
    return db_manager.get_dataframe(query, params=(today, tomorrow))


def get_recent_predictions():
    """Get recent predictions."""
    query = """
    SELECT 
        p.match_id,
        p.model_name,
        p.prediction_date,
        p.home_win_prob,
        p.draw_prob,
        p.away_win_prob,
        p.confidence_score,
        m.match_date,
        ht.name as home_team,
        at.name as away_team,
        m.league
    FROM predictions p
    JOIN matches m ON p.match_id = m.id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    WHERE p.prediction_date >= CURRENT_DATE - INTERVAL '7 days'
    ORDER BY p.prediction_date DESC, p.confidence_score DESC
    """
    
    return db_manager.get_dataframe(query)


def get_model_performance():
    """Get model performance metrics."""
    query = """
    SELECT 
        model_name,
        evaluation_date,
        dataset_type,
        accuracy,
        precision,
        recall,
        f1_score,
        log_loss,
        sample_size
    FROM model_performance
    WHERE evaluation_date >= CURRENT_DATE - INTERVAL '30 days'
    ORDER BY evaluation_date DESC
    """
    
    return db_manager.get_dataframe(query)


def get_portfolio_performance():
    """Get portfolio performance data."""
    query = """
    SELECT 
        date,
        total_bankroll,
        daily_pnl,
        total_pnl,
        roi,
        sharpe_ratio,
        max_drawdown,
        win_rate,
        total_bets
    FROM portfolio_performance
    WHERE date >= CURRENT_DATE - INTERVAL '90 days'
    ORDER BY date
    """
    
    return db_manager.get_dataframe(query)


def main():
    """Main application."""
    # Header
    st.markdown('<h1 class="main-header">⚽ Bot Quantum Max</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Football Betting Intelligence Platform**")
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.selectbox(
            "Select Page",
            ["🏠 Dashboard", "🔮 Predictions", "📊 Analytics", "💰 Portfolio", "🎯 Combinations", "⚙️ Settings"]
        )
        
        # Status indicators
        st.subheader("System Status")
        
        try:
            data = load_data()
            if data:
                st.success("✅ Data loaded successfully")
                st.info(f"📊 {len(data['matches'])} matches today")
                st.info(f"🔮 {len(data['predictions'])} recent predictions")
            else:
                st.error("❌ Data loading failed")
        except Exception as e:
            st.error(f"❌ System error: {str(e)[:50]}...")
        
        # Quick stats
        if 'data' in locals() and data:
            st.subheader("Quick Stats")
            if not data['portfolio_performance'].empty:
                latest_perf = data['portfolio_performance'].iloc[-1]
                st.metric("Current ROI", f"{latest_perf['roi']:.1%}")
                st.metric("Win Rate", f"{latest_perf['win_rate']:.1%}")
    
    # Main content based on selected page
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "🔮 Predictions":
        predictions_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "💰 Portfolio":
        portfolio_page()
    elif page == "🎯 Combinations":
        combinations_page()
    elif page == "⚙️ Settings":
        settings_page()


def dashboard_page():
    """Dashboard overview page."""
    st.header("📊 Dashboard Overview")
    
    data = load_data()
    if not data:
        st.error("Unable to load data")
        return
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Today's Matches",
            len(data['matches']),
            delta=None
        )
    
    with col2:
        high_conf_preds = len([p for _, p in data['predictions'].iterrows() 
                              if p['confidence_score'] >= settings.confidence_threshold])
        st.metric(
            "High Confidence Predictions",
            high_conf_preds,
            delta=None
        )
    
    with col3:
        if not data['portfolio_performance'].empty:
            latest_roi = data['portfolio_performance'].iloc[-1]['roi']
            st.metric(
                "Current ROI",
                f"{latest_roi:.1%}",
                delta=f"{latest_roi:.1%}"
            )
    
    with col4:
        if not data['model_performance'].empty:
            avg_accuracy = data['model_performance'].groupby('model_name')['accuracy'].mean().mean()
            st.metric(
                "Avg Model Accuracy",
                f"{avg_accuracy:.1%}",
                delta=None
            )
    
    # Today's matches overview
    st.subheader("🏆 Today's Matches")
    
    if not data['matches'].empty:
        for _, match in data['matches'].iterrows():
            # Get prediction for this match
            match_preds = data['predictions'][data['predictions']['match_id'] == match['match_id']]
            
            if not match_preds.empty:
                pred = match_preds.iloc[0]  # Get first prediction
                
                # Determine confidence level for styling
                conf_level = "high" if pred['confidence_score'] >= 0.7 else "medium" if pred['confidence_score'] >= 0.5 else "low"
                
                st.markdown(f"""
                <div class="prediction-card {conf_level}-confidence">
                    <h4>{match['home_team']} vs {match['away_team']}</h4>
                    <p><strong>League:</strong> {match['league']} | <strong>Time:</strong> {match['match_date']}</p>
                    <p><strong>Prediction:</strong> Home: {pred['home_win_prob']:.1%} | Draw: {pred['draw_prob']:.1%} | Away: {pred['away_win_prob']:.1%}</p>
                    <p><strong>Confidence:</strong> {pred['confidence_score']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No matches scheduled for today")
    
    # Quick charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Recent ROI Trend")
        if not data['portfolio_performance'].empty:
            fig = px.line(
                data['portfolio_performance'], 
                x='date', 
                y='roi',
                title="ROI Over Time"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No portfolio data available")
    
    with col2:
        st.subheader("🎯 Model Performance")
        if not data['model_performance'].empty:
            model_acc = data['model_performance'].groupby('model_name')['accuracy'].mean().reset_index()
            fig = px.bar(
                model_acc,
                x='model_name',
                y='accuracy',
                title="Model Accuracy Comparison"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model performance data available")


def predictions_page():
    """Predictions page."""
    st.header("🔮 Match Predictions")
    
    data = load_data()
    if not data:
        st.error("Unable to load data")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        leagues = ['All'] + list(data['predictions']['league'].unique())
        selected_league = st.selectbox("League", leagues)
    
    with col2:
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)
    
    with col3:
        models = ['All'] + list(data['predictions']['model_name'].unique())
        selected_model = st.selectbox("Model", models)
    
    # Filter predictions
    filtered_preds = data['predictions'].copy()
    
    if selected_league != 'All':
        filtered_preds = filtered_preds[filtered_preds['league'] == selected_league]
    
    if selected_model != 'All':
        filtered_preds = filtered_preds[filtered_preds['model_name'] == selected_model]
    
    filtered_preds = filtered_preds[filtered_preds['confidence_score'] >= min_confidence]
    
    # Display predictions
    st.subheader(f"📊 Filtered Predictions ({len(filtered_preds)} results)")
    
    if not filtered_preds.empty:
        # Create interactive table
        display_df = filtered_preds[['home_team', 'away_team', 'league', 'match_date', 
                                   'home_win_prob', 'draw_prob', 'away_win_prob', 
                                   'confidence_score', 'model_name']].copy()
        
        # Format percentages
        for col in ['home_win_prob', 'draw_prob', 'away_win_prob', 'confidence_score']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Prediction distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Confidence Distribution")
            fig = px.histogram(
                filtered_preds,
                x='confidence_score',
                nbins=20,
                title="Distribution of Prediction Confidence"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Outcome Probability Distribution")
            outcome_data = []
            for _, pred in filtered_preds.iterrows():
                outcome_data.extend([
                    {'Outcome': 'Home Win', 'Probability': pred['home_win_prob']},
                    {'Outcome': 'Draw', 'Probability': pred['draw_prob']},
                    {'Outcome': 'Away Win', 'Probability': pred['away_win_prob']}
                ])
            
            outcome_df = pd.DataFrame(outcome_data)
            fig = px.box(
                outcome_df,
                x='Outcome',
                y='Probability',
                title="Outcome Probability Distributions"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No predictions match the selected criteria")


def analytics_page():
    """Analytics and model performance page."""
    st.header("📊 Analytics & Model Performance")
    
    data = load_data()
    if not data:
        st.error("Unable to load data")
        return
    
    if data['model_performance'].empty:
        st.warning("No model performance data available")
        return
    
    # Model performance comparison
    st.subheader("🤖 Model Performance Comparison")
    
    perf_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    selected_metric = st.selectbox("Select Metric", perf_metrics)
    
    # Group by model and get latest performance
    latest_perf = data['model_performance'].sort_values('evaluation_date').groupby('model_name').tail(1)
    
    fig = px.bar(
        latest_perf,
        x='model_name',
        y=selected_metric,
        title=f"Latest {selected_metric.title()} by Model",
        color=selected_metric,
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance over time
    st.subheader("📈 Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            data['model_performance'],
            x='evaluation_date',
            y='accuracy',
            color='model_name',
            title="Accuracy Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(
            data['model_performance'],
            x='evaluation_date',
            y='f1_score',
            color='model_name',
            title="F1 Score Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📋 Detailed Performance Metrics")
    
    metrics_summary = latest_perf.groupby('model_name').agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'log_loss': 'mean',
        'sample_size': 'sum'
    }).round(3)
    
    st.dataframe(metrics_summary, use_container_width=True)


def portfolio_page():
    """Portfolio management page."""
    st.header("💰 Portfolio Management")
    
    data = load_data()
    if not data:
        st.error("Unable to load data")
        return
    
    if data['portfolio_performance'].empty:
        st.warning("No portfolio data available")
        return
    
    # Portfolio overview
    latest_portfolio = data['portfolio_performance'].iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Bankroll",
            f"${latest_portfolio['total_bankroll']:,.0f}",
            delta=f"${latest_portfolio['daily_pnl']:,.0f}"
        )
    
    with col2:
        st.metric(
            "Total P&L",
            f"${latest_portfolio['total_pnl']:,.0f}",
            delta=f"{latest_portfolio['roi']:.1%}"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{latest_portfolio['sharpe_ratio']:.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{latest_portfolio['win_rate']:.1%}",
            delta=None
        )
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Bankroll Evolution")
        fig = px.line(
            data['portfolio_performance'],
            x='date',
            y='total_bankroll',
            title="Bankroll Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💹 Daily P&L")
        fig = px.bar(
            data['portfolio_performance'],
            x='date',
            y='daily_pnl',
            title="Daily Profit & Loss",
            color='daily_pnl',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk metrics
    st.subheader("⚠️ Risk Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            data['portfolio_performance'],
            x='date',
            y='max_drawdown',
            title="Maximum Drawdown Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(
            data['portfolio_performance'],
            x='date',
            y='sharpe_ratio',
            title="Sharpe Ratio Evolution"
        )
        st.plotly_chart(fig, use_container_width=True)


def ai_assistant_page():
    """AI Assistant page with chatbot interface."""
    st.header("🤖 Assistant IA Quantum")
    
    st.markdown("""
    **Commandez Bot Quantum Max avec des instructions en langage naturel !**
    
    **Exemples de commandes :**
    - *"Crée un combiné avec 3 matchs d'aujourd'hui ayant des cotes supérieures à 1.50"*
    - *"Montre-moi les prédictions les plus confiantes pour demain"*
    - *"Analyse le match PSG vs Lyon en détail"*
    - *"Quel est le statut de mon portefeuille ?"*
    - *"Trouve-moi des paris à valeur élevée pour ce weekend"*
    """)
    
    # Check for OpenAI API key
    openai_key = st.secrets.get('OPENAI_API_KEY', '') if hasattr(st, 'secrets') else os.environ.get('OPENAI_API_KEY', '')
    
    if not openai_key:
        st.warning("⚠️ Clé API OpenAI non configurée")
        
        with st.expander("Configuration OpenAI"):
            api_key_input = st.text_input(
                "Entrez votre clé API OpenAI:",
                type="password",
                help="Obtenez votre clé sur https://platform.openai.com/api-keys"
            )
            
            if api_key_input:
                os.environ['OPENAI_API_KEY'] = api_key_input
                st.success("✅ Clé API configurée")
                st.rerun()
        return
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        from src.ui.chatbot import QuantumChatBot
        try:
            st.session_state.chatbot = QuantumChatBot(openai_key, model="gpt-4")
            st.success("🤖 Assistant IA initialisé")
        except Exception as e:
            st.error(f"Erreur lors de l'initialisation: {e}")
            return
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Quick action buttons
    st.subheader("🚀 Actions Rapides")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Combiné du Jour"):
            quick_command = "Crée un combiné équilibré avec 3 matchs d'aujourd'hui ayant des cotes entre 1.50 et 3.00"
            st.session_state.pending_command = quick_command
    
    with col2:
        if st.button("🎯 Top Prédictions"):
            quick_command = "Montre-moi les 5 meilleures prédictions avec une confiance supérieure à 70%"
            st.session_state.pending_command = quick_command
    
    with col3:
        if st.button("💰 Statut Portfolio"):
            quick_command = "Quel est le statut actuel de mon portefeuille de paris ?"
            st.session_state.pending_command = quick_command
    
    with col4:
        if st.button("🔍 Paris Valeur"):
            quick_command = "Trouve-moi des paris à valeur élevée pour les prochains matchs"
            st.session_state.pending_command = quick_command
    
    # Chat interface
    st.subheader("💬 Discussion avec l'Assistant")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    if isinstance(message['content'], str):
                        st.write(message['content'])
                    else:
                        # Display structured response
                        response = message['content']
                        if response.get('type') == 'success':
                            st.success(response.get('explanation', ''))
                            
                            # Display results based on command
                            result = response.get('result', {})
                            command = response.get('command', '')
                            
                            if command == 'create_combo' and result.get('success'):
                                display_combo_in_chat(result)
                            elif command == 'get_predictions' and result.get('success'):
                                display_predictions_in_chat(result)
                            elif command == 'portfolio_status' and result.get('success'):
                                display_portfolio_in_chat(result)
                            elif command == 'find_value_bets' and result.get('success'):
                                display_value_bets_in_chat(result)
                            else:
                                st.json(result)
                        
                        elif response.get('type') == 'clarification':
                            st.warning(response.get('message', ''))
                            if response.get('suggestions'):
                                st.write("**Suggestions:**")
                                for suggestion in response['suggestions']:
                                    st.write(f"• {suggestion}")
                        
                        else:  # error
                            st.error(response.get('message', 'Erreur inconnue'))
    
    # Handle pending command from quick buttons
    if hasattr(st.session_state, 'pending_command'):
        user_input = st.session_state.pending_command
        del st.session_state.pending_command
        process_chat_input(user_input)
    
    # User input
    user_input = st.chat_input("Tapez votre commande ici...")
    if user_input:
        process_chat_input(user_input)


def process_chat_input(user_input: str):
    """Process user input and get chatbot response."""
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input
    })
    
    # Get chatbot response
    with st.spinner("🤖 L'assistant réfléchit..."):
        try:
            response = st.session_state.chatbot.chat(user_input)
            
            # Add response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            
        except Exception as e:
            error_response = {
                'type': 'error',
                'message': f"Erreur de traitement: {str(e)}"
            }
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': error_response
            })
    
    st.rerun()


def display_combo_in_chat(result):
    """Display combo result in chat."""
    if result.get('success'):
        combo = result['combo']
        
        st.success("✅ Combiné créé avec succès!")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cote Totale", f"{combo['total_odds']:.2f}")
        with col2:
            st.metric("Valeur Attendue", f"{combo['expected_value']:.1%}")
        with col3:
            st.metric("Mise Recommandée", result['recommended_stake'])
        
        # Selections
        st.write("**Sélections:**")
        selections_df = pd.DataFrame(combo['selections'])
        st.dataframe(
            selections_df[['match', 'selection', 'odds', 'probability', 'confidence']],
            use_container_width=True
        )
    else:
        st.error(result.get('message', 'Échec de création du combiné'))


def display_predictions_in_chat(result):
    """Display predictions result in chat."""
    if result.get('success'):
        predictions = result['predictions']
        
        st.info(f"🔮 {len(predictions)} prédictions trouvées")
        
        # Create predictions dataframe
        pred_data = []
        for pred in predictions:
            pred_data.append({
                'Match': f"{pred.get('home_team', 'Home')} vs {pred.get('away_team', 'Away')}",
                'Prédiction': pred['predicted_outcome'],
                'Confiance': f"{pred.get('confidence', 0):.1%}",
                'Probabilité': f"{pred.get('home_win_prob' if pred['predicted_outcome'] == 'Home Win' else 'away_win_prob' if pred['predicted_outcome'] == 'Away Win' else 'draw_prob', 0):.1%}",
                'Ligue': pred.get('league', 'N/A')
            })
        
        if pred_data:
            st.dataframe(pd.DataFrame(pred_data), use_container_width=True)
    else:
        st.warning(result.get('message', 'Aucune prédiction trouvée'))


def display_portfolio_in_chat(result):
    """Display portfolio result in chat."""
    if result.get('success'):
        portfolio = result['portfolio']
        
        st.info("💰 Statut du Portefeuille")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bankroll", f"€{portfolio['total_bankroll']:,.0f}")
        with col2:
            st.metric("ROI", portfolio['roi'])
        with col3:
            st.metric("Win Rate", portfolio['win_rate'])
        
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("P&L Total", f"€{portfolio['total_pnl']:,.0f}")
        with col5:
            st.metric("Sharpe", f"{portfolio['sharpe_ratio']:.2f}")
        with col6:
            st.metric("Drawdown", portfolio['max_drawdown'])
    else:
        st.error(result.get('message', 'Impossible d\'obtenir le statut du portefeuille'))


def display_value_bets_in_chat(result):
    """Display value bets in chat."""
    if result.get('success'):
        value_bets = result['value_bets']
        
        st.info(f"💎 {len(value_bets)} paris à valeur trouvés")
        
        if value_bets:
            value_df = pd.DataFrame(value_bets)
            st.dataframe(value_df, use_container_width=True)
        else:
            st.write("Aucun pari à valeur trouvé avec les critères actuels.")
    else:
        st.warning(result.get('message', 'Impossible de trouver des paris à valeur'))
    """Betting combinations page."""
    st.header("🎯 Betting Combinations")
    
    st.info("This feature would generate optimal betting combinations using the ComboGenerator class")
    
    # Placeholder for combination generator integration
    if st.button("Generate Today's Combinations"):
        with st.spinner("Generating combinations..."):
            st.success("Combination generation would be implemented here")


def settings_page():
    """Settings and configuration page."""
    st.header("⚙️ Settings & Configuration")
    
    # Model settings
    st.subheader("🤖 Model Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        0.5, 1.0, settings.confidence_threshold, 0.05,
        help="Minimum confidence required for high-confidence predictions"
    )
    
    min_edge = st.slider(
        "Minimum Edge",
        0.01, 0.2, settings.min_edge, 0.01,
        help="Minimum expected edge required for betting opportunities"
    )
    
    max_kelly = st.slider(
        "Maximum Kelly Fraction",
        0.1, 0.5, settings.max_kelly_fraction, 0.05,
        help="Maximum fraction of bankroll to bet using Kelly Criterion"
    )
    
    # Data settings
    st.subheader("📊 Data Settings")
    
    lookback_days = st.number_input(
        "Lookback Period (days)",
        30, 365, settings.lookback_days,
        help="Number of days to look back for feature engineering"
    )
    
    monte_carlo_runs = st.number_input(
        "Monte Carlo Runs",
        1000, 50000, settings.max_monte_carlo_runs,
        help="Number of Monte Carlo simulation runs"
    )
    
    # Save settings
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
    
    # System information
    st.subheader("ℹ️ System Information")
    
    st.info(f"Environment: {settings.environment}")
    st.info(f"Log Level: {settings.log_level}")
    st.info(f"Supported Leagues: {', '.join(settings.supported_leagues)}")


if __name__ == "__main__":
    main()