"""Modern mobile-like interface for Bot Quantum Max."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append('src')
sys.path.append('.')

from src.data.database import db_manager
from src.ui.chatbot import QuantumChatBot

# Configure Streamlit for mobile-like appearance
st.set_page_config(
    page_title="⚽ Quantum Bet",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/quantum-sports',
        'Report a bug': "https://github.com/quantum-sports/issues",
        'About': "Quantum Bet - AI-Powered Sports Betting"
    }
)

# Progressive Web App Setup
def setup_pwa():
    """Setup Progressive Web App features."""
    
    # PWA Manifest
    st.markdown("""
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#667eea">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="Quantum Bet">
    <link rel="apple-touch-icon" href="/static/icons/icon-152x152.png">
    <meta name="msapplication-TileImage" content="/static/icons/icon-144x144.png">
    <meta name="msapplication-TileColor" content="#667eea">
    """, unsafe_allow_html=True)
    
    # Service Worker Registration
    st.markdown("""
    <script>
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', function() {
        navigator.serviceWorker.register('/static/sw.js')
          .then(function(registration) {
            console.log('SW registered: ', registration);
            
            // Request notification permission
            if ('Notification' in window && Notification.permission === 'default') {
              Notification.requestPermission().then(function(permission) {
                console.log('Notification permission:', permission);
              });
            }
          })
          .catch(function(registrationError) {
            console.log('SW registration failed: ', registrationError);
          });
      });
    }
    
    // Install prompt
    let deferredPrompt;
    const installButton = document.getElementById('install-app');
    
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      deferredPrompt = e;
      if (installButton) {
        installButton.style.display = 'block';
      }
    });
    
    function installApp() {
      if (deferredPrompt) {
        deferredPrompt.prompt();
        deferredPrompt.userChoice.then((result) => {
          console.log('Install prompt result:', result);
          deferredPrompt = null;
        });
      }
    }
    </script>
    """, unsafe_allow_html=True)

# Custom CSS for mobile app appearance
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stActionButton {display:none;}
    header {visibility: hidden;}
    
    /* Mobile app styling */
    .main .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* PWA Install Button */
    #install-app {
        display: none;
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 15px 20px;
        font-weight: bold;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        cursor: pointer;
        z-index: 1000;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-10px);
        }
        60% {
            transform: translateY(-5px);
        }
    }
    
    /* App header */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0 0 25px 25px;
        margin: -1rem -1rem 1rem -1rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        position: sticky;
        top: -1rem;
        z-index: 100;
    }
    
    .app-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0 0 0.3rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .app-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    
    /* Bottom Navigation */
    .bottom-nav {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        border-top: 1px solid #e5e7eb;
        display: flex;
        justify-content: space-around;
        padding: 0.5rem 0;
        z-index: 1000;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
    }
    
    .nav-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border-radius: 15px;
        min-width: 60px;
    }
    
    .nav-item:hover {
        background: #f8f9ff;
        transform: translateY(-2px);
    }
    
    .nav-item.active {
        color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    .nav-icon {
        font-size: 1.2rem;
        margin-bottom: 0.2rem;
    }
    
    .nav-label {
        font-size: 0.7rem;
        font-weight: 500;
    }
    
    /* Cards with glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
    }
    
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        transition: transform 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.1);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }"""Modern mobile-like interface for Bot Quantum Max."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append('src')
sys.path.append('.')

from src.data.database import db_manager
from src.ui.chatbot import QuantumChatBot

# Configure Streamlit for mobile-like appearance
st.set_page_config(
    page_title="⚽ Quantum Bet",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/quantum-sports',
        'Report a bug': "https://github.com/quantum-sports/issues",
        'About': "Quantum Bet - AI-Powered Sports Betting"
    }
)

# Custom CSS for mobile app appearance
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stActionButton {display:none;}
    
    /* Mobile app styling */
    .main .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* App header */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .app-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Navigation tabs */
    .nav-container {
        background: white;
        border-radius: 20px;
        padding: 0.3rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        display: flex;
        justify-content: space-around;
    }
    
    .nav-tab {
        flex: 1;
        text-align: center;
        padding: 0.8rem;
        border-radius: 15px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .nav-tab.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .nav-tab:hover {
        background: #f8f9ff;
        transform: translateY(-2px);
    }
    
    /* Cards */
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border-radius: 20px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 6px 25px rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .prediction-card:active {
        transform: scale(0.98);
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 50px;
        height: 50px;
        background: radial-gradient(circle, rgba(102,126,234,0.1) 0%, transparent 70%);
    }
    
    .high-confidence {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
    }
    
    .medium-confidence {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }
    
    .low-confidence {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    }
    
    /* Floating Action Button */
    .fab {
        position: fixed;
        bottom: 80px;
        right: 20px;
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        z-index: 999;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .fab:hover {
        transform: scale(1.1);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Chat interface improvements */
    .chat-container {
        background: white;
        border-radius: 25px;
        padding: 1rem;
        margin: 1rem 0 100px 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        max-height: 60vh;
        overflow-y: auto;
    }
    
    .chat-message {
        margin: 1rem 0;
        animation: slideIn 0.3s ease-out;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin-left: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        word-wrap: break-word;
    }
    
    .bot-message {
        background: #f8f9fa;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin-right: 2rem;
        border: 1px solid #e9ecef;
        word-wrap: break-word;
    }
    
    /* Quick actions grid */
    .quick-actions {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .quick-action {
        background: white;
        border: 2px solid #667eea;
        color: #667eea;
        border-radius: 15px;
        padding: 1rem;
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        min-height: 80px;
        justify-content: center;
    }
    
    .quick-action:hover {
        background: #667eea;
        color: white;
        transform: scale(1.02);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .quick-action-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Statistics grid */
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1.5rem 1rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stat-item:hover {
        transform: translateY(-2px);
    }
    
    .stat-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6, #f59e0b);
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 500;
    }
    
    .stat-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }
    
    /* Swipeable cards */
    .swipeable-card {
        touch-action: pan-y;
        transition: transform 0.3s ease;
    }
    
    .swipeable-card.swiping {
        transition: none;
    }
    
    /* Loading animations */
    .loading-skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
    }
    
    @keyframes loading {
        0% {
            background-position: 200% 0;
        }
        100% {
            background-position: -200% 0;
        }
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Success/Error states */
    .success-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border-left: 5px solid #10b981;
        color: #065f46;
    }
    
    .error-card {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 5px solid #ef4444;
        color: #7f1d1d;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 5px solid #f59e0b;
        color: #78350f;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .metric-card, .prediction-card, .chat-container, .stat-item {
            background: #1f2937;
            color: white;
            border-color: #374151;
        }
        
        .bottom-nav {
            background: #1f2937;
            border-top-color: #374151;
        }
        
        .app-header {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        }
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .stats-grid {
            grid-template-columns: 1fr;
        }
        
        .quick-actions {
            grid-template-columns: 1fr;
        }
        
        .app-title {
            font-size: 1.5rem;
        }
        
        .prediction-card {
            margin: 0.5rem 0;
            padding: 1rem;
        }
        
        .user-message, .bot-message {
            margin-left: 0.5rem;
            margin-right: 0.5rem;
            padding: 0.8rem 1.2rem;
        }
        
        .fab {
            bottom: 90px;
        }
    }
    
    /* Ultra-wide screens */
    @media (min-width: 1200px) {
        .main .block-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .stats-grid {
            grid-template-columns: repeat(4, 1fr);
        }
        
        .quick-actions {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    
    /* Streamlit overrides */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stSelectbox > div > div {
        border-radius: 15px;
        border: 2px solid #e5e7eb;
    }
    
    .stTextInput > div > div {
        border-radius: 15px;
        border: 2px solid #e5e7eb;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide scrollbar but keep functionality */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #5a6fd8;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application with mobile-like interface."""
    
    # Setup PWA features
    setup_pwa()
    
    # Install button
    st.markdown("""
    <button id="install-app" onclick="installApp()">
        📱 Install App
    </button>
    """, unsafe_allow_html=True)
    
    # App Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">
            <span class="status-indicator"></span>
            ⚽ Quantum Bet
        </h1>
        <p class="app-subtitle">AI-Powered Sports Betting Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'home'
    
    # Main content based on active tab
    if st.session_state.active_tab == 'home':
        render_home_tab()
    elif st.session_state.active_tab == 'chat':
        render_chat_tab()
    elif st.session_state.active_tab == 'predictions':
        render_predictions_tab()
    elif st.session_state.active_tab == 'portfolio':
        render_portfolio_tab()
    elif st.session_state.active_tab == 'settings':
        render_settings_tab()
    
    # Bottom Navigation
    render_bottom_navigation()
    
    # Floating Action Button for quick chat
    if st.session_state.active_tab != 'chat':
        st.markdown("""
        <button class="fab" onclick="document.querySelector('[data-testid=\\\"nav_chat\\\"]').click()">
            💬
        </button>
        """, unsafe_allow_html=True)


def render_bottom_navigation():
    """Render mobile-style bottom navigation."""
    
    tabs = [
        {'key': 'home', 'icon': '🏠', 'label': 'Home'},
        {'key': 'chat', 'icon': '🤖', 'label': 'AI Chat'},
        {'key': 'predictions', 'icon': '🔮', 'label': 'Predictions'},
        {'key': 'portfolio', 'icon': '💰', 'label': 'Portfolio'},
        {'key': 'settings', 'icon': '⚙️', 'label': 'Settings'}
    ]
    
    # Create bottom navigation HTML
    nav_html = '<div class="bottom-nav">'
    
    for tab in tabs:
        active_class = 'active' if st.session_state.active_tab == tab['key'] else ''
        nav_html += f'''
        <div class="nav-item {active_class}">
            <div class="nav-icon">{tab['icon']}</div>
            <div class="nav-label">{tab['label']}</div>
        </div>
        '''
    
    nav_html += '</div>'
    st.markdown(nav_html, unsafe_allow_html=True)
    
    # Hidden Streamlit buttons for functionality
    st.markdown('<div style="display:none;">', unsafe_allow_html=True)
    cols = st.columns(len(tabs))
    for i, tab in enumerate(tabs):
        with cols[i]:
            if st.button(f"{tab['icon']}", key=f"nav_{tab['key']}", help=tab['label']):
                st.session_state.active_tab = tab['key']
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_home_tab():
    """Render home dashboard tab with enhanced mobile design."""
    
    # Welcome message with time-based greeting
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Good morning! ☀️"
    elif 12 <= current_hour < 18:
        greeting = "Good afternoon! 🌤️"
    else:
        greeting = "Good evening! 🌙"
    
    st.markdown(f"### {greeting}")
    st.markdown("*Ready to make some intelligent bets?*")
    
    # Quick stats grid
    col1, col2 = st.columns(2)
    
    with col1:
        render_enhanced_stat_card("12", "Today's Matches", "⚽", "#10b981")
        render_enhanced_stat_card("€2,547", "Portfolio", "💰", "#3b82f6")
    
    with col2:
        render_enhanced_stat_card("8", "High Confidence", "🎯", "#f59e0b")
        render_enhanced_stat_card("+12.3%", "ROI", "📈", "#8b5cf6")
    
    # Quick actions with enhanced design
    st.markdown("### 🚀 Quick Actions")
    
    quick_actions_html = '''
    <div class="quick-actions">
        <div class="quick-action" onclick="document.querySelector('[data-testid=\\\"quick_predictions\\\"]').click()">
            <div class="quick-action-icon">🎯</div>
            <div>Best Predictions</div>
        </div>
        <div class="quick-action" onclick="document.querySelector('[data-testid=\\\"quick_combo\\\"]').click()">
            <div class="quick-action-icon">📊</div>
            <div>Create Combo</div>
        </div>
        <div class="quick-action" onclick="document.querySelector('[data-testid=\\\"quick_value\\\"]').click()">
            <div class="quick-action-icon">💎</div>
            <div>Value Bets</div>
        </div>
        <div class="quick-action" onclick="document.querySelector('[data-testid=\\\"quick_portfolio\\\"]').click()">
            <div class="quick-action-icon">💰</div>
            <div>Portfolio Status</div>
        </div>
    </div>
    '''
    
    st.markdown(quick_actions_html, unsafe_allow_html=True)
    
    # Hidden buttons for functionality
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Predictions", key="quick_predictions", help="Best Predictions"):
            st.session_state.active_tab = 'predictions'
            st.rerun()
    with col2:
        if st.button("Combo", key="quick_combo", help="Create Combo"):
            st.session_state.active_tab = 'chat'
            st.session_state.pending_command = "Create a combo for today"
            st.rerun()
    with col3:
        if st.button("Value", key="quick_value", help="Value Bets"):
            st.session_state.active_tab = 'chat' 
            st.session_state.pending_command = "Find value bets for today"
            st.rerun()
    with col4:
        if st.button("Portfolio", key="quick_portfolio", help="Portfolio"):
            st.session_state.active_tab = 'portfolio'
            st.rerun()
    
    # Today's top predictions with enhanced cards
    st.markdown("### 🔮 Top Predictions")
    
    predictions = get_sample_predictions()
    
    for pred in predictions:
        render_enhanced_prediction_card(pred)
    
    # Mini portfolio chart
    st.markdown("### 📈 Portfolio Trend")
    
    # Create mini chart for mobile
    dates = pd.date_range(start='2024-01-15', end='2024-01-30', freq='D')
    values = 10000 + np.cumsum(np.random.normal(25, 50, len(dates)))
    
    fig = create_mini_portfolio_chart(dates, values)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_enhanced_stat_card(value: str, label: str, icon: str, color: str):
    """Render enhanced stat card with animations."""
    st.markdown(f"""
    <div class="stat-item" style="animation: fadeInUp 0.6s ease-out;">
        <div class="stat-icon">{icon}</div>
        <div class="stat-value" style="color: {color};">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_enhanced_prediction_card(pred: dict):
    """Render enhanced prediction card with mobile optimizations."""
    confidence_level = get_confidence_level(pred['confidence'])
    confidence_color = get_confidence_color(confidence_level)
    
    st.markdown(f"""
    <div class="prediction-card {confidence_level}-confidence swipeable-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
            <div style="flex: 1;">
                <h4 style="margin: 0 0 0.3rem 0; color: #1f2937; font-size: 1.1rem;">{pred['match']}</h4>
                <div style="display: flex; align-items: center; gap: 0.8rem;">
                    <span style="background: rgba(102, 126, 234, 0.1); color: #667eea; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 500;">
                        {pred['league']}
                    </span>
                    <span style="color: #6b7280; font-size: 0.8rem;">
                        {pred['time']}
                    </span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="background: {confidence_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: bold; margin-bottom: 0.3rem;">
                    {pred['confidence']:.1f}%
                </div>
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: bold; font-size: 1rem; color: #1f2937;">{pred['prediction']}</div>
                <div style="color: #6b7280; font-size: 0.8rem;">@{pred['odds']}</div>
            </div>
            
            <div style="display: flex; gap: 0.5rem;">
                <button onclick="addToBetSlip('{pred['match']}')" style="background: {confidence_color}; color: white; border: none; border-radius: 15px; padding: 0.4rem 0.8rem; font-size: 0.8rem; cursor: pointer;">
                    Add to Slip
                </button>
                <button onclick="analyzeMatch('{pred['match']}')" style="background: transparent; color: {confidence_color}; border: 1px solid {confidence_color}; border-radius: 15px; padding: 0.4rem 0.8rem; font-size: 0.8rem; cursor: pointer;">
                    Analyze
                </button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_mini_portfolio_chart(dates, values):
    """Create mini portfolio chart optimized for mobile."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='<b>%{y:€,.0f}</b><br>%{x}<extra></extra>'
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False
        ),
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


def get_sample_predictions():
    """Get sample predictions for demo."""
    return [
        {
            'match': 'Manchester City vs Arsenal',
            'prediction': 'Man City Win',
            'confidence': 87.5,
            'odds': 1.75,
            'league': 'Premier League',
            'time': '15:30'
        },
        {
            'match': 'Barcelona vs Real Madrid',
            'prediction': 'Over 2.5 Goals',
            'confidence': 82.1,
            'odds': 1.85,
            'league': 'La Liga',
            'time': '21:00'
        },
        {
            'match': 'PSG vs Lyon',
            'prediction': 'PSG Win',
            'confidence': 76.8,
            'odds': 1.45,
            'league': 'Ligue 1',
            'time': '20:45'
        }
    ]


def get_confidence_level(confidence):
    """Get confidence level string."""
    if confidence > 80:
        return "high"
    elif confidence > 65:
        return "medium"
    else:
        return "low"


def get_confidence_color(level):
    """Get color for confidence level."""
    colors = {
        "high": "#10b981",
        "medium": "#f59e0b", 
        "low": "#ef4444"
    }
    return colors.get(level, "#6b7280")


# JavaScript for PWA functionality
st.markdown("""
<script>
function addToBetSlip(match) {
    // Add to bet slip functionality
    alert('Added ' + match + ' to bet slip!');
}

function analyzeMatch(match) {
    // Navigate to analysis
    document.querySelector('[data-testid="nav_chat"]').click();
    setTimeout(() => {
        const chatInput = document.querySelector('.stChatInput input');
        if (chatInput) {
            chatInput.value = 'Analyze the match ' + match;
        }
    }, 500);
}

// Swipe gestures for mobile
let startX, startY, endX, endY;

document.addEventListener('touchstart', function(e) {
    startX = e.touches[0].clientX;
    startY = e.touches[0].clientY;
});

document.addEventListener('touchend', function(e) {
    endX = e.changedTouches[0].clientX;
    endY = e.changedTouches[0].clientY;
    
    const diffX = startX - endX;
    const diffY = startY - endY;
    
    // Horizontal swipe
    if (Math.abs(diffX) > Math.abs(diffY) && Math.abs(diffX) > 50) {
        if (diffX > 0) {
            // Swipe left - next tab
            const tabs = ['home', 'chat', 'predictions', 'portfolio', 'settings'];
            const currentIndex = tabs.indexOf(window.activeTab || 'home');
            if (currentIndex < tabs.length - 1) {
                document.querySelector(`[data-testid="nav_${tabs[currentIndex + 1]}"]`).click();
            }
        } else {
            // Swipe right - previous tab
            const tabs = ['home', 'chat', 'predictions', 'portfolio', 'settings'];
            const currentIndex = tabs.indexOf(window.activeTab || 'home');
            if (currentIndex > 0) {
                document.querySelector(`[data-testid="nav_${tabs[currentIndex - 1]}"]`).click();
            }
        }
    }
});

// Store current tab
window.activeTab = 'home';

// Update active tab when navigation changes
document.addEventListener('click', function(e) {
    if (e.target.dataset.testid && e.target.dataset.testid.startsWith('nav_')) {
        window.activeTab = e.target.dataset.testid.replace('nav_', '');
    }
});
</script>
""", unsafe_allow_html=True)
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border-radius: 20px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 6px 25px rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 50px;
        height: 50px;
        background: radial-gradient(circle, rgba(102,126,234,0.1) 0%, transparent 70%);
    }
    
    .high-confidence {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
    }
    
    .medium-confidence {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }
    
    .low-confidence {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    }
    
    /* Chat interface */
    .chat-container {
        background: white;
        border-radius: 25px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
    }
    
    .chat-message {
        margin: 1rem 0;
        animation: slideIn 0.3s ease-out;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin-left: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .bot-message {
        background: #f8f9fa;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin-right: 2rem;
        border: 1px solid #e9ecef;
    }
    
    /* Buttons */
    .action-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 1.5rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .quick-action {
        background: white;
        border: 2px solid #667eea;
        color: #667eea;
        border-radius: 25px;
        padding: 0.6rem 1.2rem;
        margin: 0.3rem;
        font-size: 0.85rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-block;
    }
    
    .quick-action:hover {
        background: #667eea;
        color: white;
        transform: scale(1.05);
    }
    
    /* Statistics */
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 0.2rem;
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .metric-card, .prediction-card, .chat-container {
            background: #1f2937;
            color: white;
            border-color: #374151;
        }
        
        .stat-item {
            background: #1f2937;
            color: white;
        }
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .stat-grid {
            grid-template-columns: 1fr;
        }
        
        .nav-tab {
            font-size: 0.7rem;
            padding: 0.6rem;
        }
        
        .app-title {
            font-size: 1.5rem;
        }
    }
    
    /* Hide Streamlit elements */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stSelectbox > div > div {
        border-radius: 15px;
        border: 2px solid #e5e7eb;
    }
    
    .stTextInput > div > div {
        border-radius: 15px;
        border: 2px solid #e5e7eb;
    }
    
    /* Chat input styling */
    .stChatInput {
        border-radius: 25px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application with mobile-like interface."""
    
    # App Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">⚽ Quantum Bet</h1>
        <p class="app-subtitle">AI-Powered Sports Betting Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'home'
    
    # Navigation
    render_navigation()
    
    # Main content based on active tab
    if st.session_state.active_tab == 'home':
        render_home_tab()
    elif st.session_state.active_tab == 'chat':
        render_chat_tab()
    elif st.session_state.active_tab == 'predictions':
        render_predictions_tab()
    elif st.session_state.active_tab == 'portfolio':
        render_portfolio_tab()
    elif st.session_state.active_tab == 'settings':
        render_settings_tab()


def render_navigation():
    """Render mobile-like navigation tabs."""
    
    tabs = [
        {'key': 'home', 'icon': '🏠', 'label': 'Home'},
        {'key': 'chat', 'icon': '🤖', 'label': 'AI Chat'},
        {'key': 'predictions', 'icon': '🔮', 'label': 'Predictions'},
        {'key': 'portfolio', 'icon': '💰', 'label': 'Portfolio'},
        {'key': 'settings', 'icon': '⚙️', 'label': 'Settings'}
    ]
    
    # Create navigation HTML
    nav_html = '<div class="nav-container">'
    
    for tab in tabs:
        active_class = 'active' if st.session_state.active_tab == tab['key'] else ''
        nav_html += f'''
        <div class="nav-tab {active_class}" onclick="setActiveTab('{tab['key']}')">
            <div>{tab['icon']}</div>
            <div>{tab['label']}</div>
        </div>
        '''
    
    nav_html += '</div>'
    
    # JavaScript for tab switching
    nav_html += '''
    <script>
    function setActiveTab(tabKey) {
        // This would normally update the session state
        // For now, we'll use Streamlit buttons
    }
    </script>
    '''
    
    st.markdown(nav_html, unsafe_allow_html=True)
    
    # Streamlit buttons for actual functionality (hidden with CSS)
    cols = st.columns(len(tabs))
    for i, tab in enumerate(tabs):
        with cols[i]:
            if st.button(f"{tab['icon']}", key=f"nav_{tab['key']}", help=tab['label']):
                st.session_state.active_tab = tab['key']
                st.rerun()


def render_home_tab():
    """Render home dashboard tab."""
    
    # Quick stats
    st.markdown("### 📊 Today's Overview")
    
    # Mock data - would be replaced with real data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_stat_card("12", "Today's Matches", "⚽", "#10b981")
    with col2:
        render_stat_card("8", "High Confidence", "🎯", "#f59e0b")
    with col3:
        render_stat_card("€2,547", "Portfolio", "💰", "#3b82f6")
    with col4:
        render_stat_card("+12.3%", "ROI", "📈", "#8b5cf6")
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    
    quick_actions = [
        "🎯 Best Predictions Today",
        "💎 Value Bets",
        "📊 Create Combo",
        "🔍 Analyze Match",
        "💰 Portfolio Status"
    ]
    
    action_html = '<div style="text-align: center; margin: 1rem 0;">'
    for action in quick_actions:
        action_html += f'<span class="quick-action">{action}</span>'
    action_html += '</div>'
    
    st.markdown(action_html, unsafe_allow_html=True)
    
    # Today's top predictions
    st.markdown("### 🔮 Top Predictions")
    
    # Mock predictions
    predictions = [
        {
            'match': 'Manchester City vs Arsenal',
            'prediction': 'Man City Win',
            'confidence': 87.5,
            'odds': 1.75,
            'league': 'Premier League',
            'time': '15:30'
        },
        {
            'match': 'Barcelona vs Real Madrid',
            'prediction': 'Over 2.5 Goals',
            'confidence': 82.1,
            'odds': 1.85,
            'league': 'La Liga',
            'time': '21:00'
        },
        {
            'match': 'PSG vs Lyon',
            'prediction': 'PSG Win',
            'confidence': 76.8,
            'odds': 1.45,
            'league': 'Ligue 1',
            'time': '20:45'
        }
    ]
    
    for pred in predictions:
        render_prediction_card(pred)
    
    # Portfolio chart
    st.markdown("### 📈 Portfolio Performance")
    
    # Mock portfolio data
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    portfolio_values = 10000 + np.cumsum(np.random.normal(50, 100, len(dates)))
    
    fig = create_portfolio_chart(dates, portfolio_values)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_chat_tab():
    """Render AI chat interface."""
    
    st.markdown("### 🤖 AI Assistant")
    st.markdown("*Command your betting strategy with natural language*")
    
    # Quick chat actions
    st.markdown("**Quick Commands:**")
    
    quick_commands = [
        "Create a combo for today",
        "Show me top predictions",
        "Portfolio status",
        "Find value bets",
        "Analyze Liverpool vs Chelsea"
    ]
    
    cols = st.columns(2)
    for i, cmd in enumerate(quick_commands):
        with cols[i % 2]:
            if st.button(f"💬 {cmd}", key=f"quick_cmd_{i}", use_container_width=True):
                process_quick_command(cmd)
    
    # Chat interface
    st.markdown("---")
    
    # Chat history
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat history
    for message in st.session_state.chat_messages:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message">
                <div class="user-message">
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message">
                <div class="bot-message">
                    🤖 {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Type your command here...", key="main_chat"):
        # Add user message
        st.session_state.chat_messages.append({
            'role': 'user',
            'content': prompt
        })
        
        # Process with AI (mock response for demo)
        bot_response = process_chat_command(prompt)
        st.session_state.chat_messages.append({
            'role': 'assistant',
            'content': bot_response
        })
        
        st.rerun()


def render_predictions_tab():
    """Render predictions tab."""
    
    st.markdown("### 🔮 Match Predictions")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        league_filter = st.selectbox(
            "League", 
            ["All", "Premier League", "La Liga", "Ligue 1", "Bundesliga", "Serie A"],
            key="pred_league"
        )
    
    with col2:
        confidence_filter = st.slider(
            "Min Confidence", 
            0.5, 1.0, 0.7, 0.05,
            key="pred_confidence"
        )
    
    with col3:
        date_filter = st.selectbox(
            "Period",
            ["Today", "Tomorrow", "This Week"],
            key="pred_date"
        )
    
    # Predictions list
    st.markdown("---")
    
    # Mock predictions with filtering
    all_predictions = [
        {'match': 'Liverpool vs Arsenal', 'prediction': 'Liverpool Win', 'confidence': 0.875, 'league': 'Premier League'},
        {'match': 'Barcelona vs Madrid', 'prediction': 'Over 2.5', 'confidence': 0.821, 'league': 'La Liga'},
        {'match': 'PSG vs Lyon', 'prediction': 'PSG Win', 'confidence': 0.768, 'league': 'Ligue 1'},
        {'match': 'Bayern vs Dortmund', 'prediction': 'Bayern Win', 'confidence': 0.692, 'league': 'Bundesliga'},
        {'match': 'Juventus vs Milan', 'prediction': 'BTTS Yes', 'confidence': 0.654, 'league': 'Serie A'},
    ]
    
    # Filter predictions
    filtered_preds = [p for p in all_predictions if p['confidence'] >= confidence_filter]
    if league_filter != "All":
        filtered_preds = [p for p in filtered_preds if p['league'] == league_filter]
    
    for pred in filtered_preds:
        render_detailed_prediction_card(pred)
    
    if not filtered_preds:
        st.info("No predictions match your criteria. Try adjusting the filters.")


def render_portfolio_tab():
    """Render portfolio tab."""
    
    st.markdown("### 💰 Portfolio Dashboard")
    
    # Portfolio metrics
    col1, col2 = st.columns(2)
    
    with col1:
        render_stat_card("€12,547", "Total Bankroll", "💰", "#10b981")
        render_stat_card("+€1,247", "Profit/Loss", "📈", "#10b981")
    
    with col2:
        render_stat_card("+12.3%", "ROI", "📊", "#3b82f6")
        render_stat_card("67.8%", "Win Rate", "🎯", "#8b5cf6")
    
    # Performance chart
    st.markdown("#### 📈 Performance Over Time")
    
    # Mock portfolio data
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
    portfolio_values = 10000 + np.cumsum(np.random.normal(25, 75, len(dates)))
    
    fig = create_detailed_portfolio_chart(dates, portfolio_values)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Recent bets
    st.markdown("#### 🎲 Recent Bets")
    
    recent_bets = [
        {'match': 'Man City vs Arsenal', 'bet': 'Man City Win', 'stake': '€150', 'status': 'Won', 'profit': '+€112.50'},
        {'match': 'Barcelona vs Madrid', 'bet': 'Over 2.5', 'stake': '€200', 'status': 'Lost', 'profit': '-€200.00'},
        {'match': 'PSG vs Lyon', 'bet': 'PSG -1', 'stake': '€100', 'status': 'Won', 'profit': '+€85.00'},
    ]
    
    for bet in recent_bets:
        render_bet_card(bet)


def render_settings_tab():
    """Render settings tab."""
    
    st.markdown("### ⚙️ Settings")
    
    # Model settings
    with st.expander("🤖 Model Configuration", expanded=True):
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.5, 1.0, 0.7, 0.05,
            help="Minimum confidence for high-confidence predictions"
        )
        
        min_edge = st.slider(
            "Minimum Edge",
            0.01, 0.2, 0.05, 0.01,
            help="Minimum expected edge for betting opportunities"
        )
        
        max_kelly = st.slider(
            "Max Kelly Fraction",
            0.1, 0.5, 0.25, 0.05,
            help="Maximum fraction of bankroll to bet"
        )
    
    # Notification settings
    with st.expander("🔔 Notifications"):
        email_alerts = st.checkbox("Email Alerts", value=True)
        telegram_alerts = st.checkbox("Telegram Alerts", value=True)
        push_notifications = st.checkbox("Push Notifications", value=False)
    
    # Display preferences
    with st.expander("🎨 Display"):
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        language = st.selectbox("Language", ["English", "Français", "Español"])
    
    # Save button
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("✅ Settings saved successfully!")


# Helper functions for rendering components

def render_stat_card(value: str, label: str, icon: str, color: str):
    """Render a stat card."""
    st.markdown(f"""
    <div class="metric-card" style="text-align: center;">
        <div style="font-size: 2rem;">{icon}</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {color}; margin: 0.5rem 0;">{value}</div>
        <div style="color: #6b7280; font-size: 0.9rem;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_prediction_card(pred: dict):
    """Render a prediction card."""
    confidence_level = "high" if pred['confidence'] > 80 else "medium" if pred['confidence'] > 65 else "low"
    confidence_color = "#10b981" if confidence_level == "high" else "#f59e0b" if confidence_level == "medium" else "#ef4444"
    
    st.markdown(f"""
    <div class="prediction-card {confidence_level}-confidence">
        <div style="display: flex; justify-content: between; align-items: center;">
            <div style="flex: 1;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1f2937;">{pred['match']}</h4>
                <p style="margin: 0; color: #6b7280; font-size: 0.85rem;">
                    {pred['league']} • {pred['time']}
                </p>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: bold; color: {confidence_color};">{pred['prediction']}</div>
                <div style="font-size: 0.85rem; color: #6b7280;">
                    {pred['confidence']:.1f}% • @{pred['odds']}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_detailed_prediction_card(pred: dict):
    """Render a detailed prediction card."""
    confidence_pct = pred['confidence'] * 100
    confidence_level = "high" if confidence_pct > 80 else "medium" if confidence_pct > 65 else "low"
    
    with st.container():
        st.markdown(f"""
        <div class="prediction-card {confidence_level}-confidence">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: #1f2937;">{pred['match']}</h4>
                <span style="background: rgba(102, 126, 234, 0.1); color: #667eea; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem;">
                    {pred['league']}
                </span>
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-weight: bold; font-size: 1.1rem; color: #1f2937;">{pred['prediction']}</div>
                    <div style="color: #6b7280; font-size: 0.9rem;">Prediction</div>
                </div>
                
                <div style="text-align: center;">
                    <div style="font-weight: bold; font-size: 1.1rem; color: #667eea;">{confidence_pct:.1f}%</div>
                    <div style="color: #6b7280; font-size: 0.9rem;">Confidence</div>
                </div>
                
                <div style="text-align: right;">
                    <button class="action-btn" style="font-size: 0.9rem; padding: 0.5rem 1rem;">
                        Add to Slip
                    </button>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_bet_card(bet: dict):
    """Render a bet card."""
    status_color = "#10b981" if bet['status'] == 'Won' else "#ef4444" if bet['status'] == 'Lost' else "#f59e0b"
    status_icon = "✅" if bet['status'] == 'Won' else "❌" if bet['status'] == 'Lost' else "⏳"
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="flex: 1;">
                <h5 style="margin: 0 0 0.3rem 0;">{bet['match']}</h5>
                <p style="margin: 0; color: #6b7280; font-size: 0.85rem;">{bet['bet']} • {bet['stake']}</p>
            </div>
            <div style="text-align: right;">
                <div style="color: {status_color}; font-weight: bold;">
                    {status_icon} {bet['status']}
                </div>
                <div style="color: {status_color}; font-weight: bold;">
                    {bet['profit']}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_portfolio_chart(dates, values):
    """Create portfolio