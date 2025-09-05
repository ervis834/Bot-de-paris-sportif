### Environment Variables
```env
# Database
DATABASE_URL=postgresql://quantum:password@localhost:5432/quantum_bot

# API Keys
FOOTBALL_DATA_API_KEY=your_football_data_key
OPENWEATHER_API_KEY=your_weather_key
ODDS_API_KEY=your_odds_key
OPENAI_API_KEY=your_openai_key_for_chatbot

# Alerts (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_token
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Model Parameters
MAX_MONTE_CARLO_RUNS=10000
CONFIDENCE_THRESHOLD=0.65
MIN_EDGE=0.05
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 12+
- OpenAI API Key (for chatbot)
- Git

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/your-org/bot-quantum-max.git
cd bot-quantum-max
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup Database**
```bash
# Create PostgreSQL database
createdb quantum_bot

# Run schema migration
psql quantum_bot < sql/schema.sql
psql quantum_bot < sql/indices.sql
```

4. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your API keys and database credentials
```

5. **Train Models**
```bash
python train.py
```

6. **Generate Predictions**
```bash
python predict_today.py
```

7. **Launch Dashboard with AI Assistant**
```bash
streamlit run src/ui/app.py
```

## 🤖 AI Assistant Features

### Natural Language Interface
The AI chatbot transforms complex betting operations into simple conversations:

**Traditional Approach:**
1. Navigate to predictions page
2. Apply filters for confidence > 70%
3. Select matches manually
4. Go to combinations page
5. Configure combo parameters
6. Generate combination

**With AI Assistant:**
Simply type: *"Crée un combiné avec les 3 meilleures prédictions d'aujourd'hui"*

### Advanced Command Processing
The chatbot uses GPT-4 to understand context and intent:

```python
# These all create the same type of combination:
"Combiné conservateur 3 matchs Premier League"
"Fais-moi un pari combiné sûr avec des matchs anglais"
"Je veux un triple sécurisé sur la PL"
```

### Smart Parameter Extraction
The AI automatically detects and processes:
- **Numerical criteria**: "cotes > 1.5", "confiance 70%", "3 matchs"
- **Date references**: "aujourd'hui", "demain", "ce weekend", "la semaine prochaine"
- **League names**: "Premier League", "PL", "Ligue 1", "Champions League"
- **Risk preferences**: "conservateur", "agressif", "équilibré"

### Interactive Clarification
When commands are ambiguous, the AI asks for clarification:

**User:** *"Crée un combiné"*
**AI:** *"Je peux créer un combiné pour vous ! Pouvez-vous préciser :
- Combien de matchs souhaitez-vous ?
- Pour quelle période (aujourd'hui, demain, weekend) ?
- Type de combiné souhaité (conservateur, équilibré, agressif) ?"*# ⚽ Bot Quantum Max

**AI-Powered Football Betting Intelligence Platform**

Bot Quantum Max is an advanced machine learning system that combines multiple AI techniques to predict football match outcomes and optimize betting strategies. The system integrates Graph Neural Networks, Bayesian inference, causal modeling, reinforcement learning, and Monte Carlo simulations to deliver superior prediction accuracy and portfolio optimization.

## 🎯 Key Features

### 🧠 Advanced AI Models
- **Supervised Learning**: XGBoost, LightGBM, Random Forest ensemble
- **Graph Neural Networks**: Player interaction and tactical analysis
- **Bayesian Models**: Dynamic probability updating with uncertainty quantification
- **Causal Inference**: DoWhy/CausalML for understanding tactical impact
- **Reinforcement Learning**: Dynamic bankroll management and bet sizing

### 📊 Comprehensive Data Pipeline
- **Multi-Source ETL**: Football-Data.org, Understat (xG), OpenWeather, Odds APIs
- **Advanced Features**: xG, PPDA, tactical graphs, form analysis
- **Real-time Updates**: Automated daily data collection and processing
- **PostgreSQL + TimescaleDB**: Optimized for time-series sports data

### 🎰 Portfolio Optimization
- **Kelly Criterion**: Optimal bet sizing with risk management
- **Modern Portfolio Theory**: Markowitz optimization for betting portfolios
- **Correlation Analysis**: Smart combination generation avoiding over-correlation
- **Risk Management**: VaR, drawdown control, position sizing limits

### 🎯 Betting Intelligence
- **Monte Carlo Simulations**: 10k+ runs per match for robust probability estimation
- **Agent-Based Modeling**: 22-player tactical simulations
- **Combination Generator**: Accumulators, systems, Yankees with optimization
- **Live Odds Integration**: Real-time odds comparison and arbitrage detection

### 📈 Interactive Dashboard
- **Streamlit Interface**: Real-time predictions and analytics
- **Model Explainability**: SHAP values and feature importance
- **Portfolio Tracking**: Performance metrics and risk analysis
- **Alert System**: Email/Telegram notifications for opportunities

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 12+
- Git

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/your-org/bot-quantum-max.git
cd bot-quantum-max
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup Database**
```bash
# Create PostgreSQL database
createdb quantum_bot

# Run schema migration
psql quantum_bot < sql/schema.sql
psql quantum_bot < sql/indices.sql
```

4. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your API keys and database credentials
```

5. **Train Models**
```bash
python train.py
```

6. **Generate Predictions**
```bash
python predict_today.py
```

7. **Launch Dashboard**
```bash
streamlit run src/ui/app.py
```

## 📋 Configuration

### Required API Keys
- **Football-Data.org**: Match data and statistics
- **OpenWeather**: Weather conditions for match venues
- **Odds API**: Real-time betting odds from multiple bookmakers
- **Understat** (optional): Advanced xG and shot data

### Environment Variables
```env
# Database
DATABASE_URL=postgresql://quantum:password@localhost:5432/quantum_bot

# API Keys
FOOTBALL_DATA_API_KEY=your_football_data_key
OPENWEATHER_API_KEY=your_weather_key
ODDS_API_KEY=your_odds_key

# Alerts (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_token
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Model Parameters
MAX_MONTE_CARLO_RUNS=10000
CONFIDENCE_THRESHOLD=0.65
MIN_EDGE=0.05
```

## 🏗️ Architecture

### Data Flow
```
APIs (Football-Data, Understat, Weather, Odds)
    ↓
ETL Pipeline (src/data/etl.py)
    ↓
PostgreSQL Database
    ↓
Feature Engineering (src/features/engineering.py)
    ↓
ML Models (src/models/)
    ↓
Predictions & Simulations (src/sim/)
    ↓
Portfolio Optimization (src/portfolio/)
    ↓
Dashboard (src/ui/app.py)
```

### Core Components

#### 📁 Data Layer (`src/data/`)
- `database.py`: PostgreSQL connection and query management
- `etl.py`: Extract, Transform, Load pipeline orchestration
- `collectors/`: API-specific data collection modules

#### 🤖 Models (`src/models/`)
- `supervised.py`: Traditional ML models (XGBoost, LightGBM, RF)
- `gnn.py`: Graph Neural Network implementation
- `bayesian.py`: Bayesian inference models
- `causal.py`: Causal inference pipeline
- `rl.py`: Reinforcement learning for bankroll management

#### 📊 Features (`src/features/`)
- `engineering.py`: Feature extraction and transformation
- `tactical.py`: Tactical analysis features
- `graph_features.py`: Graph-based feature engineering

#### 🎲 Simulation (`src/sim/`)
- `monte_carlo.py`: Monte Carlo match simulation
- `agent_based.py`: Agent-based tactical modeling

#### 💰 Portfolio (`src/portfolio/`)
- `optimizer.py`: Portfolio optimization and Kelly Criterion
- `combos.py`: Betting combination generation
- `kelly.py`: Advanced Kelly Criterion implementations

#### 🖥️ UI (`src/ui/`)
- `app.py`: Streamlit dashboard application

## 🎮 Usage

### Training Models
```bash
# Train all models
python train.py

# Train specific models only
python train.py --models supervised ensemble

# Skip ETL (use existing data)
python train.py --skip-etl
```

### Generating Predictions
```bash
# Generate predictions for today's matches
python predict_today.py

# Skip Monte Carlo simulation (faster)
python predict_today.py --skip-simulation

# Save results only (no console output)
python predict_today.py --save-only
```

### Dashboard Features
- **🏠 Dashboard**: Overview of today's matches and key metrics
- **🔮 Predictions**: Detailed match predictions with filtering
- **📊 Analytics**: Model performance and accuracy tracking
- **💰 Portfolio**: Bankroll management and performance analysis
- **🎯 Combinations**: Optimized betting combinations
- **⚙️ Settings