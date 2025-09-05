-- Bot Quantum Max Database Schema

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;

-- Teams table
CREATE TABLE teams (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_id INTEGER UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    short_name VARCHAR(10),
    country VARCHAR(50) NOT NULL,
    league VARCHAR(10) NOT NULL,
    founded INTEGER,
    venue VARCHAR(100),
    logo_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Players table
CREATE TABLE players (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_id INTEGER UNIQUE NOT NULL,
    team_id UUID REFERENCES teams(id),
    name VARCHAR(100) NOT NULL,
    position VARCHAR(20),
    age INTEGER,
    nationality VARCHAR(50),
    height INTEGER, -- in cm
    weight INTEGER, -- in kg
    market_value BIGINT, -- in euros
    contract_until DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Matches table
CREATE TABLE matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_id INTEGER UNIQUE NOT NULL,
    home_team_id UUID REFERENCES teams(id) NOT NULL,
    away_team_id UUID REFERENCES teams(id) NOT NULL,
    league VARCHAR(10) NOT NULL,
    season VARCHAR(10) NOT NULL,
    matchday INTEGER,
    match_date TIMESTAMP NOT NULL,
    status VARCHAR(20) DEFAULT 'SCHEDULED',
    home_score INTEGER,
    away_score INTEGER,
    ht_home_score INTEGER, -- half-time scores
    ht_away_score INTEGER,
    referee VARCHAR(100),
    venue VARCHAR(100),
    attendance INTEGER,
    weather_conditions JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Match events table
CREATE TABLE match_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) NOT NULL,
    player_id UUID REFERENCES players(id),
    team_id UUID REFERENCES teams(id) NOT NULL,
    event_type VARCHAR(30) NOT NULL, -- GOAL, CARD, SUBSTITUTION, etc.
    minute INTEGER NOT NULL,
    extra_time INTEGER DEFAULT 0,
    details JSONB, -- Additional event-specific data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Match statistics table
CREATE TABLE match_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) NOT NULL,
    team_id UUID REFERENCES teams(id) NOT NULL,
    possession DECIMAL(5,2),
    shots_total INTEGER,
    shots_on_target INTEGER,
    shots_off_target INTEGER,
    shots_blocked INTEGER,
    corners INTEGER,
    fouls INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    offsides INTEGER,
    passes_total INTEGER,
    passes_accurate INTEGER,
    pass_accuracy DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Advanced statistics (xG, PPDA, etc.)
CREATE TABLE advanced_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) NOT NULL,
    team_id UUID REFERENCES teams(id) NOT NULL,
    xg DECIMAL(4,2), -- Expected Goals
    xga DECIMAL(4,2), -- Expected Goals Against
    npxg DECIMAL(4,2), -- Non-Penalty xG
    ppda DECIMAL(6,2), -- Passes Per Defensive Action
    deep_completions INTEGER,
    xa DECIMAL(4,2), -- Expected Assists
    key_passes INTEGER,
    progressive_passes INTEGER,
    final_third_entries INTEGER,
    penalty_area_entries INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Player match performance
CREATE TABLE player_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) NOT NULL,
    player_id UUID REFERENCES players(id) NOT NULL,
    team_id UUID REFERENCES teams(id) NOT NULL,
    minutes_played INTEGER,
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    shots INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    key_passes INTEGER DEFAULT 0,
    passes INTEGER DEFAULT 0,
    pass_accuracy DECIMAL(5,2),
    crosses INTEGER DEFAULT 0,
    touches INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    tackles INTEGER DEFAULT 0,
    clearances INTEGER DEFAULT 0,
    duels_won INTEGER DEFAULT 0,
    duels_lost INTEGER DEFAULT 0,
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    rating DECIMAL(3,1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Features table (engineered features for ML)
CREATE TABLE features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) NOT NULL,
    team_id UUID REFERENCES teams(id) NOT NULL,
    feature_date DATE NOT NULL,
    -- Form features
    form_5_games DECIMAL(3,2),
    form_10_games DECIMAL(3,2),
    home_form DECIMAL(3,2),
    away_form DECIMAL(3,2),
    -- Goal features
    goals_scored_avg DECIMAL(4,2),
    goals_conceded_avg DECIMAL(4,2),
    goals_scored_home_avg DECIMAL(4,2),
    goals_conceded_away_avg DECIMAL(4,2),
    -- xG features
    xg_avg DECIMAL(4,2),
    xga_avg DECIMAL(4,2),
    xg_home_avg DECIMAL(4,2),
    xga_away_avg DECIMAL(4,2),
    -- Tactical features
    possession_avg DECIMAL(5,2),
    shots_avg DECIMAL(5,2),
    shots_on_target_avg DECIMAL(5,2),
    pass_accuracy_avg DECIMAL(5,2),
    crosses_avg DECIMAL(5,2),
    corners_avg DECIMAL(5,2),
    -- Defensive features
    tackles_avg DECIMAL(5,2),
    interceptions_avg DECIMAL(5,2),
    clearances_avg DECIMAL(5,2),
    fouls_avg DECIMAL(5,2),
    cards_avg DECIMAL(5,2),
    -- Head-to-head
    h2h_wins INTEGER,
    h2h_draws INTEGER,
    h2h_losses INTEGER,
    h2h_goals_for DECIMAL(4,2),
    h2h_goals_against DECIMAL(4,2),
    -- Context features
    days_since_last_match INTEGER,
    is_european_competition BOOLEAN,
    injury_count INTEGER,
    suspension_count INTEGER,
    market_value_total BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions table
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),
    prediction_date TIMESTAMP NOT NULL,
    -- Basic predictions
    home_win_prob DECIMAL(5,4),
    draw_prob DECIMAL(5,4),
    away_win_prob DECIMAL(5,4),
    -- Over/Under predictions
    over_1_5_prob DECIMAL(5,4),
    over_2_5_prob DECIMAL(5,4),
    over_3_5_prob DECIMAL(5,4),
    under_1_5_prob DECIMAL(5,4),
    under_2_5_prob DECIMAL(5,4),
    under_3_5_prob DECIMAL(5,4),
    -- BTTS predictions
    btts_yes_prob DECIMAL(5,4),
    btts_no_prob DECIMAL(5,4),
    -- Score predictions
    exact_score_probs JSONB, -- JSON object with score predictions
    -- Confidence metrics
    confidence_score DECIMAL(5,4),
    model_uncertainty DECIMAL(5,4),
    feature_importance JSONB,
    -- Simulation results
    monte_carlo_runs INTEGER,
    simulation_results JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Odds table
CREATE TABLE odds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    odds_date TIMESTAMP NOT NULL,
    market_type VARCHAR(30) NOT NULL, -- 1X2, OVER_UNDER, BTTS, etc.
    odds_data JSONB NOT NULL, -- Flexible structure for different bet types
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Betting tickets table
CREATE TABLE betting_tickets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticket_type VARCHAR(20) NOT NULL, -- SINGLE, COMBO, SYSTEM
    total_stake DECIMAL(10,2) NOT NULL,
    potential_return DECIMAL(10,2) NOT NULL,
    total_odds DECIMAL(8,3) NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING', -- PENDING, WON, LOST, VOID
    placed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settled_date TIMESTAMP,
    actual_return DECIMAL(10,2),
    profit_loss DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual bets within tickets
CREATE TABLE bets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticket_id UUID REFERENCES betting_tickets(id) NOT NULL,
    match_id UUID REFERENCES matches(id) NOT NULL,
    prediction_id UUID REFERENCES predictions(id),
    bet_type VARCHAR(30) NOT NULL,
    selection VARCHAR(100) NOT NULL, -- e.g., "Manchester United", "Over 2.5", etc.
    odds DECIMAL(8,3) NOT NULL,
    stake DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING',
    edge DECIMAL(5,4), -- Expected edge over bookmaker
    kelly_fraction DECIMAL(5,4), -- Kelly criterion fraction
    confidence DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Portfolio performance tracking
CREATE TABLE portfolio_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    total_bankroll DECIMAL(12,2) NOT NULL,
    daily_pnl DECIMAL(10,2),
    total_pnl DECIMAL(12,2),
    roi DECIMAL(6,4),
    sharpe_ratio DECIMAL(6,4),
    max_drawdown DECIMAL(6,4),
    win_rate DECIMAL(5,4),
    avg_odds DECIMAL(8,3),
    total_bets INTEGER,
    winning_bets INTEGER,
    losing_bets INTEGER,
    void_bets INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance tracking
CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(50) NOT NULL,
    evaluation_date DATE NOT NULL,
    dataset_type VARCHAR(20) NOT NULL, -- TRAIN, VALIDATION, TEST
    accuracy DECIMAL(6,4),
    precision DECIMAL(6,4),
    recall DECIMAL(6,4),
    f1_score DECIMAL(6,4),
    auc_roc DECIMAL(6,4),
    log_loss DECIMAL(8,6),
    brier_score DECIMAL(8,6),
    calibration_error DECIMAL(8,6),
    profit_loss DECIMAL(10,2),
    roi DECIMAL(6,4),
    sample_size INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System logs
CREATE TABLE system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level VARCHAR(10) NOT NULL,
    module VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    error_traceback TEXT
);

-- Create hypertables for time-series data (TimescaleDB)
SELECT create_hypertable('matches', 'match_date', chunk_time_interval => INTERVAL '1 month');
SELECT create_hypertable('predictions', 'prediction_date', chunk_time_interval => INTERVAL '1 month');
SELECT create_hypertable('odds', 'odds_date', chunk_time_interval => INTERVAL '1 week');
SELECT create_hypertable('portfolio_performance', 'date', chunk_time_interval => INTERVAL '1 month');
SELECT create_hypertable('system_logs', 'timestamp', chunk_time_interval => INTERVAL '1 week');