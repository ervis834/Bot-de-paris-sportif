-- Indices for Bot Quantum Max Database

-- Teams indices
CREATE INDEX idx_teams_api_id ON teams(api_id);
CREATE INDEX idx_teams_league ON teams(league);
CREATE INDEX idx_teams_country ON teams(country);

-- Players indices
CREATE INDEX idx_players_api_id ON players(api_id);
CREATE INDEX idx_players_team_id ON players(team_id);
CREATE INDEX idx_players_position ON players(position);
CREATE INDEX idx_players_name ON players(name);

-- Matches indices
CREATE INDEX idx_matches_api_id ON matches(api_id);
CREATE INDEX idx_matches_home_team ON matches(home_team_id);
CREATE INDEX idx_matches_away_team ON matches(away_team_id);
CREATE INDEX idx_matches_date ON matches(match_date);
CREATE INDEX idx_matches_league ON matches(league);
CREATE INDEX idx_matches_season ON matches(season);
CREATE INDEX idx_matches_status ON matches(status);
CREATE INDEX idx_matches_league_season ON matches(league, season);
CREATE INDEX idx_matches_date_league ON matches(match_date, league);

-- Match events indices
CREATE INDEX idx_match_events_match_id ON match_events(match_id);
CREATE INDEX idx_match_events_player_id ON match_events(player_id);
CREATE INDEX idx_match_events_team_id ON match_events(team_id);
CREATE INDEX idx_match_events_type ON match_events(event_type);
CREATE INDEX idx_match_events_minute ON match_events(minute);

-- Match stats indices
CREATE INDEX idx_match_stats_match_id ON match_stats(match_id);
CREATE INDEX idx_match_stats_team_id ON match_stats(team_id);

-- Advanced stats indices
CREATE INDEX idx_advanced_stats_match_id ON advanced_stats(match_id);
CREATE INDEX idx_advanced_stats_team_id ON advanced_stats(team_id);

-- Player stats indices
CREATE INDEX idx_player_stats_match_id ON player_stats(match_id);
CREATE INDEX idx_player_stats_player_id ON player_stats(player_id);
CREATE INDEX idx_player_stats_team_id ON player_stats(team_id);

-- Features indices
CREATE INDEX idx_features_match_id ON features(match_id);
CREATE INDEX idx_features_team_id ON features(team_id);
CREATE INDEX idx_features_date ON features(feature_date);
CREATE INDEX idx_features_team_date ON features(team_id, feature_date);

-- Predictions indices
CREATE INDEX idx_predictions_match_id ON predictions(match_id);
CREATE INDEX idx_predictions_model ON predictions(model_name);
CREATE INDEX idx_predictions_date ON predictions(prediction_date);
CREATE INDEX idx_predictions_model_date ON predictions(model_name, prediction_date);
CREATE INDEX idx_predictions_confidence ON predictions(confidence_score);

-- Odds indices
CREATE INDEX idx_odds_match_id ON odds(match_id);
CREATE INDEX idx_odds_bookmaker ON odds(bookmaker);
CREATE INDEX idx_odds_date ON odds(odds_date);
CREATE INDEX idx_odds_market_type ON odds(market_type);
CREATE INDEX idx_odds_match_market ON odds(match_id, market_type);
CREATE INDEX idx_odds_bookmaker_date ON odds(bookmaker, odds_date);

-- Betting tickets indices
CREATE INDEX idx_betting_tickets_type ON betting_tickets(ticket_type);
CREATE INDEX idx_betting_tickets_status ON betting_tickets(status);
CREATE INDEX idx_betting_tickets_placed_date ON betting_tickets(placed_date);
CREATE INDEX idx_betting_tickets_settled_date ON betting_tickets(settled_date);

-- Bets indices
CREATE INDEX idx_bets_ticket_id ON bets(ticket_id);
CREATE INDEX idx_bets_match_id ON bets(match_id);
CREATE INDEX idx_bets_prediction_id ON bets(prediction_id);
CREATE INDEX idx_bets_type ON bets(bet_type);
CREATE INDEX idx_bets_status ON bets(status);
CREATE INDEX idx_bets_edge ON bets(edge);
CREATE INDEX idx_bets_confidence ON bets(confidence);

-- Portfolio performance indices
CREATE INDEX idx_portfolio_performance_date ON portfolio_performance(date);
CREATE INDEX idx_portfolio_performance_roi ON portfolio_performance(roi);

-- Model performance indices
CREATE INDEX idx_model_performance_name ON model_performance(model_name);
CREATE INDEX idx_model_performance_date ON model_performance(evaluation_date);
CREATE INDEX idx_model_performance_type ON model_performance(dataset_type);
CREATE INDEX idx_model_performance_accuracy ON model_performance(accuracy);
CREATE INDEX idx_model_performance_roi ON model_performance(roi);

-- System logs indices
CREATE INDEX idx_system_logs_timestamp ON system_logs(timestamp);
CREATE INDEX idx_system_logs_level ON system_logs(level);
CREATE INDEX idx_system_logs_module ON system_logs(module);

-- Composite indices for common queries
CREATE INDEX idx_matches_teams_date ON matches(home_team_id, away_team_id, match_date);
CREATE INDEX idx_matches_upcoming ON matches(match_date, status) WHERE status = 'SCHEDULED';
CREATE INDEX idx_recent_matches ON matches(match_date DESC, league) WHERE match_date >= CURRENT_DATE - INTERVAL '30 days';

-- Partial indices for better performance
CREATE INDEX idx_pending_predictions ON predictions(match_id, prediction_date) 
WHERE prediction_date >= CURRENT_DATE;

CREATE INDEX idx_active_odds ON odds(match_id, bookmaker, odds_date) 
WHERE odds_date >= CURRENT_DATE - INTERVAL '7 days';

CREATE INDEX idx_pending_bets ON bets(match_id, status, confidence) 
WHERE status = 'PENDING';

-- GIN indices for JSONB columns
CREATE INDEX idx_match_weather_gin ON matches USING GIN (weather_conditions);
CREATE INDEX idx_match_events_details_gin ON match_events USING GIN (details);
CREATE INDEX idx_predictions_exact_scores_gin ON predictions USING GIN (exact_score_probs);
CREATE INDEX idx_predictions_feature_importance_gin ON predictions USING GIN (feature_importance);
CREATE INDEX idx_predictions_simulation_gin ON predictions USING GIN (simulation_results);
CREATE INDEX idx_odds_data_gin ON odds USING GIN (odds_data);
CREATE INDEX idx_system_logs_metadata_gin ON system_logs USING GIN (metadata);