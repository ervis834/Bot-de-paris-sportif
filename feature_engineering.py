        # Defensive features
        features.update(self._get_defensive_features(team_id, as_of_date, venue))
        
        # Performance streaks
        features.update(self._get_streak_features(team_id, as_of_date, venue))
        
        # Player quality features
        features.update(self._get_player_quality_features(team_id, as_of_date))
        
        # Injury/suspension features
        features.update(self._get_availability_features(team_id, as_of_date))
        
        return features
    
    def _get_form_features(self, team_id: str, as_of_date: datetime, venue: str = 'all') -> Dict:
        """Calculate form-based features."""
        venue_filter = ""
        if venue == 'home':
            venue_filter = "AND m.home_team_id = :team_id"
        elif venue == 'away':
            venue_filter = "AND m.away_team_id = :team_id"
        
        query = f"""
        WITH recent_matches AS (
            SELECT 
                m.match_date,
                CASE 
                    WHEN m.home_team_id = :team_id THEN 
                        CASE 
                            WHEN m.home_score > m.away_score THEN 3
                            WHEN m.home_score = m.away_score THEN 1
                            ELSE 0
                        END
                    ELSE 
                        CASE 
                            WHEN m.away_score > m.home_score THEN 3
                            WHEN m.away_score = m.home_score THEN 1
                            ELSE 0
                        END
                END as points,
                CASE 
                    WHEN m.home_team_id = :team_id THEN m.home_score
                    ELSE m.away_score
                END as goals_for,
                CASE 
                    WHEN m.home_team_id = :team_id THEN m.away_score
                    ELSE m.home_score
                END as goals_against,
                ROW_NUMBER() OVER (ORDER BY m.match_date DESC) as match_num
            FROM matches m
            WHERE (m.home_team_id = :team_id OR m.away_team_id = :team_id)
            AND m.status = 'FINISHED'
            AND m.match_date < :as_of_date
            {venue_filter}
            ORDER BY m.match_date DESC
            LIMIT 20
        )
        SELECT 
            AVG(CASE WHEN match_num <= 5 THEN points END) as form_5_games,
            AVG(CASE WHEN match_num <= 10 THEN points END) as form_10_games,
            AVG(points) as form_20_games,
            SUM(CASE WHEN match_num <= 5 AND points = 3 THEN 1 ELSE 0 END) as wins_last_5,
            SUM(CASE WHEN match_num <= 5 AND points = 1 THEN 1 ELSE 0 END) as draws_last_5,
            SUM(CASE WHEN match_num <= 5 AND points = 0 THEN 1 ELSE 0 END) as losses_last_5,
            AVG(CASE WHEN match_num <= 5 THEN goals_for END) as goals_for_avg_5,
            AVG(CASE WHEN match_num <= 5 THEN goals_against END) as goals_against_avg_5,
            STDDEV(CASE WHEN match_num <= 5 THEN points END) as form_consistency_5
        FROM recent_matches
        """
        
        result = db_manager.execute_query(query, {
            "team_id": team_id,
            "as_of_date": as_of_date
        })
        
        if result:
            return {k: v or 0 for k, v in result[0].items()}
        return {}
    
    def _get_goal_features(self, team_id: str, as_of_date: datetime, venue: str = 'all') -> Dict:
        """Calculate goal-related features."""
        venue_filter = ""
        if venue == 'home':
            venue_filter = "AND m.home_team_id = :team_id"
        elif venue == 'away':
            venue_filter = "AND m.away_team_id = :team_id"
        
        query = f"""
        WITH goal_stats AS (
            SELECT 
                CASE 
                    WHEN m.home_team_id = :team_id THEN m.home_score
                    ELSE m.away_score
                END as goals_for,
                CASE 
                    WHEN m.home_team_id = :team_id THEN m.away_score
                    ELSE m.home_score
                END as goals_against,
                CASE 
                    WHEN m.home_team_id = :team_id THEN m.ht_home_score
                    ELSE m.ht_away_score
                END as ht_goals_for,
                CASE 
                    WHEN m.home_team_id = :team_id THEN m.ht_away_score
                    ELSE m.ht_home_score
                END as ht_goals_against,
                m.match_date
            FROM matches m
            WHERE (m.home_team_id = :team_id OR m.away_team_id = :team_id)
            AND m.status = 'FINISHED'
            AND m.match_date < :as_of_date
            AND m.match_date >= :as_of_date - INTERVAL '{self.lookback_days} days'
            {venue_filter}
        )
        SELECT 
            AVG(goals_for) as avg_goals_for,
            AVG(goals_against) as avg_goals_against,
            AVG(goals_for + goals_against) as avg_total_goals,
            STDDEV(goals_for) as goals_for_consistency,
            STDDEV(goals_against) as goals_against_consistency,
            AVG(ht_goals_for) as avg_ht_goals_for,
            AVG(ht_goals_against) as avg_ht_goals_against,
            SUM(CASE WHEN goals_for > 0 AND goals_against > 0 THEN 1 ELSE 0 END)::float / COUNT(*) as btts_rate,
            SUM(CASE WHEN goals_for + goals_against > 2.5 THEN 1 ELSE 0 END)::float / COUNT(*) as over_2_5_rate,
            MAX(goals_for) as max_goals_for,
            MAX(goals_against) as max_goals_against,
            COUNT(*) as matches_count
        FROM goal_stats
        """
        
        result = db_manager.execute_query(query, {
            "team_id": team_id,
            "as_of_date": as_of_date
        })
        
        if result:
            return {k: v or 0 for k, v in result[0].items()}
        return {}
    
    def _get_xg_features(self, team_id: str, as_of_date: datetime, venue: str = 'all') -> Dict:
        """Calculate xG-based features."""
        venue_filter = ""
        if venue == 'home':
            venue_filter = "AND m.home_team_id = :team_id"
        elif venue == 'away':
            venue_filter = "AND m.away_team_id = :team_id"
        
        query = f"""
        WITH xg_stats AS (
            SELECT 
                adv.xg,
                adv.xga,
                adv.npxg,
                adv.xa,
                adv.deep_completions,
                adv.ppda,
                m.match_date
            FROM advanced_stats adv
            JOIN matches m ON adv.match_id = m.id
            WHERE adv.team_id = :team_id
            AND m.match_date < :as_of_date
            AND m.match_date >= :as_of_date - INTERVAL '{self.lookback_days} days'
            {venue_filter}
        )
        SELECT 
            AVG(xg) as avg_xg,
            AVG(xga) as avg_xga,
            AVG(xg - xga) as avg_xg_diff,
            AVG(npxg) as avg_npxg,
            AVG(xa) as avg_xa,
            AVG(deep_completions) as avg_deep_completions,
            AVG(ppda) as avg_ppda,
            STDDEV(xg) as xg_consistency,
            STDDEV(xga) as xga_consistency,
            COUNT(*) as xg_matches_count
        FROM xg_stats
        """
        
        result = db_manager.execute_query(query, {
            "team_id": team_id,
            "as_of_date": as_of_date
        })
        
        if result:
            return {k: v or 0 for k, v in result[0].items()}
        return {}
    
    def _get_tactical_features(self, team_id: str, as_of_date: datetime, venue: str = 'all') -> Dict:
        """Calculate tactical features."""
        venue_filter = ""
        if venue == 'home':
            venue_filter = "AND m.home_team_id = :team_id"
        elif venue == 'away':
            venue_filter = "AND m.away_team_id = :team_id"
        
        query = f"""
        WITH tactical_stats AS (
            SELECT 
                ms.possession,
                ms.shots_total,
                ms.shots_on_target,
                ms.pass_accuracy,
                ms.corners,
                m.match_date,
                CASE WHEN ms.shots_total > 0 THEN ms.shots_on_target::float / ms.shots_total ELSE 0 END as shot_accuracy
            FROM match_stats ms
            JOIN matches m ON ms.match_id = m.id
            WHERE ms.team_id = :team_id
            AND m.match_date < :as_of_date
            AND m.match_date >= :as_of_date - INTERVAL '{self.lookback_days} days'
            {venue_filter}
        )
        SELECT 
            AVG(possession) as avg_possession,
            AVG(shots_total) as avg_shots_total,
            AVG(shots_on_target) as avg_shots_on_target,
            AVG(shot_accuracy) as avg_shot_accuracy,
            AVG(pass_accuracy) as avg_pass_accuracy,
            AVG(corners) as avg_corners,
            STDDEV(possession) as possession_consistency,
            STDDEV(shots_total) as shots_consistency,
            COUNT(*) as tactical_matches_count
        FROM tactical_stats
        """
        
        result = db_manager.execute_query(query, {
            "team_id": team_id,
            "as_of_date": as_of_date
        })
        
        if result:
            return {k: v or 0 for k, v in result[0].items()}
        return {}
    
    def _get_defensive_features(self, team_id: str, as_of_date: datetime, venue: str = 'all') -> Dict:
        """Calculate defensive features."""
        venue_filter = ""
        if venue == 'home':
            venue_filter = "AND m.home_team_id = :team_id"
        elif venue == 'away':
            venue_filter = "AND m.away_team_id = :team_id"
        
        query = f"""
        WITH defensive_stats AS (
            SELECT 
                ms.fouls,
                ms.yellow_cards,
                ms.red_cards,
                ms.offsides,
                CASE 
                    WHEN m.home_team_id = :team_id THEN m.away_score
                    ELSE m.home_score
                END as goals_conceded,
                m.match_date
            FROM match_stats ms
            JOIN matches m ON ms.match_id = m.id
            WHERE ms.team_id = :team_id
            AND m.match_date < :as_of_date
            AND m.match_date >= :as_of_date - INTERVAL '{self.lookback_days} days'
            {venue_filter}
        )
        SELECT 
            AVG(fouls) as avg_fouls,
            AVG(yellow_cards) as avg_yellow_cards,
            AVG(red_cards) as avg_red_cards,
            AVG(offsides) as avg_offsides,
            SUM(CASE WHEN goals_conceded = 0 THEN 1 ELSE 0 END)::float / COUNT(*) as clean_sheet_rate,
            AVG(goals_conceded) as avg_goals_conceded,
            COUNT(*) as defensive_matches_count
        FROM defensive_stats
        """
        
        result = db_manager.execute_query(query, {
            "team_id": team_id,
            "as_of_date": as_of_date
        })
        
        if result:
            return {k: v or 0 for k, v in result[0].items()}
        return {}
    
    def _get_streak_features(self, team_id: str, as_of_date: datetime, venue: str = 'all') -> Dict:
        """Calculate streak-based features."""
        venue_filter = ""
        if venue == 'home':
            venue_filter = "AND m.home_team_id = :team_id"
        elif venue == 'away':
            venue_filter = "AND m.away_team_id = :team_id"
        
        # Get recent match results in order
        query = f"""
        SELECT 
            CASE 
                WHEN m.home_team_id = :team_id THEN 
                    CASE 
                        WHEN m.home_score > m.away_score THEN 'W'
                        WHEN m.home_score = m.away_score THEN 'D'
                        ELSE 'L'
                    END
                ELSE 
                    CASE 
                        WHEN m.away_score > m.home_score THEN 'W'
                        WHEN m.away_score = m.home_score THEN 'D'
                        ELSE 'L'
                    END
            END as result,
            CASE 
                WHEN m.home_team_id = :team_id THEN m.home_score
                ELSE m.away_score
            END as goals_for,
            CASE 
                WHEN m.home_team_id = :team_id THEN m.away_score
                ELSE m.home_score
            END as goals_against
        FROM matches m
        WHERE (m.home_team_id = :team_id OR m.away_team_id = :team_id)
        AND m.status = 'FINISHED'
        AND m.match_date < :as_of_date
        {venue_filter}
        ORDER BY m.match_date DESC
        LIMIT 15
        """
        
        results = db_manager.execute_query(query, {
            "team_id": team_id,
            "as_of_date": as_of_date
        })
        
        if not results:
            return {}
        
        # Calculate streaks
        current_win_streak = 0
        current_unbeaten_streak = 0
        current_loss_streak = 0
        current_scoring_streak = 0
        current_clean_sheet_streak = 0
        
        for match in results:
            result = match['result']
            goals_for = match['goals_for'] or 0
            goals_against = match['goals_against'] or 0
            
            # Win streak
            if result == 'W':
                current_win_streak += 1
            else:
                break
        
        # Unbeaten streak
        for match in results:
            if match['result'] in ['W', 'D']:
                current_unbeaten_streak += 1
            else:
                break
        
        # Loss streak
        for match in results:
            if match['result'] == 'L':
                current_loss_streak += 1
            else:
                break
        
        # Scoring streak
        for match in results:
            if (match['goals_for'] or 0) > 0:
                current_scoring_streak += 1
            else:
                break
        
        # Clean sheet streak
        for match in results:
            if (match['goals_against'] or 0) == 0:
                current_clean_sheet_streak += 1
            else:
                break
        
        return {
            'current_win_streak': current_win_streak,
            'current_unbeaten_streak': current_unbeaten_streak,
            'current_loss_streak': current_loss_streak,
            'current_scoring_streak': current_scoring_streak,
            'current_clean_sheet_streak': current_clean_sheet_streak
        }
    
    def _get_player_quality_features(self, team_id: str, as_of_date: datetime) -> Dict:
        """Calculate player quality features."""
        query = """
        SELECT 
            AVG(p.market_value) as avg_market_value,
            SUM(p.market_value) as total_market_value,
            MAX(p.market_value) as max_market_value,
            COUNT(*) as squad_size,
            AVG(p.age) as avg_age,
            COUNT(CASE WHEN p.position = 'Goalkeeper' THEN 1 END) as goalkeepers,
            COUNT(CASE WHEN p.position IN ('Defender', 'Centre-Back', 'Left-Back', 'Right-Back') THEN 1 END) as defenders,
            COUNT(CASE WHEN p.position IN ('Midfielder', 'Defensive Midfield', 'Central Midfield', 'Attacking Midfield') THEN 1 END) as midfielders,
            COUNT(CASE WHEN p.position IN ('Forward', 'Centre-Forward', 'Left Winger', 'Right Winger') THEN 1 END) as forwards
        FROM players p
        WHERE p.team_id = :team_id
        """
        
        result = db_manager.execute_query(query, {"team_id": team_id})
        
        if result:
            return {k: v or 0 for k, v in result[0].items()}
        return {}
    
    def _get_availability_features(self, team_id: str, as_of_date: datetime) -> Dict:
        """Calculate injury/suspension features."""
        # This would require additional tables for injuries/suspensions
        # For now, return placeholder values
        return {
            'injured_players': 0,
            'suspended_players': 0,
            'key_players_available': 1.0
        }
    
    def _generate_match_level_features(self, home_team_id: str, away_team_id: str, as_of_date: datetime) -> Dict:
        """Generate features that compare the two teams directly."""
        # Head-to-head record
        h2h_features = self._get_head_to_head_features(home_team_id, away_team_id, as_of_date)
        
        # League position difference (if available)
        position_features = self._get_position_features(home_team_id, away_team_id, as_of_date)
        
        # Recent form comparison
        form_features = self._get_form_comparison_features(home_team_id, away_team_id, as_of_date)
        
        return {
            **h2h_features,
            **position_features,
            **form_features
        }
    
    def _get_head_to_head_features(self, home_team_id: str, away_team_id: str, as_of_date: datetime) -> Dict:
        """Calculate head-to-head features."""
        query = """
        WITH h2h_matches AS (
            SELECT 
                m.match_date,
                CASE 
                    WHEN m.home_team_id = :home_team_id THEN 
                        CASE 
                            WHEN m.home_score > m.away_score THEN 'home_win'
                            WHEN m.home_score = m.away_score THEN 'draw'
                            ELSE 'away_win'
                        END
                    ELSE 
                        CASE 
                            WHEN m.away_score > m.home_score THEN 'home_win'
                            WHEN m.away_score = m.home_score THEN 'draw'
                            ELSE 'away_win'
                        END
                END as result,
                CASE 
                    WHEN m.home_team_id = :home_team_id THEN m.home_score
                    ELSE m.away_score
                END as home_goals,
                CASE 
                    WHEN m.home_team_id = :home_team_id THEN m.away_score
                    ELSE m.home_score
                END as away_goals
            FROM matches m
            WHERE ((m.home_team_id = :home_team_id AND m.away_team_id = :away_team_id)
                OR (m.home_team_id = :away_team_id AND m.away_team_id = :home_team_id))
            AND m.status = 'FINISHED'
            AND m.match_date < :as_of_date
            ORDER BY m.match_date DESC
            LIMIT 10
        )
        SELECT 
            COUNT(*) as h2h_matches,
            SUM(CASE WHEN result = 'home_win' THEN 1 ELSE 0 END) as home_wins,
            SUM(CASE WHEN result = 'draw' THEN 1 ELSE 0 END) as draws,
            SUM(CASE WHEN result = 'away_win' THEN 1 ELSE 0 END) as away_wins,
            AVG(home_goals) as avg_home_goals_h2h,
            AVG(away_goals) as avg_away_goals_h2h,
            AVG(home_goals + away_goals) as avg_total_goals_h2h
        FROM h2h_matches
        """
        
        result = db_manager.execute_query(query, {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "as_of_date": as_of_date
        })
        
        if result and result[0]['h2h_matches'] > 0:
            data = result[0]
            total_matches = data['h2h_matches']
            return {
                'h2h_matches': total_matches,
                'h2h_home_win_rate': data['home_wins'] / total_matches,
                'h2h_draw_rate': data['draws'] / total_matches,
                'h2h_away_win_rate': data['away_wins'] / total_matches,
                'h2h_avg_home_goals': data['avg_home_goals_h2h'] or 0,
                'h2h_avg_away_goals': data['avg_away_goals_h2h'] or 0,
                'h2h_avg_total_goals': data['avg_total_goals_h2h'] or 0
            }
        
        return {
            'h2h_matches': 0,
            'h2h_home_win_rate': 0.33,  # Default expectation
            'h2h_draw_rate': 0.27,
            'h2h_away_win_rate': 0.40,
            'h2h_avg_home_goals': 1.3,
            'h2h_avg_away_goals': 1.1,
            'h2h_avg_total_goals': 2.4
        }
    
    def _get_position_features(self, home_team_id: str, away_team_id: str, as_of_date: datetime) -> Dict:
        """Calculate league position-based features."""
        # This would require standings data which we'd calculate from matches
        # For now, return placeholder
        return {
            'position_difference': 0,
            'points_difference': 0
        }
    
    def _get_form_comparison_features(self, home_team_id: str, away_team_id: str, as_of_date: datetime) -> Dict:
        """Compare recent form between teams."""
        home_form = self._get_form_features(home_team_id, as_of_date, 'all')
        away_form = self._get_form_features(away_team_id, as_of_date, 'all')
        
        return {
            'form_difference_5': (home_form.get('form_5_games', 0) - away_form.get('form_5_games', 0)),
            'form_difference_10': (home_form.get('form_10_games', 0) - away_form.get('form_10_games', 0)),
            'goals_for_difference': (home_form.get('goals_for_avg_5', 0) - away_form.get('goals_for_avg_5', 0)),
            'goals_against_difference': (home_form.get('goals_against_avg_5', 0) - away_form.get('goals_against_avg_5', 0))
        }
    
    def _generate_contextual_features(self, match_info: Dict, as_of_date: datetime) -> Dict:
        """Generate contextual features for the match."""
        features = {}
        
        # Day of week
        match_date = match_info['match_date']
        features['day_of_week'] = match_date.weekday()
        features['is_weekend'] = 1 if match_date.weekday() >= 5 else 0
        
        # Month/season timing
        features['month'] = match_date.month
        features['is_winter'] = 1 if match_date.month in [12, 1, 2] else 0
        features['is_summer'] = 1 if match_date.month in [6, 7, 8] else 0
        
        # Competition type
        league = match_info.get('league', '')
        features['is_domestic_league'] = 1 if league in ['PL', 'FL1', 'BL1', 'SA', 'PD'] else 0
        features['is_european_competition'] = 1 if league in ['CL', 'EL'] else 0
        
        # Weather impact (if available)
        weather = match_info.get('weather_conditions')
        if weather:
            features['weather_impact'] = self._calculate_weather_impact(weather)
        else:
            features['weather_impact'] = 0
        
        return features
    
    def _calculate_weather_impact(self, weather_data: Dict) -> float:
        """Calculate weather impact score."""
        if not isinstance(weather_data, dict):
            return 0
        
        impact = 0
        
        # Temperature
        temp = weather_data.get('temperature', 20)
        if temp < 5 or temp > 30:
            impact += 2
        elif temp < 10 or temp > 25:
            impact += 1
        
        # Wind
        wind_speed = weather_data.get('wind_speed', 0)
        if wind_speed > 15:
            impact += 3
        elif wind_speed > 10:
            impact += 2
        elif wind_speed > 5:
            impact += 1
        
        # Precipitation
        rain = weather_data.get('rain_3h', 0)
        if rain > 5:
            impact += 3
        elif rain > 1:
            impact += 2
        elif rain > 0:
            impact += 1
        
        return min(impact, 5)  # Cap at 5


class FeaturePipeline:
    """Pipeline for processing features for ML models."""
    
    def __init__(self):
        self.engineer = FeatureEngineer()
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, match_ids: List[str], as_of_dates: List[datetime] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit the pipeline and transform match data."""
        if not as_of_dates:
            as_of_dates = [datetime.now()] * len(match_ids)
        
        # Generate features for all matches
        features_list = []
        targets = []
        
        for match_id, as_of_date in zip(match_ids, as_of_dates):
            features = self.engineer.generate_match_features(match_id, as_of_date)
            if features:
                features_list.append(features)
                # Get match result for target
                target = self._get_match_target(match_id)
                targets.append(target)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        targets_series = pd.Series(targets)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Store feature columns
        self.feature_columns = features_df.columns.tolist()
        
        # Scale numerical features
        numerical_features = features_df.select_dtypes(include=[np.number]).columns
        features_df[numerical_features] = self.scaler.fit_transform(features_df[numerical_features])
        
        self.is_fitted = True
        return features_df, targets_series
    
    def transform(self, match_ids: List[str], as_of_dates: List[datetime] = None) -> pd.DataFrame:
        """Transform new match data using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        if not as_of_dates:
            as_of_dates = [datetime.now()] * len(match_ids)
        
        # Generate features
        features_list = []
        for match_id, as_of_date in zip(match_ids, as_of_dates):
            features = self.engineer.generate_match_features(match_id, as_of_date)
            if features:
                features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Ensure all columns are present
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns to match training
        features_df = features_df[self.feature_columns]
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Scale numerical features
        numerical_features = features_df.select_dtypes(include=[np.number]).columns
        features_df[numerical_features] = self.scaler.transform(features_df[numerical_features])
        
        return features_df
    
    def _get_match_target(self, match_id: str) -> int:
        """Get match result for target variable."""
        query = """
        SELECT home_score, away_score
        FROM matches
        WHERE id = :match_id AND status = 'FINISHED'
        """
        
        result = db_manager.execute_query(query, {"match_id": match_id})
        
        if result:
            home_score = result[0]['home_score']
            away_score = result[0]['away_score']
            
            if home_score > away_score:
                return 2  # Home win
            elif away_score > home"""Feature engineering module for Bot Quantum Max."""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import networkx as nx

from src.data.database import db_manager, feature_queries
from config.settings import settings

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Main feature engineering class."""
    
    def __init__(self):
        self.lookback_days = settings.lookback_days
        self.min_matches = settings.min_matches_for_prediction
        self.scalers = {}
        
    def generate_match_features(self, match_id: str, as_of_date: datetime = None) -> Dict:
        """Generate all features for a match."""
        if not as_of_date:
            as_of_date = datetime.now()
        
        # Get match details
        match_info = self._get_match_info(match_id)
        if not match_info:
            return {}
        
        home_team_id = match_info['home_team_id']
        away_team_id = match_info['away_team_id']
        
        # Generate features for both teams
        home_features = self._generate_team_features(
            home_team_id, as_of_date, venue='home', opponent_id=away_team_id
        )
        away_features = self._generate_team_features(
            away_team_id, as_of_date, venue='away', opponent_id=home_team_id
        )
        
        # Generate match-level features
        match_features = self._generate_match_level_features(
            home_team_id, away_team_id, as_of_date
        )
        
        # Generate contextual features
        context_features = self._generate_contextual_features(match_info, as_of_date)
        
        # Combine all features
        all_features = {
            **{f"home_{k}": v for k, v in home_features.items()},
            **{f"away_{k}": v for k, v in away_features.items()},
            **match_features,
            **context_features
        }
        
        return all_features
    
    def _get_match_info(self, match_id: str) -> Optional[Dict]:
        """Get basic match information."""
        query = """
        SELECT m.id, m.home_team_id, m.away_team_id, m.match_date, 
               m.league, m.season, m.venue, m.weather_conditions,
               ht.name as home_team_name, at.name as away_team_name
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE m.id = :match_id
        """
        
        result = db_manager.execute_query(query, {"match_id": match_id})
        return result[0] if result else None
    
    def _generate_team_features(self, team_id: str, as_of_date: datetime, 
                               venue: str = 'all', opponent_id: str = None) -> Dict:
        """Generate comprehensive team features."""
        features = {}
        
        # Form features
        features.update(self._get_form_features(team_id, as_of_date, venue))
        
        # Goal features
        features.update(self._get_goal_features(team_id, as_of_date, venue))
        
        # xG features
        features.update(self._get_xg_features(team_id, as_of_date, venue))
        
        # Tactical features
        features.update(self._get_tactical_features(team_id, as_of_date, venue))
        
        # Defensive features
        features.update(self._get_defensive_features(