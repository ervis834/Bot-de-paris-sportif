"""ETL pipeline for Bot Quantum Max."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from sqlalchemy import text
import uuid

from src.data.database import db_manager
from src.data.collectors.football_data import FootballDataCollector
from src.data.collectors.understat import UnderstatCollector
from src.data.collectors.weather import WeatherCollector
from src.data.collectors.odds_api import OddsAPICollector

logger = logging.getLogger(__name__)


class ETLPipeline:
    """Main ETL pipeline orchestrator."""
    
    def __init__(self):
        self.football_data = FootballDataCollector()
        self.understat = UnderstatCollector()
        self.weather = WeatherCollector()
        self.odds_api = OddsAPICollector()
        
    def run_daily_etl(self):
        """Run the daily ETL process."""
        logger.info("Starting daily ETL process")
        
        try:
            # 1. Update teams
            self._update_teams()
            
            # 2. Update recent matches
            self._update_recent_matches()
            
            # 3. Update upcoming matches
            self._update_upcoming_matches()
            
            # 4. Update xG data from Understat
            self._update_xg_data()
            
            # 5. Update weather data for upcoming matches
            self._update_weather_data()
            
            # 6. Update odds data
            self._update_odds_data()
            
            # 7. Update match statistics
            self._update_match_statistics()
            
            logger.info("Daily ETL process completed successfully")
            
        except Exception as e:
            logger.error(f"Daily ETL process failed: {e}")
            raise
    
    def _update_teams(self):
        """Update teams data."""
        logger.info("Updating teams data")
        
        competitions = self.football_data.get_competitions()
        all_teams = []
        
        for comp in competitions:
            try:
                teams = self.football_data.get_teams(comp["api_id"])
                for team in teams:
                    team["league"] = comp["code"]
                all_teams.extend(teams)
            except Exception as e:
                logger.error(f"Error fetching teams for {comp['name']}: {e}")
                continue
        
        if all_teams:
            # Convert to DataFrame for easier manipulation
            teams_df = pd.DataFrame(all_teams)
            
            # Insert/update teams
            for _, team in teams_df.iterrows():
                query = """
                INSERT INTO teams (id, api_id, name, short_name, country, league, founded, venue, logo_url)
                VALUES (:id, :api_id, :name, :short_name, :country, :league, :founded, :venue, :logo_url)
                ON CONFLICT (api_id) 
                DO UPDATE SET
                    name = EXCLUDED.name,
                    short_name = EXCLUDED.short_name,
                    country = EXCLUDED.country,
                    league = EXCLUDED.league,
                    founded = EXCLUDED.founded,
                    venue = EXCLUDED.venue,
                    logo_url = EXCLUDED.logo_url,
                    updated_at = CURRENT_TIMESTAMP
                """
                
                db_manager.execute_insert(query, {
                    "id": str(uuid.uuid4()),
                    "api_id": team["api_id"],
                    "name": team["name"],
                    "short_name": team.get("short_name"),
                    "country": team.get("country"),
                    "league": team["league"],
                    "founded": team.get("founded"),
                    "venue": team.get("venue"),
                    "logo_url": team.get("logo_url")
                })
        
        logger.info(f"Updated {len(all_teams)} teams")
    
    def _update_recent_matches(self):
        """Update recent matches data."""
        logger.info("Updating recent matches")
        
        matches = self.football_data.get_recent_matches(days_back=7)
        
        for match in matches:
            try:
                self._insert_or_update_match(match)
            except Exception as e:
                logger.error(f"Error updating match {match.get('api_id')}: {e}")
                continue
        
        logger.info(f"Updated {len(matches)} recent matches")
    
    def _update_upcoming_matches(self):
        """Update upcoming matches data."""
        logger.info("Updating upcoming matches")
        
        matches = self.football_data.get_upcoming_matches(days_ahead=14)
        
        for match in matches:
            try:
                self._insert_or_update_match(match)
            except Exception as e:
                logger.error(f"Error updating upcoming match {match.get('api_id')}: {e}")
                continue
        
        logger.info(f"Updated {len(matches)} upcoming matches")
    
    def _insert_or_update_match(self, match: Dict):
        """Insert or update a match record."""
        # Get team IDs from API IDs
        home_team_id = self._get_team_id_by_api_id(match["home_team_api_id"])
        away_team_id = self._get_team_id_by_api_id(match["away_team_api_id"])
        
        if not home_team_id or not away_team_id:
            logger.warning(f"Could not find team IDs for match {match['api_id']}")
            return
        
        query = """
        INSERT INTO matches (
            id, api_id, home_team_id, away_team_id, league, season, matchday,
            match_date, status, home_score, away_score, ht_home_score, ht_away_score,
            referee, venue, attendance
        )
        VALUES (
            :id, :api_id, :home_team_id, :away_team_id, :league, :season, :matchday,
            :match_date, :status, :home_score, :away_score, :ht_home_score, :ht_away_score,
            :referee, :venue, :attendance
        )
        ON CONFLICT (api_id)
        DO UPDATE SET
            status = EXCLUDED.status,
            home_score = EXCLUDED.home_score,
            away_score = EXCLUDED.away_score,
            ht_home_score = EXCLUDED.ht_home_score,
            ht_away_score = EXCLUDED.ht_away_score,
            referee = EXCLUDED.referee,
            venue = EXCLUDED.venue,
            attendance = EXCLUDED.attendance,
            updated_at = CURRENT_TIMESTAMP
        """
        
        db_manager.execute_insert(query, {
            "id": str(uuid.uuid4()),
            "api_id": match["api_id"],
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "league": match.get("league"),
            "season": match.get("season"),
            "matchday": match.get("matchday"),
            "match_date": match["match_date"],
            "status": match["status"],
            "home_score": match.get("home_score"),
            "away_score": match.get("away_score"),
            "ht_home_score": match.get("ht_home_score"),
            "ht_away_score": match.get("ht_away_score"),
            "referee": match.get("referee"),
            "venue": match.get("venue"),
            "attendance": match.get("attendance")
        })
    
    def _get_team_id_by_api_id(self, api_id: int) -> Optional[str]:
        """Get team UUID by API ID."""
        query = "SELECT id FROM teams WHERE api_id = :api_id"
        result = db_manager.execute_query(query, {"api_id": api_id})
        return result[0]["id"] if result else None
    
    def _update_xg_data(self):
        """Update xG data from Understat."""
        logger.info("Updating xG data from Understat")
        
        # Get recent matches that need xG data
        query = """
        SELECT m.id, m.api_id, m.match_date, ht.name as home_team, at.name as away_team
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE m.status = 'FINISHED' 
        AND m.match_date >= CURRENT_DATE - INTERVAL '7 days'
        AND NOT EXISTS (
            SELECT 1 FROM advanced_stats adv 
            WHERE adv.match_id = m.id
        )
        """
        
        matches_needing_xg = db_manager.execute_query(query)
        
        for match in matches_needing_xg:
            try:
                xg_data = self.understat.get_match_xg_data(
                    match["home_team"], 
                    match["away_team"], 
                    match["match_date"]
                )
                
                if xg_data:
                    self._insert_advanced_stats(match["id"], xg_data)
                    
            except Exception as e:
                logger.error(f"Error updating xG data for match {match['id']}: {e}")
                continue
        
        logger.info(f"Updated xG data for {len(matches_needing_xg)} matches")
    
    def _insert_advanced_stats(self, match_id: str, xg_data: Dict):
        """Insert advanced statistics for a match."""
        for team_name, stats in xg_data.items():
            team_id = self._get_team_id_by_name(team_name)
            if not team_id:
                continue
                
            query = """
            INSERT INTO advanced_stats (
                id, match_id, team_id, xg, xga, npxg, ppda, deep_completions, xa
            )
            VALUES (
                :id, :match_id, :team_id, :xg, :xga, :npxg, :ppda, :deep_completions, :xa
            )
            ON CONFLICT (match_id, team_id)
            DO UPDATE SET
                xg = EXCLUDED.xg,
                xga = EXCLUDED.xga,
                npxg = EXCLUDED.npxg,
                ppda = EXCLUDED.ppda,
                deep_completions = EXCLUDED.deep_completions,
                xa = EXCLUDED.xa
            """
            
            db_manager.execute_insert(query, {
                "id": str(uuid.uuid4()),
                "match_id": match_id,
                "team_id": team_id,
                "xg": stats.get("xG"),
                "xga": stats.get("xGA"),
                "npxg": stats.get("npxG"),
                "ppda": stats.get("PPDA"),
                "deep_completions": stats.get("deep_completions"),
                "xa": stats.get("xA")
            })
    
    def _get_team_id_by_name(self, team_name: str) -> Optional[str]:
        """Get team UUID by name."""
        query = "SELECT id FROM teams WHERE name ILIKE :name OR short_name ILIKE :name"
        result = db_manager.execute_query(query, {"name": f"%{team_name}%"})
        return result[0]["id"] if result else None
    
    def _update_weather_data(self):
        """Update weather data for upcoming matches."""
        logger.info("Updating weather data")
        
        # Get upcoming matches in next 7 days
        query = """
        SELECT m.id, m.match_date, m.venue, ht.name as home_team
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        WHERE m.match_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'
        AND m.status = 'SCHEDULED'
        AND m.weather_conditions IS NULL
        """
        
        upcoming_matches = db_manager.execute_query(query)
        
        for match in upcoming_matches:
            try:
                # Get weather data for match location
                weather_data = self.weather.get_weather_forecast(
                    match["venue"] or match["home_team"],
                    match["match_date"]
                )
                
                if weather_data:
                    # Update match with weather conditions
                    update_query = """
                    UPDATE matches 
                    SET weather_conditions = :weather_conditions
                    WHERE id = :match_id
                    """
                    
                    db_manager.execute_insert(update_query, {
                        "match_id": match["id"],
                        "weather_conditions": weather_data
                    })
                    
            except Exception as e:
                logger.error(f"Error updating weather for match {match['id']}: {e}")
                continue
        
        logger.info(f"Updated weather data for {len(upcoming_matches)} matches")
    
    def _update_odds_data(self):
        """Update odds data from multiple bookmakers."""
        logger.info("Updating odds data")
        
        # Get upcoming matches for odds collection
        query = """
        SELECT m.id, m.api_id, m.match_date, ht.name as home_team, at.name as away_team
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE m.match_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'
        AND m.status = 'SCHEDULED'
        """
        
        upcoming_matches = db_manager.execute_query(query)
        
        for match in upcoming_matches:
            try:
                # Get odds from multiple bookmakers
                odds_data = self.odds_api.get_match_odds(
                    match["home_team"],
                    match["away_team"],
                    match["match_date"]
                )
                
                for bookmaker, odds in odds_data.items():
                    self._insert_odds_data(match["id"], bookmaker, odds)
                    
            except Exception as e:
                logger.error(f"Error updating odds for match {match['id']}: {e}")
                continue
        
        logger.info(f"Updated odds for {len(upcoming_matches)} matches")
    
    def _insert_odds_data(self, match_id: str, bookmaker: str, odds_data: Dict):
        """Insert odds data for a match and bookmaker."""
        for market_type, odds in odds_data.items():
            query = """
            INSERT INTO odds (id, match_id, bookmaker, odds_date, market_type, odds_data)
            VALUES (:id, :match_id, :bookmaker, :odds_date, :market_type, :odds_data)
            ON CONFLICT (match_id, bookmaker, market_type, odds_date)
            DO UPDATE SET
                odds_data = EXCLUDED.odds_data
            """
            
            db_manager.execute_insert(query, {
                "id": str(uuid.uuid4()),
                "match_id": match_id,
                "bookmaker": bookmaker,
                "odds_date": datetime.now(),
                "market_type": market_type,
                "odds_data": odds
            })
    
    def _update_match_statistics(self):
        """Update detailed match statistics."""
        logger.info("Updating match statistics")
        
        # Get finished matches from last 3 days that need detailed stats
        query = """
        SELECT m.id, m.api_id
        FROM matches m
        WHERE m.status = 'FINISHED'
        AND m.match_date >= CURRENT_DATE - INTERVAL '3 days'
        AND NOT EXISTS (
            SELECT 1 FROM match_stats ms WHERE ms.match_id = m.id
        )
        """
        
        matches_needing_stats = db_manager.execute_query(query)
        
        for match in matches_needing_stats:
            try:
                # Get detailed match data from Football-Data API
                match_details = self.football_data.get_match_details(match["api_id"])
                
                if match_details and "home_stats" in match_details:
                    self._insert_match_statistics(
                        match["id"], 
                        match_details["home_stats"], 
                        match_details["away_stats"]
                    )
                    
            except Exception as e:
                logger.error(f"Error updating stats for match {match['id']}: {e}")
                continue
        
        logger.info(f"Updated statistics for {len(matches_needing_stats)} matches")
    
    def _insert_match_statistics(self, match_id: str, home_stats: Dict, away_stats: Dict):
        """Insert match statistics for both teams."""
        # Get team IDs
        teams_query = """
        SELECT home_team_id, away_team_id 
        FROM matches 
        WHERE id = :match_id
        """
        
        teams_result = db_manager.execute_query(teams_query, {"match_id": match_id})
        if not teams_result:
            return
            
        home_team_id = teams_result[0]["home_team_id"]
        away_team_id = teams_result[0]["away_team_id"]
        
        # Insert home team stats
        if home_stats:
            self._insert_team_match_stats(match_id, home_team_id, home_stats)
        
        # Insert away team stats
        if away_stats:
            self._insert_team_match_stats(match_id, away_team_id, away_stats)
    
    def _insert_team_match_stats(self, match_id: str, team_id: str, stats: Dict):
        """Insert statistics for a team in a match."""
        query = """
        INSERT INTO match_stats (
            id, match_id, team_id, possession, shots_total, shots_on_target,
            shots_off_target, shots_blocked, corners, fouls, yellow_cards,
            red_cards, offsides, passes_total, passes_accurate, pass_accuracy
        )
        VALUES (
            :id, :match_id, :team_id, :possession, :shots_total, :shots_on_target,
            :shots_off_target, :shots_blocked, :corners, :fouls, :yellow_cards,
            :red_cards, :offsides, :passes_total, :passes_accurate, :pass_accuracy
        )
        ON CONFLICT (match_id, team_id)
        DO UPDATE SET
            possession = EXCLUDED.possession,
            shots_total = EXCLUDED.shots_total,
            shots_on_target = EXCLUDED.shots_on_target,
            shots_off_target = EXCLUDED.shots_off_target,
            shots_blocked = EXCLUDED.shots_blocked,
            corners = EXCLUDED.corners,
            fouls = EXCLUDED.fouls,
            yellow_cards = EXCLUDED.yellow_cards,
            red_cards = EXCLUDED.red_cards,
            offsides = EXCLUDED.offsides,
            passes_total = EXCLUDED.passes_total,
            passes_accurate = EXCLUDED.passes_accurate,
            pass_accuracy = EXCLUDED.pass_accuracy
        """
        
        # Calculate pass accuracy
        passes_total = stats.get("passes", 0)
        passes_accurate = stats.get("passes_accurate", 0)
        pass_accuracy = (passes_accurate / passes_total * 100) if passes_total > 0 else None
        
        db_manager.execute_insert(query, {
            "id": str(uuid.uuid4()),
            "match_id": match_id,
            "team_id": team_id,
            "possession": stats.get("ball_possession"),
            "shots_total": stats.get("total_shots"),
            "shots_on_target": stats.get("shots_on_goal"),
            "shots_off_target": stats.get("shots_off_goal"),
            "shots_blocked": stats.get("blocked_shots"),
            "corners": stats.get("corner_kicks"),
            "fouls": stats.get("fouls"),
            "yellow_cards": stats.get("yellow_cards"),
            "red_cards": stats.get("red_cards"),
            "offsides": stats.get("offsides"),
            "passes_total": passes_total,
            "passes_accurate": passes_accurate,
            "pass_accuracy": pass_accuracy
        })
    
    def run_historical_etl(self, seasons: List[str] = None):
        """Run ETL for historical data."""
        logger.info("Starting historical ETL process")
        
        if not seasons:
            # Default to last 3 seasons
            current_year = datetime.now().year
            seasons = [str(year) for year in range(current_year - 3, current_year)]
        
        competitions = self.football_data.get_competitions()
        
        for season in seasons:
            logger.info(f"Processing season {season}")
            
            for comp in competitions:
                try:
                    # Get all matches for the season
                    matches = self.football_data.get_matches(
                        comp["api_id"], 
                        season=season
                    )
                    
                    for match in matches:
                        try:
                            self._insert_or_update_match(match)
                        except Exception as e:
                            logger.error(f"Error processing match {match.get('api_id')}: {e}")
                            continue
                    
                    logger.info(f"Processed {len(matches)} matches for {comp['name']} {season}")
                    
                except Exception as e:
                    logger.error(f"Error processing {comp['name']} {season}: {e}")
                    continue
        
        logger.info("Historical ETL process completed")


# Utility functions for standalone ETL operations
def run_daily_etl():
    """Run daily ETL process."""
    etl = ETLPipeline()
    etl.run_daily_etl()


def run_historical_etl(seasons: List[str] = None):
    """Run historical ETL process."""
    etl = ETLPipeline()
    etl.run_historical_etl(seasons)