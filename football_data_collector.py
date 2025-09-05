"""Football-Data.org API collector."""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
from config.settings import settings

logger = logging.getLogger(__name__)


class FootballDataCollector:
    """Collects data from Football-Data.org API."""
    
    def __init__(self):
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {
            "X-Auth-Token": settings.football_data_api_key,
            "Content-Type": "application/json"
        }
        self.rate_limit_delay = 6  # seconds between requests (free tier: 10 requests/minute)
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting and error handling."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, headers=self.headers, params=params or {})
            response.raise_for_status()
            
            logger.info(f"Successfully fetched data from {endpoint}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from {endpoint}: {e}")
            raise
    
    def get_competitions(self) -> List[Dict]:
        """Get available competitions."""
        data = self._make_request("competitions")
        competitions = []
        
        for comp in data.get("competitions", []):
            if comp["code"] in settings.supported_leagues:
                competitions.append({
                    "api_id": comp["id"],
                    "name": comp["name"],
                    "code": comp["code"],
                    "country": comp["area"]["name"],
                    "current_season": comp.get("currentSeason", {})
                })
        
        return competitions
    
    def get_teams(self, competition_id: int) -> List[Dict]:
        """Get teams for a competition."""
        data = self._make_request(f"competitions/{competition_id}/teams")
        teams = []
        
        for team in data.get("teams", []):
            teams.append({
                "api_id": team["id"],
                "name": team["name"],
                "short_name": team["tla"],
                "country": team["area"]["name"],
                "founded": team.get("founded"),
                "venue": team.get("venue"),
                "logo_url": team.get("crest"),
                "league": data.get("competition", {}).get("code", ""),
            })
        
        return teams
    
    def get_matches(self, competition_id: int, season: str = None, 
                   date_from: str = None, date_to: str = None) -> List[Dict]:
        """Get matches for a competition."""
        params = {}
        if season:
            params["season"] = season
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        
        data = self._make_request(f"competitions/{competition_id}/matches", params)
        matches = []
        
        for match in data.get("matches", []):
            # Parse match date
            match_date = datetime.fromisoformat(
                match["utcDate"].replace("Z", "+00:00")
            )
            
            # Extract scores
            score = match.get("score", {})
            home_score = score.get("fullTime", {}).get("home")
            away_score = score.get("fullTime", {}).get("away")
            ht_home_score = score.get("halfTime", {}).get("home")
            ht_away_score = score.get("halfTime", {}).get("away")
            
            matches.append({
                "api_id": match["id"],
                "home_team_api_id": match["homeTeam"]["id"],
                "away_team_api_id": match["awayTeam"]["id"],
                "league": data.get("competition", {}).get("code", ""),
                "season": match.get("season", {}).get("startDate", "")[:4],
                "matchday": match.get("matchday"),
                "match_date": match_date,
                "status": match["status"],
                "home_score": home_score,
                "away_score": away_score,
                "ht_home_score": ht_home_score,
                "ht_away_score": ht_away_score,
                "referee": match.get("referees", [{}])[0].get("name") if match.get("referees") else None,
                "venue": match["homeTeam"].get("venue"),
                "attendance": match.get("attendance")
            })
        
        return matches
    
    def get_match_details(self, match_id: int) -> Dict:
        """Get detailed match information."""
        try:
            data = self._make_request(f"matches/{match_id}")
            match = data
            
            # Extract basic match info
            match_date = datetime.fromisoformat(
                match["utcDate"].replace("Z", "+00:00")
            )
            
            # Extract scores
            score = match.get("score", {})
            home_score = score.get("fullTime", {}).get("home")
            away_score = score.get("fullTime", {}).get("away")
            
            # Extract statistics if available
            home_stats = {}
            away_stats = {}
            
            if "statistics" in match:
                for stat in match["statistics"]:
                    team_stats = {}
                    for item in stat.get("statistics", []):
                        team_stats[item["type"].lower().replace(" ", "_")] = item["value"]
                    
                    if stat["team"]["id"] == match["homeTeam"]["id"]:
                        home_stats = team_stats
                    else:
                        away_stats = team_stats
            
            return {
                "match": {
                    "api_id": match["id"],
                    "home_team_api_id": match["homeTeam"]["id"],
                    "away_team_api_id": match["awayTeam"]["id"],
                    "match_date": match_date,
                    "status": match["status"],
                    "home_score": home_score,
                    "away_score": away_score,
                    "venue": match.get("venue"),
                    "attendance": match.get("attendance")
                },
                "home_stats": home_stats,
                "away_stats": away_stats
            }
            
        except Exception as e:
            logger.error(f"Error fetching match details for {match_id}: {e}")
            return {}
    
    def get_standings(self, competition_id: int, season: str = None) -> List[Dict]:
        """Get competition standings."""
        params = {}
        if season:
            params["season"] = season
            
        data = self._make_request(f"competitions/{competition_id}/standings", params)
        standings = []
        
        if "standings" in data and data["standings"]:
            for table_entry in data["standings"][0]["table"]:
                standings.append({
                    "team_api_id": table_entry["team"]["id"],
                    "position": table_entry["position"],
                    "points": table_entry["points"],
                    "played_games": table_entry["playedGames"],
                    "won": table_entry["won"],
                    "draw": table_entry["draw"],
                    "lost": table_entry["lost"],
                    "goals_for": table_entry["goalsFor"],
                    "goals_against": table_entry["goalsAgainst"],
                    "goal_difference": table_entry["goalDifference"]
                })
        
        return standings
    
    def get_recent_matches(self, days_back: int = 30) -> List[Dict]:
        """Get recent matches across all supported leagues."""
        date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        date_to = datetime.now().strftime("%Y-%m-%d")
        
        all_matches = []
        competitions = self.get_competitions()
        
        for comp in competitions:
            try:
                matches = self.get_matches(
                    comp["api_id"],
                    date_from=date_from,
                    date_to=date_to
                )
                all_matches.extend(matches)
                logger.info(f"Fetched {len(matches)} matches from {comp['name']}")
            except Exception as e:
                logger.error(f"Error fetching matches for {comp['name']}: {e}")
                continue
        
        return all_matches
    
    def get_upcoming_matches(self, days_ahead: int = 7) -> List[Dict]:
        """Get upcoming matches across all supported leagues."""
        date_from = datetime.now().strftime("%Y-%m-%d")
        date_to = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        all_matches = []
        competitions = self.get_competitions()
        
        for comp in competitions:
            try:
                matches = self.get_matches(
                    comp["api_id"],
                    date_from=date_from,
                    date_to=date_to
                )
                all_matches.extend(matches)
                logger.info(f"Fetched {len(matches)} upcoming matches from {comp['name']}")
            except Exception as e:
                logger.error(f"Error fetching upcoming matches for {comp['name']}: {e}")
                continue
        
        return all_matches