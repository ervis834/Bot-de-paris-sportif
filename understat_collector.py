"""Understat.com data collector for xG and advanced statistics."""

import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)


class UnderstatCollector:
    """Collects xG and advanced statistics from Understat.com."""
    
    def __init__(self):
        self.base_url = "https://understat.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Team name mappings from various APIs to Understat format
        self.team_mappings = {
            'Manchester United': 'Manchester_United',
            'Manchester City': 'Manchester_City',
            'Tottenham Hotspur': 'Tottenham',
            'Newcastle United': 'Newcastle_United',
            'West Ham United': 'West_Ham',
            'Brighton & Hove Albion': 'Brighton',
            'Nottingham Forest': 'Nottingham_Forest',
            'Sheffield United': 'Sheffield_United',
            'Crystal Palace': 'Crystal_Palace',
            # Add more mappings as needed
        }
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team name for Understat format."""
        return self.team_mappings.get(team_name, team_name.replace(' ', '_'))
    
    def _make_request(self, url: str, delay: float = 1.0) -> Optional[requests.Response]:
        """Make HTTP request with rate limiting."""
        try:
            time.sleep(delay)
            response = self.session.get(url)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def get_league_data(self, league: str, season: str) -> Optional[pd.DataFrame]:
        """Get league data for a specific season."""
        url = f"{self.base_url}/league/{league}/{season}"
        response = self._make_request(url)
        
        if not response:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the script containing the data
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'teamsData' in script.string:
                # Extract JSON data from script
                json_match = re.search(r'JSON\.parse\(\'(.+?)\'\)', script.string)
                if json_match:
                    try:
                        json_str = json_match.group(1).encode().decode('unicode_escape')
                        data = json.loads(json_str)
                        
                        # Convert to DataFrame
                        teams_data = []
                        for team_data in data:
                            teams_data.append({
                                'team_name': team_data.get('title'),
                                'matches': team_data.get('matches'),
                                'wins': team_data.get('wins'),
                                'draws': team_data.get('draws'),
                                'loses': team_data.get('loses'),
                                'scored': team_data.get('scored'),
                                'missed': team_data.get('missed'),
                                'xG': float(team_data.get('xG', 0)),
                                'xGA': float(team_data.get('xGA', 0)),
                                'xPTS': float(team_data.get('xPTS', 0)),
                                'deep': team_data.get('deep'),
                                'ppda': team_data.get('ppda')
                            })
                        
                        return pd.DataFrame(teams_data)
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Error parsing league data: {e}")
                        return None
        
        return None
    
    def get_team_matches(self, team: str, season: str) -> Optional[pd.DataFrame]:
        """Get detailed match data for a team."""
        team_normalized = self._normalize_team_name(team)
        url = f"{self.base_url}/team/{team_normalized}/{season}"
        response = self._make_request(url)
        
        if not response:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the script containing match data
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'datesData' in script.string:
                # Extract JSON data
                json_match = re.search(r'JSON\.parse\(\'(.+?)\'\)', script.string)
                if json_match:
                    try:
                        json_str = json_match.group(1).encode().decode('unicode_escape')
                        data = json.loads(json_str)
                        
                        matches_data = []
                        for match in data:
                            matches_data.append({
                                'date': match.get('datetime'),
                                'opponent': match.get('title'),
                                'venue': 'home' if match.get('side') == 'h' else 'away',
                                'goals_for': match.get('goals', {}).get('h' if match.get('side') == 'h' else 'a'),
                                'goals_against': match.get('goals', {}).get('a' if match.get('side') == 'h' else 'h'),
                                'xG_for': float(match.get('xG', 0)),
                                'xG_against': float(match.get('xGA', 0)),
                                'npxG': float(match.get('npxG', 0)),
                                'deep': match.get('deep'),
                                'ppda': match.get('ppda'),
                                'result': match.get('result')
                            })
                        
                        return pd.DataFrame(matches_data)
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Error parsing team matches: {e}")
                        return None
        
        return None
    
    def get_match_xg_data(self, home_team: str, away_team: str, 
                         match_date: datetime) -> Optional[Dict]:
        """Get xG data for a specific match."""
        # Try to find the match in recent data for both teams
        season = str(match_date.year)
        
        home_matches = self.get_team_matches(home_team, season)
        if home_matches is None:
            return None
            
        # Look for the specific match
        match_date_str = match_date.strftime('%Y-%m-%d')
        away_team_normalized = self._normalize_team_name(away_team)
        
        match_row = home_matches[
            (home_matches['date'].str.contains(match_date_str)) &
            (home_matches['opponent'].str.contains(away_team_normalized, case=False))
        ]
        
        if match_row.empty:
            logger.warning(f"Could not find xG data for {home_team} vs {away_team} on {match_date_str}")
            return None
        
        match_info = match_row.iloc[0]
        
        return {
            home_team: {
                'xG': match_info['xG_for'],
                'xGA': match_info['xG_against'],
                'npxG': match_info['npxG'],
                'deep_completions': match_info['deep'],
                'PPDA': match_info['ppda']
            },
            away_team: {
                'xG': match_info['xG_against'],
                'xGA': match_info['xG_for'],
                'npxG': None,  # Would need away team data for this
                'deep_completions': None,
                'PPDA': None
            }
        }
    
    def get_player_xg_data(self, team: str, season: str) -> Optional[pd.DataFrame]:
        """Get player xG data for a team."""
        team_normalized = self._normalize_team_name(team)
        url = f"{self.base_url}/team/{team_normalized}/{season}"
        response = self._make_request(url)
        
        if not response:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the script containing player data
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'playersData' in script.string:
                # Extract JSON data
                json_match = re.search(r'JSON\.parse\(\'(.+?)\'\)', script.string)
                if json_match:
                    try:
                        json_str = json_match.group(1).encode().decode('unicode_escape')
                        data = json.loads(json_str)
                        
                        players_data = []
                        for player in data:
                            players_data.append({
                                'player_name': player.get('player_name'),
                                'games': player.get('games'),
                                'time': player.get('time'),
                                'goals': player.get('goals'),
                                'xG': float(player.get('xG', 0)),
                                'assists': player.get('assists'),
                                'xA': float(player.get('xA', 0)),
                                'shots': player.get('shots'),
                                'key_passes': player.get('key_passes'),
                                'yellow_cards': player.get('yellow_cards'),
                                'red_cards': player.get('red_cards'),
                                'position': player.get('position')
                            })
                        
                        return pd.DataFrame(players_data)
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Error parsing player data: {e}")
                        return None
        
        return None
    
    def get_shot_data(self, match_id: str) -> Optional[Dict]:
        """Get detailed shot data for a match."""
        url = f"{self.base_url}/match/{match_id}"
        response = self._make_request(url)
        
        if not response:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the script containing shot data
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'shotsData' in script.string:
                # Extract JSON data
                json_match = re.search(r'JSON\.parse\(\'(.+?)\'\)', script.string)
                if json_match:
                    try:
                        json_str = json_match.group(1).encode().decode('unicode_escape')
                        data = json.loads(json_str)
                        
                        shots_data = {'home': [], 'away': []}
                        
                        for side in ['h', 'a']:
                            side_key = 'home' if side == 'h' else 'away'
                            if side in data:
                                for shot in data[side]:
                                    shots_data[side_key].append({
                                        'minute': shot.get('minute'),
                                        'result': shot.get('result'),
                                        'xG': float(shot.get('xG', 0)),
                                        'player': shot.get('player'),
                                        'x': float(shot.get('X', 0)),
                                        'y': float(shot.get('Y', 0)),
                                        'situation': shot.get('situation'),
                                        'season': shot.get('season'),
                                        'shotType': shot.get('shotType'),
                                        'match_id': shot.get('match_id'),
                                        'player_id': shot.get('player_id')
                                    })
                        
                        return shots_data
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Error parsing shot data: {e}")
                        return None
        
        return None