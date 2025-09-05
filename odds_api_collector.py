"""Odds API collector for betting odds data."""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
from config.settings import settings

logger = logging.getLogger(__name__)


class OddsAPICollector:
    """Collects betting odds from The Odds API."""
    
    def __init__(self):
        self.api_key = settings.odds_api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.rate_limit_delay = 1.0  # seconds between requests
        
        # Sport keys for different leagues
        self.sport_keys = {
            'PL': 'soccer_epl',
            'FL1': 'soccer_france_ligue_one', 
            'BL1': 'soccer_germany_bundesliga',
            'SA': 'soccer_italy_serie_a',
            'PD': 'soccer_spain_la_liga',
            'CL': 'soccer_uefa_champs_league',
            'EL': 'soccer_uefa_europa_league'
        }
        
        # Market mappings
        self.markets = [
            'h2h',           # Head to head (1X2)
            'spreads',       # Handicap
            'totals',        # Over/Under
            'btts',          # Both teams to score
            'draw_no_bet'    # Draw no bet
        ]
        
        # Bookmaker preferences (ordered by reliability/limits)
        self.preferred_bookmakers = [
            'pinnacle',
            'bet365',
            'betfair',
            'william_hill',
            'ladbrokes',
            'coral',
            'paddy_power',
            'sky_bet',
            'unibet'
        ]
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request with rate limiting."""
        url = f"{self.base_url}/{endpoint}"
        
        if not params:
            params = {}
        params['apiKey'] = self.api_key
        
        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Check remaining requests
            remaining = response.headers.get('x-requests-remaining')
            if remaining and int(remaining) < 50:
                logger.warning(f"Low API quota remaining: {remaining}")
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from {endpoint}: {e}")
            return None
    
    def get_sports(self) -> Optional[List[Dict]]:
        """Get available sports."""
        data = self._make_request("sports")
        if data:
            return [sport for sport in data if sport['key'] in self.sport_keys.values()]
        return None
    
    def get_odds_for_league(self, league_code: str, markets: Optional[List[str]] = None, 
                           bookmakers: Optional[List[str]] = None) -> Optional[List[Dict]]:
        """Get odds for all upcoming matches in a league."""
        sport_key = self.sport_keys.get(league_code)
        if not sport_key:
            logger.error(f"Unknown league code: {league_code}")
            return None
        
        if not markets:
            markets = self.markets
        
        if not bookmakers:
            bookmakers = self.preferred_bookmakers
        
        params = {
            'regions': 'uk,us,eu',
            'markets': ','.join(markets),
            'bookmakers': ','.join(bookmakers),
            'oddsFormat': 'decimal',
            'dateFormat': 'iso'
        }
        
        return self._make_request(f"sports/{sport_key}/odds", params)
    
    def get_match_odds(self, home_team: str, away_team: str, 
                      match_date: datetime) -> Dict[str, Dict]:
        """Get odds for a specific match from all available bookmakers."""
        all_odds = {}
        
        # Try to find the match in all supported leagues
        for league_code in self.sport_keys.keys():
            try:
                league_odds = self.get_odds_for_league(league_code)
                if not league_odds:
                    continue
                    
                # Look for the specific match
                for match in league_odds:
                    match_time = datetime.fromisoformat(
                        match['commence_time'].replace('Z', '+00:00')
                    )
                    
                    # Check if this is the right match (within 2 hours)
                    time_diff = abs((match_time - match_date).total_seconds())
                    if time_diff > 7200:  # 2 hours
                        continue
                        
                    # Check team names (fuzzy matching)
                    if (self._team_name_match(match['home_team'], home_team) and 
                        self._team_name_match(match['away_team'], away_team)):
                        
                        # Process odds for each bookmaker
                        for bookmaker in match.get('bookmakers', []):
                            bookmaker_name = bookmaker['key']
                            
                            if bookmaker_name not in all_odds:
                                all_odds[bookmaker_name] = {}
                            
                            # Process each market
                            for market in bookmaker.get('markets', []):
                                market_key = market['key']
                                all_odds[bookmaker_name][market_key] = self._process_market_odds(market)
                        
                        break
                        
            except Exception as e:
                logger.error(f"Error processing odds for {league_code}: {e}")
                continue
        
        return all_odds
    
    def _team_name_match(self, api_name: str, target_name: str) -> bool:
        """Check if team names match (with fuzzy logic)."""
        api_clean = api_name.lower().replace(' ', '').replace('-', '').replace('_', '')
        target_clean = target_name.lower().replace(' ', '').replace('-', '').replace('_', '')
        
        # Direct match
        if api_clean == target_clean:
            return True
            
        # Contains match
        if api_clean in target_clean or target_clean in api_clean:
            return True
            
        # Common abbreviations and variations
        team_variations = {
            'manchester united': ['man utd', 'manutd', 'mufc'],
            'manchester city': ['man city', 'mancity', 'mcfc'],
            'tottenham hotspur': ['tottenham', 'spurs', 'thfc'],
            'crystal palace': ['palace', 'cpfc'],
            'west ham united': ['west ham', 'whu'],
            'brighton & hove albion': ['brighton', 'bhfc'],
            'newcastle united': ['newcastle', 'nufc'],
            'nottingham forest': ['notts forest', 'nffc'],
            'sheffield united': ['sheffield utd', 'sufc'],
            'wolverhampton wanderers': ['wolves', 'wwfc'],
            'leicester city': ['leicester', 'lcfc']
        }
        
        for full_name, variations in team_variations.items():
            if (full_name.replace(' ', '') in target_clean and 
                any(var.replace(' ', '') in api_clean for var in variations)):
                return True
            if (full_name.replace(' ', '') in api_clean and 
                any(var.replace(' ', '') in target_clean for var in variations)):
                return True
        
        return False
    
    def _process_market_odds(self, market: Dict) -> Dict:
        """Process odds for a specific market."""
        market_key = market['key']
        outcomes = market.get('outcomes', [])
        
        if market_key == 'h2h':
            # Head-to-head (1X2)
            odds_dict = {}
            for outcome in outcomes:
                if outcome['name'] == 'Draw':
                    odds_dict['draw'] = float(outcome['price'])
                else:
                    # Determine if home or away based on position
                    if len(odds_dict) == 0:
                        odds_dict['home'] = float(outcome['price'])
                    elif 'home' in odds_dict and 'draw' not in odds_dict:
                        odds_dict['away'] = float(outcome['price'])
                    elif 'draw' in odds_dict:
                        odds_dict['away'] = float(outcome['price'])
            return odds_dict
            
        elif market_key == 'totals':
            # Over/Under
            odds_dict = {}
            for outcome in outcomes:
                point = outcome.get('point', 2.5)
                name = outcome['name'].lower()
                key = f"{name}_{point}".replace('.', '_')
                odds_dict[key] = float(outcome['price'])
            return odds_dict
            
        elif market_key == 'btts':
            # Both teams to score
            odds_dict = {}
            for outcome in outcomes:
                name = outcome['name'].lower().replace(' ', '_')
                odds_dict[name] = float(outcome['price'])
            return odds_dict
            
        elif market_key == 'spreads':
            # Handicap
            odds_dict = {}
            for outcome in outcomes:
                point = outcome.get('point', 0)
                name = outcome['name'].lower().replace(' ', '_')
                key = f"{name}_{abs(point)}".replace('.', '_')
                odds_dict[key] = float(outcome['price'])
            return odds_dict
            
        else:
            # Generic processing
            odds_dict = {}
            for outcome in outcomes:
                name = outcome['name'].lower().replace(' ', '_')
                odds_dict[name] = float(outcome['price'])
            return odds_dict
    
    def get_historical_odds(self, sport_key: str, event_id: str) -> Optional[Dict]:
        """Get historical odds for a completed match."""
        # Note: Historical odds require a higher tier subscription
        endpoint = f"sports/{sport_key}/events/{event_id}/odds/history"
        
        params = {
            'regions': 'uk,us,eu',
            'markets': ','.join(self.markets),
            'bookmakers': ','.join(self.preferred_bookmakers),
            'oddsFormat': 'decimal',
            'dateFormat': 'iso'
        }
        
        return self._make_request(endpoint, params)
    
    def calculate_implied_probability(self, odds: float) -> float:
        """Calculate implied probability from decimal odds."""
        return 1.0 / odds if odds > 0 else 0.0
    
    def calculate_vig(self, odds_dict: Dict) -> float:
        """Calculate bookmaker's vigorish (overround)."""
        total_prob = sum(self.calculate_implied_probability(odds) 
                        for odds in odds_dict.values())
        return max(0, total_prob - 1.0)
    
    def find_arbitrage_opportunities(self, match_odds: Dict[str, Dict]) -> List[Dict]:
        """Find arbitrage opportunities across bookmakers."""
        opportunities = []
        
        # For each market type
        market_types = set()
        for bookmaker_odds in match_odds.values():
            market_types.update(bookmaker_odds.keys())
        
        for market_type in market_types:
            # Get all outcome types for this market
            outcome_types = set()
            for bookmaker_odds in match_odds.values():
                if market_type in bookmaker_odds:
                    outcome_types.update(bookmaker_odds[market_type].keys())
            
            if len(outcome_types) < 2:
                continue
                
            # Find best odds for each outcome
            best_odds = {}
            for outcome in outcome_types:
                best_odd = 0
                best_bookmaker = None
                
                for bookmaker, bookmaker_odds in match_odds.items():
                    if (market_type in bookmaker_odds and 
                        outcome in bookmaker_odds[market_type]):
                        odd = bookmaker_odds[market_type][outcome]
                        if odd > best_odd:
                            best_odd = odd
                            best_bookmaker = bookmaker
                
                if best_bookmaker:
                    best_odds[outcome] = {
                        'odds': best_odd,
                        'bookmaker': best_bookmaker
                    }
            
            # Check if arbitrage exists
            if len(best_odds) >= 2:
                total_inv_odds = sum(1.0 / data['odds'] for data in best_odds.values())
                
                if total_inv_odds < 1.0:  # Arbitrage opportunity
                    profit_margin = (1.0 - total_inv_odds) * 100
                    opportunities.append({
                        'market_type': market_type,
                        'profit_margin': profit_margin,
                        'bets': best_odds,
                        'total_stake_ratio': total_inv_odds
                    })
        
        return opportunities
    
    def get_odds_movement(self, sport_key: str, hours_back: int = 24) -> Optional[List[Dict]]:
        """Get odds movement data for recent matches."""
        # This would require storing historical odds data
        # For now, return current odds with timestamp
        current_odds = self.get_odds_for_league(sport_key.split('_')[-1])
        
        if current_odds:
            for match in current_odds:
                match['timestamp'] = datetime.now().isoformat()
        
        return current_odds