"""Weather data collector using OpenWeatherMap API."""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import json
from config.settings import settings

logger = logging.getLogger(__name__)


class WeatherCollector:
    """Collects weather data from OpenWeatherMap API."""
    
    def __init__(self):
        self.api_key = settings.openweather_api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        # Stadium/city coordinates for major football cities
        self.city_coordinates = {
            # Premier League
            'London': (51.5074, -0.1278),
            'Manchester': (53.4808, -2.2426),
            'Liverpool': (53.4084, -2.9916),
            'Birmingham': (52.4862, -1.8904),
            'Newcastle': (54.9783, -1.6178),
            'Brighton': (50.8225, -0.1372),
            'Leicester': (52.6369, -1.1398),
            'Southampton': (50.9097, -1.4044),
            'Leeds': (53.8008, -1.5491),
            'Sheffield': (53.3811, -1.4701),
            'Nottingham': (52.9548, -1.1581),
            
            # La Liga
            'Madrid': (40.4168, -3.7038),
            'Barcelona': (41.3851, 2.1734),
            'Valencia': (39.4699, -0.3763),
            'Seville': (37.3886, -5.9823),
            'Bilbao': (43.2630, -2.9350),
            'San Sebastian': (43.3183, -1.9812),
            'Villarreal': (39.9440, -0.1000),
            
            # Bundesliga
            'Munich': (48.1351, 11.5820),
            'Berlin': (52.5200, 13.4050),
            'Hamburg': (53.5511, 9.9937),
            'Cologne': (50.9375, 6.9603),
            'Frankfurt': (50.1109, 8.6821),
            'Stuttgart': (48.7758, 9.1829),
            'Dortmund': (51.5136, 7.4653),
            'Leipzig': (51.3397, 12.3731),
            
            # Serie A
            'Rome': (41.9028, 12.4964),
            'Milan': (45.4642, 9.1900),
            'Turin': (45.0703, 7.6869),
            'Naples': (40.8518, 14.2681),
            'Florence': (43.7696, 11.2558),
            'Bologna': (44.4949, 11.3426),
            'Genoa': (44.4056, 8.9463),
            
            # Ligue 1
            'Paris': (48.8566, 2.3522),
            'Marseille': (43.2965, 5.3698),
            'Lyon': (45.7640, 4.8357),
            'Lille': (50.6292, 3.0573),
            'Nice': (43.7102, 7.2620),
            'Bordeaux': (44.8378, -0.5792),
            'Nantes': (47.2184, -1.5536),
            'Strasbourg': (48.5734, 7.7521),
        }
    
    def _get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location."""
        # First try direct lookup
        for city, coords in self.city_coordinates.items():
            if city.lower() in location.lower():
                return coords
        
        # If not found, try geocoding API
        try:
            geocode_url = "http://api.openweathermap.org/geo/1.0/direct"
            params = {
                'q': location,
                'limit': 1,
                'appid': self.api_key
            }
            
            response = requests.get(geocode_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data:
                return (data[0]['lat'], data[0]['lon'])
                
        except Exception as e:
            logger.error(f"Error geocoding location {location}: {e}")
        
        return None
    
    def get_current_weather(self, location: str) -> Optional[Dict]:
        """Get current weather for a location."""
        coordinates = self._get_coordinates(location)
        if not coordinates:
            logger.warning(f"Could not find coordinates for {location}")
            return None
        
        lat, lon = coordinates
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'location': location,
                'coordinates': coordinates,
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 0),
                'weather_main': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],
                'cloudiness': data['clouds']['all'],
                'visibility': data.get('visibility', 10000),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching current weather for {location}: {e}")
            return None
    
    def get_weather_forecast(self, location: str, target_date: datetime) -> Optional[Dict]:
        """Get weather forecast for a specific date and location."""
        coordinates = self._get_coordinates(location)
        if not coordinates:
            logger.warning(f"Could not find coordinates for {location}")
            return None
        
        lat, lon = coordinates
        
        # Check if target date is within 5 days (free tier limit)
        days_ahead = (target_date - datetime.now()).days
        
        if days_ahead < 0:
            logger.warning(f"Cannot get forecast for past date: {target_date}")
            return None
        
        if days_ahead > 5:
            logger.warning(f"Forecast only available for next 5 days, requested: {days_ahead} days")
            return None
        
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Find the forecast closest to the target date
            target_timestamp = target_date.timestamp()
            closest_forecast = None
            min_time_diff = float('inf')
            
            for forecast in data['list']:
                forecast_time = datetime.fromtimestamp(forecast['dt'])
                time_diff = abs(forecast_time.timestamp() - target_timestamp)
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_forecast = forecast
            
            if closest_forecast:
                return {
                    'location': location,
                    'coordinates': coordinates,
                    'forecast_date': datetime.fromtimestamp(closest_forecast['dt']),
                    'target_date': target_date,
                    'temperature': closest_forecast['main']['temp'],
                    'feels_like': closest_forecast['main']['feels_like'],
                    'humidity': closest_forecast['main']['humidity'],
                    'pressure': closest_forecast['main']['pressure'],
                    'wind_speed': closest_forecast['wind']['speed'],
                    'wind_direction': closest_forecast['wind'].get('deg', 0),
                    'weather_main': closest_forecast['weather'][0]['main'],
                    'weather_description': closest_forecast['weather'][0]['description'],
                    'cloudiness': closest_forecast['clouds']['all'],
                    'precipitation_probability': closest_forecast.get('pop', 0) * 100,
                    'rain_3h': closest_forecast.get('rain', {}).get('3h', 0),
                    'snow_3h': closest_forecast.get('snow', {}).get('3h', 0),
                    'timestamp': datetime.now()
                }
            
        except Exception as e:
            logger.error(f"Error fetching weather forecast for {location}: {e}")
            return None
    
    def get_historical_weather(self, location: str, date: datetime) -> Optional[Dict]:
        """Get historical weather data (requires paid plan)."""
        coordinates = self._get_coordinates(location)
        if not coordinates:
            return None
        
        lat, lon = coordinates
        
        try:
            # Note: This requires a paid OpenWeatherMap subscription
            url = f"{self.base_url}/onecall/timemachine"
            params = {
                'lat': lat,
                'lon': lon,
                'dt': int(date.timestamp()),
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            current = data.get('current', {})
            
            return {
                'location': location,
                'coordinates': coordinates,
                'date': date,
                'temperature': current.get('temp'),
                'feels_like': current.get('feels_like'),
                'humidity': current.get('humidity'),
                'pressure': current.get('pressure'),
                'wind_speed': current.get('wind_speed'),
                'wind_direction': current.get('wind_deg', 0),
                'weather_main': current.get('weather', [{}])[0].get('main'),
                'weather_description': current.get('weather', [{}])[0].get('description'),
                'cloudiness': current.get('clouds'),
                'visibility': current.get('visibility', 10000),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching historical weather for {location}: {e}")
            return None
    
    def classify_weather_impact(self, weather_data: Dict) -> Dict:
        """Classify weather conditions and their potential impact on football matches."""
        if not weather_data:
            return {}
        
        temperature = weather_data.get('temperature', 20)
        wind_speed = weather_data.get('wind_speed', 0)
        precipitation = weather_data.get('rain_3h', 0) + weather_data.get('snow_3h', 0)
        weather_main = weather_data.get('weather_main', '').lower()
        humidity = weather_data.get('humidity', 50)
        
        impact_score = 0
        factors = []
        
        # Temperature impact
        if temperature < 5:
            impact_score += 2
            factors.append('very_cold')
        elif temperature < 10:
            impact_score += 1
            factors.append('cold')
        elif temperature > 30:
            impact_score += 2
            factors.append('very_hot')
        elif temperature > 25:
            impact_score += 1
            factors.append('hot')
        
        # Wind impact
        if wind_speed > 15:
            impact_score += 3
            factors.append('very_windy')
        elif wind_speed > 10:
            impact_score += 2
            factors.append('windy')
        elif wind_speed > 5:
            impact_score += 1
            factors.append('breezy')
        
        # Precipitation impact
        if precipitation > 5:
            impact_score += 3
            factors.append('heavy_rain')
        elif precipitation > 1:
            impact_score += 2
            factors.append('light_rain')
        elif 'rain' in weather_main or 'drizzle' in weather_main:
            impact_score += 1
            factors.append('drizzle')
        
        if 'snow' in weather_main:
            impact_score += 4
            factors.append('snow')
        
        # Humidity impact (extreme values)
        if humidity > 85:
            impact_score += 1
            factors.append('high_humidity')
        elif humidity < 30:
            impact_score += 1
            factors.append('low_humidity')
        
        # Weather condition impacts
        if weather_main in ['thunderstorm', 'tornado']:
            impact_score += 5
            factors.append('severe_weather')
        elif weather_main in ['fog', 'mist']:
            impact_score += 2
            factors.append('poor_visibility')
        
        # Classify overall impact
        if impact_score >= 8:
            impact_level = 'severe'
        elif impact_score >= 5:
            impact_level = 'high'
        elif impact_score >= 3:
            impact_level = 'moderate'
        elif impact_score >= 1:
            impact_level = 'low'
        else:
            impact_level = 'minimal'
        
        return {
            'impact_score': impact_score,
            'impact_level': impact_level,
            'factors': factors,
            'playing_conditions': self._get_playing_conditions(weather_data),
            'recommendations': self._get_weather_recommendations(impact_level, factors)
        }
    
    def _get_playing_conditions(self, weather_data: Dict) -> Dict:
        """Get detailed playing conditions based on weather."""
        temperature = weather_data.get('temperature', 20)
        wind_speed = weather_data.get('wind_speed', 0)
        precipitation = weather_data.get('rain_3h', 0) + weather_data.get('snow_3h', 0)
        
        return {
            'ball_behavior': {
                'affected_by_wind': wind_speed > 8,
                'wet_conditions': precipitation > 0.5,
                'temperature_impact': 'low' if temperature < 10 else 'normal'
            },
            'pitch_conditions': {
                'slippery': precipitation > 1 or weather_data.get('weather_main', '').lower() in ['rain', 'drizzle'],
                'hard': temperature < 0,
                'muddy': precipitation > 2
            },
            'player_comfort': {
                'temperature_stress': temperature < 5 or temperature > 28,
                'breathing_difficulty': weather_data.get('humidity', 50) > 80 and temperature > 25,
                'visibility_issues': weather_data.get('weather_main', '').lower() in ['fog', 'mist', 'snow']
            }
        }
    
    def _get_weather_recommendations(self, impact_level: str, factors: List[str]) -> List[str]:
        """Get betting recommendations based on weather conditions."""
        recommendations = []
        
        if impact_level in ['severe', 'high']:
            recommendations.append('Consider Under 2.5 Goals - Poor weather often leads to lower-scoring games')
            
        if 'heavy_rain' in factors or 'snow' in factors:
            recommendations.append('Favor defensive teams - Wet conditions benefit organized defenses')
            recommendations.append('Avoid over-based bets - Difficult playing conditions')
            
        if 'very_windy' in factors:
            recommendations.append('Consider goalkeeping errors - Wind affects ball trajectory')
            recommendations.append('Fade long-shot specialists - Wind impacts long-range efforts')
            
        if 'very_cold' in factors:
            recommendations.append('Watch for muscle injuries - Cold weather increases injury risk')
            recommendations.append('Consider slow starts - Players need time to warm up')
            
        if 'very_hot' in factors:
            recommendations.append('Expect more substitutions - Heat affects stamina')
            recommendations.append('Consider second-half unders - Players tire in heat')
            
        if impact_level == 'minimal':
            recommendations.append('Weather unlikely to significantly impact match outcome')
        
        return recommendations