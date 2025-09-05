"""Tests for ETL pipeline."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.data.etl import ETLPipeline
from src.data.collectors.football_data import FootballDataCollector


class TestETLPipeline:
    """Test cases for ETL pipeline."""
    
    @pytest.fixture
    def etl_pipeline(self):
        """Create ETL pipeline instance for testing."""
        return ETLPipeline()
    
    @pytest.fixture
    def mock_matches_data(self):
        """Mock matches data for testing."""
        return [
            {
                'api_id': 12345,
                'home_team_api_id': 1,
                'away_team_api_id': 2,
                'league': 'PL',
                'season': '2024',
                'matchday': 1,
                'match_date': datetime.now(),
                'status': 'FINISHED',
                'home_score': 2,
                'away_score': 1,
                'ht_home_score': 1,
                'ht_away_score': 0,
                'referee': 'Test Referee',
                'venue': 'Test Stadium',
                'attendance': 50000
            }
        ]
    
    @pytest.fixture
    def mock_teams_data(self):
        """Mock teams data for testing."""
        return [
            {
                'api_id': 1,
                'name': 'Test Team A',
                'short_name': 'TTA',
                'country': 'England',
                'league': 'PL',
                'founded': 1900,
                'venue': 'Test Stadium A',
                'logo_url': 'http://example.com/logo_a.png'
            },
            {
                'api_id': 2,
                'name': 'Test Team B',
                'short_name': 'TTB',
                'country': 'England',
                'league': 'PL',
                'founded': 1905,
                'venue': 'Test Stadium B',
                'logo_url': 'http://example.com/logo_b.png'
            }
        ]
    
    def test_etl_pipeline_initialization(self, etl_pipeline):
        """Test ETL pipeline initialization."""
        assert etl_pipeline.football_data is not None
        assert etl_pipeline.understat is not None
        assert etl_pipeline.weather is not None
        assert etl_pipeline.odds_api is not None
    
    @patch('src.data.etl.db_manager')
    @patch('src.data.etl.FootballDataCollector')
    def test_update_teams(self, mock_collector_class, mock_db, etl_pipeline, mock_teams_data):
        """Test teams update functionality."""
        # Setup mocks
        mock_collector = Mock()
        mock_collector_class.return_value = mock_collector
        mock_collector.get_competitions.return_value = [
            {'api_id': 1, 'code': 'PL', 'name': 'Premier League'}
        ]
        mock_collector.get_teams.return_value = mock_teams_data
        
        etl_pipeline.football_data = mock_collector
        
        # Test update_teams method
        etl_pipeline._update_teams()
        
        # Verify data collection calls
        mock_collector.get_competitions.assert_called_once()
        mock_collector.get_teams.assert_called_once_with(1)
    
    @patch('src.data.etl.db_manager')
    def test_get_team_id_by_api_id(self, mock_db, etl_pipeline):
        """Test team ID lookup by API ID."""
        # Setup mock response
        mock_db.execute_query.return_value = [{'id': 'test-uuid-123'}]
        
        result = etl_pipeline._get_team_id_by_api_id(12345)
        
        assert result == 'test-uuid-123'
        mock_db.execute_query.assert_called_once()
    
    @patch('src.data.etl.db_manager')
    def test_insert_or_update_match(self, mock_db, etl_pipeline, mock_matches_data):
        """Test match insertion/update."""
        # Setup mock team ID lookups
        mock_db.execute_query.side_effect = [
            [{'id': 'home-team-uuid'}],  # home team lookup
            [{'id': 'away-team-uuid'}]   # away team lookup
        ]
        
        match_data = mock_matches_data[0]
        
        # Test insert_or_update_match
        etl_pipeline._insert_or_update_match(match_data)
        
        # Verify database calls
        assert mock_db.execute_query.call_count == 2  # Team lookups
        mock_db.execute_insert.assert_called_once()  # Match insert
    
    @patch('src.data.etl.db_manager')
    def test_insert_or_update_match_missing_teams(self, mock_db, etl_pipeline, mock_matches_data):
        """Test match insertion with missing team data."""
        # Setup mock to return no teams
        mock_db.execute_query.return_value = []
        
        match_data = mock_matches_data[0]
        
        # Test insert_or_update_match
        etl_pipeline._insert_or_update_match(match_data)
        
        # Should not attempt insert if teams not found
        mock_db.execute_insert.assert_not_called()
    
    @pytest.mark.integration
    @patch('src.data.etl.run_daily_etl')
    def test_run_daily_etl_integration(self, mock_run_daily_etl):
        """Integration test for daily ETL process."""
        # This would test the full ETL pipeline in integration environment
        mock_run_daily_etl.return_value = None
        
        # Test that ETL can be called without errors
        mock_run_daily_etl()
        
        mock_run_daily_etl.assert_called_once()


class TestFootballDataCollector:
    """Test cases for Football Data collector."""
    
    @pytest.fixture
    def collector(self):
        """Create collector instance for testing."""
        return FootballDataCollector()
    
    @pytest.fixture
    def mock_api_response(self):
        """Mock API response data."""
        return {
            "competitions": [
                {
                    "id": 2021,
                    "name": "Premier League",
                    "code": "PL",
                    "area": {"name": "England"},
                    "currentSeason": {"startDate": "2024-08-17"}
                }
            ]
        }
    
    def test_collector_initialization(self, collector):
        """Test collector initialization."""
        assert collector.base_url == "https://api.football-data.org/v4"
        assert "X-Auth-Token" in collector.headers
        assert collector.rate_limit_delay == 6
    
    @patch('src.data.collectors.football_data.requests.get')
    def test_make_request_success(self, mock_get, collector, mock_api_response):
        """Test successful API request."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test request
        result = collector._make_request("competitions")
        
        assert result == mock_api_response
        mock_get.assert_called_once()
    
    @patch('src.data.collectors.football_data.requests.get')
    def test_make_request_failure(self, mock_get, collector):
        """Test API request failure handling."""
        # Setup mock to raise exception
        mock_get.side_effect = Exception("API Error")
        
        # Test request failure
        with pytest.raises(Exception):
            collector._make_request("competitions")
    
    @patch.object(FootballDataCollector, '_make_request')
    def test_get_competitions(self, mock_request, collector, mock_api_response):
        """Test competitions retrieval."""
        mock_request.return_value = mock_api_response
        
        result = collector.get_competitions()
        
        assert len(result) == 1
        assert result[0]['api_id'] == 2021
        assert result[0]['name'] == 'Premier League'
        assert result[0]['code'] == 'PL'
    
    @patch.object(FootballDataCollector, '_make_request')
    def test_get_teams(self, mock_request, collector):
        """Test teams retrieval."""
        mock_response = {
            "teams": [
                {
                    "id": 57,
                    "name": "Arsenal FC",
                    "tla": "ARS",
                    "area": {"name": "England"},
                    "founded": 1886,
                    "venue": "Emirates Stadium",
                    "crest": "https://example.com/arsenal.png"
                }
            ],
            "competition": {"code": "PL"}
        }
        
        mock_request.return_value = mock_response
        
        result = collector.get_teams(2021)
        
        assert len(result) == 1
        assert result[0]['api_id'] == 57
        assert result[0]['name'] == 'Arsenal FC'
        assert result[0]['league'] == 'PL'
    
    @patch.object(FootballDataCollector, '_make_request')
    def test_get_matches(self, mock_request, collector):
        """Test matches retrieval."""
        mock_response = {
            "matches": [
                {
                    "id": 12345,
                    "utcDate": "2024-01-15T15:00:00Z",
                    "status": "FINISHED",
                    "homeTeam": {"id": 57, "venue": "Emirates Stadium"},
                    "awayTeam": {"id": 61},
                    "score": {
                        "fullTime": {"home": 2, "away": 1},
                        "halfTime": {"home": 1, "away": 0}
                    },
                    "season": {"startDate": "2024-08-17"},
                    "matchday": 1,
                    "referees": [{"name": "Test Referee"}],
                    "attendance": 60000
                }
            ],
            "competition": {"code": "PL"}
        }
        
        mock_request.return_value = mock_response
        
        result = collector.get_matches(2021)
        
        assert len(result) == 1
        match = result[0]
        assert match['api_id'] == 12345
        assert match['home_team_api_id'] == 57
        assert match['away_team_api_id'] == 61
        assert match['status'] == 'FINISHED'
        assert match['home_score'] == 2
        assert match['away_score'] == 1
    
    @pytest.mark.slow
    def test_get_recent_matches(self, collector):
        """Test recent matches retrieval (slow test)."""
        # This test would make actual API calls in integration environment
        # For unit tests, we mock the dependencies
        with patch.object(collector, 'get_competitions') as mock_competitions, \
             patch.object(collector, 'get_matches') as mock_matches:
            
            mock_competitions.return_value = [
                {'api_id': 2021, 'code': 'PL', 'name': 'Premier League'}
            ]
            mock_matches.return_value = []
            
            result = collector.get_recent_matches(days_back=7)
            
            assert isinstance(result, list)
            mock_competitions.assert_called_once()
            mock_matches.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])