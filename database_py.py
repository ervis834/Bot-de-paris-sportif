"""Database connection and management module."""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
import pandas as pd
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import psycopg2
from config.settings import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self):
        self.engine = create_engine(
            settings.database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=settings.environment == "development"
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.metadata = MetaData()
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a query and return results as list of dictionaries."""
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            return [dict(row._mapping) for row in result]
    
    def execute_insert(self, query: str, params: Optional[Dict] = None) -> None:
        """Execute an insert/update/delete query."""
        with self.get_session() as session:
            session.execute(text(query), params or {})
    
    def bulk_insert(self, table_name: str, data: List[Dict], 
                   on_conflict: str = "DO NOTHING") -> int:
        """Bulk insert data with conflict resolution."""
        if not data:
            return 0
            
        df = pd.DataFrame(data)
        with self.engine.connect() as conn:
            rows_inserted = df.to_sql(
                table_name,
                conn,
                if_exists='append',
                index=False,
                method='multi'
            )
        
        logger.info(f"Inserted {rows_inserted} rows into {table_name}")
        return rows_inserted
    
    def get_dataframe(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute query and return as pandas DataFrame."""
        return pd.read_sql_query(text(query), self.engine, params=params or {})
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in database."""
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = :table_name
        )
        """
        result = self.execute_query(query, {"table_name": table_name})
        return result[0]["exists"]
    
    def get_table_row_count(self, table_name: str) -> int:
        """Get row count for a table."""
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.execute_query(query)
        return result[0]["count"]
    
    def create_schema(self, schema_file: str = "sql/schema.sql") -> None:
        """Create database schema from SQL file."""
        try:
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            with self.engine.connect() as conn:
                conn.execute(text(schema_sql))
                conn.commit()
            
            logger.info("Database schema created successfully")
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            raise
    
    def create_indices(self, indices_file: str = "sql/indices.sql") -> None:
        """Create database indices from SQL file."""
        try:
            with open(indices_file, 'r') as f:
                indices_sql = f.read()
            
            with self.engine.connect() as conn:
                # Split and execute each index creation separately
                statements = [stmt.strip() for stmt in indices_sql.split(';') if stmt.strip()]
                for stmt in statements:
                    try:
                        conn.execute(text(stmt))
                    except Exception as e:
                        logger.warning(f"Index creation failed (might already exist): {e}")
                conn.commit()
            
            logger.info("Database indices created successfully")
        except Exception as e:
            logger.error(f"Error creating indices: {e}")
            raise


class MatchQueries:
    """Pre-defined queries for match data."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def get_upcoming_matches(self, days_ahead: int = 7) -> pd.DataFrame:
        """Get upcoming matches within specified days."""
        query = """
        SELECT 
            m.id, m.api_id, m.match_date,
            ht.name as home_team, at.name as away_team,
            m.league, m.season
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE m.match_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '%s days'
        AND m.status = 'SCHEDULED'
        ORDER BY m.match_date
        """ % days_ahead
        
        return self.db.get_dataframe(query)
    
    def get_team_recent_matches(self, team_id: str, count: int = 10) -> pd.DataFrame:
        """Get recent matches for a team."""
        query = """
        SELECT 
            m.id, m.api_id, m.match_date,
            CASE 
                WHEN m.home_team_id = :team_id THEN 'HOME'
                ELSE 'AWAY'
            END as venue,
            CASE 
                WHEN m.home_team_id = :team_id THEN at.name
                ELSE ht.name
            END as opponent,
            CASE 
                WHEN m.home_team_id = :team_id THEN m.home_score
                ELSE m.away_score
            END as goals_for,
            CASE 
                WHEN m.home_team_id = :team_id THEN m.away_score
                ELSE m.home_score
            END as goals_against,
            m.league, m.season
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE (m.home_team_id = :team_id OR m.away_team_id = :team_id)
        AND m.status = 'FINISHED'
        ORDER BY m.match_date DESC
        LIMIT :count
        """
        
        return self.db.get_dataframe(query, {
            "team_id": team_id,
            "count": count
        })
    
    def get_head_to_head(self, team1_id: str, team2_id: str, count: int = 10) -> pd.DataFrame:
        """Get head-to-head matches between two teams."""
        query = """
        SELECT 
            m.id, m.match_date,
            CASE 
                WHEN m.home_team_id = :team1_id THEN 'HOME'
                ELSE 'AWAY'
            END as team1_venue,
            m.home_score, m.away_score,
            m.league, m.season
        FROM matches m
        WHERE ((m.home_team_id = :team1_id AND m.away_team_id = :team2_id)
            OR (m.home_team_id = :team2_id AND m.away_team_id = :team1_id))
        AND m.status = 'FINISHED'
        ORDER BY m.match_date DESC
        LIMIT :count
        """
        
        return self.db.get_dataframe(query, {
            "team1_id": team1_id,
            "team2_id": team2_id,
            "count": count
        })


class FeatureQueries:
    """Pre-defined queries for feature engineering."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def get_team_form_features(self, team_id: str, as_of_date: str) -> Dict[str, float]:
        """Calculate team form features as of a specific date."""
        query = """
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
                CASE 
                    WHEN m.home_team_id = :team_id THEN 1
                    ELSE 0
                END as is_home
            FROM matches m
            WHERE (m.home_team_id = :team_id OR m.away_team_id = :team_id)
            AND m.status = 'FINISHED'
            AND m.match_date < :as_of_date
            ORDER BY m.match_date DESC
            LIMIT 10
        )
        SELECT 
            AVG(CASE WHEN ROW_NUMBER() OVER (ORDER BY match_date DESC) <= 5 THEN points END) as form_5_games,
            AVG(points) as form_10_games,
            AVG(CASE WHEN is_home = 1 THEN points END) as home_form,
            AVG(CASE WHEN is_home = 0 THEN points END) as away_form,
            AVG(goals_for) as goals_for_avg,
            AVG(goals_against) as goals_against_avg
        FROM recent_matches
        """
        
        result = self.db.execute_query(query, {
            "team_id": team_id,
            "as_of_date": as_of_date
        })
        
        return result[0] if result else {}


# Global database instance
db_manager = DatabaseManager()
match_queries = MatchQueries(db_manager)
feature_queries = FeatureQueries(db_manager)