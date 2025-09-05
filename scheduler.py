"""Automated scheduling for Bot Quantum Max operations."""

import logging
import schedule
import time
import threading
from datetime import datetime, timedelta
import subprocess
import sys
import os

from config.settings import settings, LOGGING_CONFIG
from src.data.etl import run_daily_etl
from src.features.engineering import update_features_table
from src.ops.alerts import AlertManager

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class QuantumScheduler:
    """Main scheduler for automated operations."""
    
    def __init__(self):
        self.alert_manager = AlertManager()
        self.running = False
        self.scheduler_thread = None
        
    def start(self):
        """Start the scheduler."""
        logger.info("Starting Quantum Bot Scheduler")
        
        # Schedule daily ETL at 6 AM
        schedule.every().day.at("06:00").do(self._run_daily_etl)
        
        # Schedule feature updates at 7 AM (after ETL)
        schedule.every().day.at("07:00").do(self._update_features)
        
        # Schedule model training weekly on Sundays at 8 AM
        schedule.every().sunday.at("08:00").do(self._run_model_training)
        
        # Schedule daily predictions at 9 AM
        schedule.every().day.at("09:00").do(self._generate_predictions)
        
        # Schedule hourly odds updates during peak hours
        for hour in [10, 12, 14, 16, 18, 20]:
            schedule.every().day.at(f"{hour:02d}:00").do(self._update_odds)
        
        # Schedule portfolio rebalancing at 10 AM
        schedule.every().day.at("10:00").do(self._rebalance_portfolio)
        
        # Schedule performance monitoring every 2 hours
        schedule.every(2).hours.do(self._monitor_performance)
        
        # Schedule system health checks every 30 minutes
        schedule.every(30).minutes.do(self._health_check)
        
        # Start scheduler in separate thread
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("Scheduler started successfully")
    
    def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping Quantum Bot Scheduler")
        self.running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        schedule.clear()
        logger.info("Scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _run_daily_etl(self):
        """Run daily ETL process."""
        logger.info("Starting scheduled daily ETL")
        
        try:
            run_daily_etl()
            self.alert_manager.send_info_alert(
                "ETL Completed",
                "Daily ETL process completed successfully"
            )
        except Exception as e:
            logger.error(f"Daily ETL failed: {e}")
            self.alert_manager.send_critical_alert(
                "ETL Failed",
                f"Daily ETL process failed: {str(e)}"
            )
    
    def _update_features(self):
        """Update features table."""
        logger.info("Starting scheduled feature update")
        
        try:
            update_features_table()
            self.alert_manager.send_info_alert(
                "Features Updated",
                "Feature engineering completed successfully"
            )
        except Exception as e:
            logger.error(f"Feature update failed: {e}")
            self.alert_manager.send_warning_alert(
                "Feature Update Failed",
                f"Feature update failed: {str(e)}"
            )
    
    def _run_model_training(self):
        """Run model training."""
        logger.info("Starting scheduled model training")
        
        try:
            # Run training script
            result = subprocess.run([
                sys.executable, "train.py", "--skip-etl"
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                self.alert_manager.send_info_alert(
                    "Model Training Completed",
                    "Weekly model training completed successfully"
                )
            else:
                self.alert_manager.send_critical_alert(
                    "Model Training Failed",
                    f"Training failed with error: {result.stderr}"
                )
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self.alert_manager.send_critical_alert(
                "Model Training Error",
                f"Training process error: {str(e)}"
            )
    
    def _generate_predictions(self):
        """Generate daily predictions."""
        logger.info("Starting scheduled prediction generation")
        
        try:
            # Run prediction script
            result = subprocess.run([
                sys.executable, "predict_today.py"
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                # Parse output for high-confidence predictions
                output_lines = result.stdout.split('\n')
                high_conf_count = 0
                for line in output_lines:
                    if "HIGH CONFIDENCE" in line:
                        high_conf_count += 1
                
                self.alert_manager.send_info_alert(
                    "Predictions Generated",
                    f"Daily predictions completed. {high_conf_count} high-confidence predictions found."
                )
            else:
                self.alert_manager.send_warning_alert(
                    "Prediction Generation Failed",
                    f"Prediction failed: {result.stderr}"
                )
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            self.alert_manager.send_warning_alert(
                "Prediction Error",
                f"Prediction process error: {str(e)}"
            )
    
    def _update_odds(self):
        """Update odds data."""
        logger.info("Starting scheduled odds update")
        
        try:
            from src.data.collectors.odds_api import OddsAPICollector
            from src.data.database import db_manager
            
            collector = OddsAPICollector()
            
            # Get today's matches
            query = """
            SELECT m.id, ht.name as home_team, at.name as away_team, m.match_date
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE DATE(m.match_date) = CURRENT_DATE
            AND m.status = 'SCHEDULED'
            """
            
            matches = db_manager.execute_query(query)
            odds_updated = 0
            
            for match in matches:
                try:
                    odds_data = collector.get_match_odds(
                        match['home_team'],
                        match['away_team'],
                        match['match_date']
                    )
                    
                    if odds_data:
                        # Store odds in database
                        self._store_odds_data(match['id'], odds_data)
                        odds_updated += 1
                
                except Exception as e:
                    logger.warning(f"Failed to update odds for match {match['id']}: {e}")
                    continue
            
            logger.info(f"Updated odds for {odds_updated} matches")
            
        except Exception as e:
            logger.error(f"Odds update failed: {e}")
    
    def _store_odds_data(self, match_id: str, odds_data: dict):
        """Store odds data in database."""
        from src.data.database import db_manager
        import uuid
        
        for bookmaker, markets in odds_data.items():
            for market_type, odds in markets.items():
                query = """
                INSERT INTO odds (id, match_id, bookmaker, odds_date, market_type, odds_data)
                VALUES (:id, :match_id, :bookmaker, :odds_date, :market_type, :odds_data)
                ON CONFLICT (match_id, bookmaker, market_type, odds_date)
                DO UPDATE SET odds_data = EXCLUDED.odds_data
                """
                
                db_manager.execute_insert(query, {
                    'id': str(uuid.uuid4()),
                    'match_id': match_id,
                    'bookmaker': bookmaker,
                    'odds_date': datetime.now(),
                    'market_type': market_type,
                    'odds_data': odds
                })
    
    def _rebalance_portfolio(self):
        """Rebalance betting portfolio."""
        logger.info("Starting portfolio rebalancing")
        
        try:
            from src.portfolio.optimizer import PortfolioOptimizer
            from src.data.database import db_manager
            
            optimizer = PortfolioOptimizer()
            
            # Get current portfolio status
            portfolio_query = """
            SELECT total_bankroll, roi, max_drawdown 
            FROM portfolio_performance 
            ORDER BY date DESC 
            LIMIT 1
            """
            
            portfolio_status = db_manager.execute_query(portfolio_query)
            
            if portfolio_status:
                current_roi = portfolio_status[0]['roi']
                max_drawdown = portfolio_status[0]['max_drawdown']
                
                # Check if rebalancing is needed
                if abs(max_drawdown) > 0.1:  # 10% drawdown
                    self.alert_manager.send_warning_alert(
                        "High Drawdown Detected",
                        f"Current drawdown: {max_drawdown:.1%}. Consider reducing position sizes."
                    )
                
                if current_roi < -0.05:  # -5% ROI
                    self.alert_manager.send_warning_alert(
                        "Negative ROI Alert",
                        f"Current ROI: {current_roi:.1%}. Portfolio review recommended."
                    )
            
        except Exception as e:
            logger.error(f"Portfolio rebalancing failed: {e}")
    
    def _monitor_performance(self):
        """Monitor system and model performance."""
        logger.info("Running performance monitoring")
        
        try:
            from src.data.database import db_manager
            
            # Check recent model performance
            performance_query = """
            SELECT model_name, accuracy, f1_score, evaluation_date
            FROM model_performance
            WHERE evaluation_date >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY evaluation_date DESC
            """
            
            recent_performance = db_manager.execute_query(performance_query)
            
            if recent_performance:
                for perf in recent_performance:
                    if perf['accuracy'] < 0.5:  # Below 50% accuracy
                        self.alert_manager.send_warning_alert(
                            "Model Performance Alert",
                            f"Model {perf['model_name']} accuracy dropped to {perf['accuracy']:.1%}"
                        )
            
            # Check database size and performance
            db_size_query = """
            SELECT 
                schemaname, 
                tablename, 
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            LIMIT 5
            """
            
            db_sizes = db_manager.execute_query(db_size_query)
            logger.info(f"Top 5 table sizes: {db_sizes}")
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
    
    def _health_check(self):
        """Perform system health check."""
        try:
            from src.data.database import db_manager
            
            # Test database connection
            db_manager.execute_query("SELECT 1")
            
            # Check recent data updates
            recent_matches_query = """
            SELECT COUNT(*) as count
            FROM matches 
            WHERE updated_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
            """
            
            recent_matches = db_manager.execute_query(recent_matches_query)
            
            if recent_matches and recent_matches[0]['count'] == 0:
                self.alert_manager.send_warning_alert(
                    "Data Staleness Alert",
                    "No matches updated in the last 24 hours"
                )
            
            # Check disk space
            disk_usage = self._check_disk_usage()
            if disk_usage > 85:  # Over 85% full
                self.alert_manager.send_warning_alert(
                    "Disk Space Alert",
                    f"Disk usage at {disk_usage}%"
                )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.alert_manager.send_critical_alert(
                "System Health Check Failed",
                f"Health check error: {str(e)}"
            )
    
    def _check_disk_usage(self) -> float:
        """Check disk usage percentage."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            return (used / total) * 100
        except:
            return 0.0
    
    def manual_trigger(self, task_name: str):
        """Manually trigger a scheduled task."""
        logger.info(f"Manually triggering task: {task_name}")
        
        task_map = {
            'etl': self._run_daily_etl,
            'features': self._update_features,
            'training': self._run_model_training,
            'predictions': self._generate_predictions,
            'odds': self._update_odds,
            'portfolio': self._rebalance_portfolio,
            'monitoring': self._monitor_performance,
            'health': self._health_check
        }
        
        if task_name in task_map:
            try:
                task_map[task_name]()
                logger.info(f"Manual task {task_name} completed successfully")
            except Exception as e:
                logger.error(f"Manual task {task_name} failed: {e}")
                raise
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def get_schedule_status(self) -> dict:
        """Get current schedule status."""
        jobs = schedule.get_jobs()
        
        status = {
            'scheduler_running': self.running,
            'total_jobs': len(jobs),
            'jobs': []
        }
        
        for job in jobs:
            status['jobs'].append({
                'job': str(job.job_func),
                'next_run': job.next_run.isoformat() if job.next_run else None,
                'interval': str(job.interval)
            })
        
        return status


def main():
    """Main function for running scheduler as standalone process."""
    scheduler = QuantumScheduler()
    
    try:
        scheduler.start()
        logger.info("Scheduler is running. Press Ctrl+C to stop.")
        
        # Keep the main thread alive
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Scheduler failed: {e}")
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                self.alert_manager.send_critical_alert(
                    "Scheduler Error",
                    f"Scheduler encountered error: {str(e)}"
                )
                time.sleep(