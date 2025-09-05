"""Alert and notification system for Bot Quantum Max."""

import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, List
import json

from config.settings import settings

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages alerts and notifications across multiple channels."""
    
    def __init__(self):
        self.email_enabled = bool(settings.email_user and settings.email_password)
        self.telegram_enabled = bool(settings.telegram_bot_token and settings.telegram_chat_id)
        
        # Alert levels
        self.levels = {
            'INFO': {'color': '🟢', 'priority': 1},
            'WARNING': {'color': '🟡', 'priority': 2},
            'ERROR': {'color': '🔴', 'priority': 3},
            'CRITICAL': {'color': '🚨', 'priority': 4}
        }
        
        # Rate limiting to avoid spam
        self.last_alerts = {}
        self.rate_limit_seconds = 300  # 5 minutes
    
    def send_info_alert(self, subject: str, message: str, channels: List[str] = None):
        """Send informational alert."""
        self._send_alert('INFO', subject, message, channels)
    
    def send_warning_alert(self, subject: str, message: str, channels: List[str] = None):
        """Send warning alert."""
        self._send_alert('WARNING', subject, message, channels)
    
    def send_error_alert(self, subject: str, message: str, channels: List[str] = None):
        """Send error alert."""
        self._send_alert('ERROR', subject, message, channels)
    
    def send_critical_alert(self, subject: str, message: str, channels: List[str] = None):
        """Send critical alert."""
        self._send_alert('CRITICAL', subject, message, channels)
    
    def _send_alert(self, level: str, subject: str, message: str, channels: List[str] = None):
        """Send alert through specified channels."""
        # Rate limiting check
        alert_key = f"{level}_{subject}"
        current_time = datetime.now()
        
        if alert_key in self.last_alerts:
            time_diff = (current_time - self.last_alerts[alert_key]).total_seconds()
            if time_diff < self.rate_limit_seconds:
                logger.debug(f"Alert rate limited: {alert_key}")
                return
        
        self.last_alerts[alert_key] = current_time
        
        # Default channels based on alert level
        if channels is None:
            if level in ['CRITICAL', 'ERROR']:
                channels = ['email', 'telegram']
            elif level == 'WARNING':
                channels = ['telegram']
            else:
                channels = ['telegram']
        
        # Format message
        formatted_message = self._format_message(level, subject, message)
        
        # Send through each channel
        for channel in channels:
            try:
                if channel == 'email' and self.email_enabled:
                    self._send_email_alert(subject, formatted_message, level)
                elif channel == 'telegram' and self.telegram_enabled:
                    self._send_telegram_alert(subject, formatted_message, level)
                elif channel == 'log':
                    self._log_alert(level, subject, message)
            except Exception as e:
                logger.error(f"Failed to send {channel} alert: {e}")
    
    def _format_message(self, level: str, subject: str, message: str) -> str:
        """Format alert message with timestamp and metadata."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_info = self.levels.get(level, {'color': '⚪', 'priority': 0})
        
        formatted = f"""
{level_info['color']} **Bot Quantum Max Alert**

**Level:** {level}
**Subject:** {subject}
**Time:** {timestamp}
**Environment:** {settings.environment}

**Message:**
{message}

---
Bot Quantum Max Alert System
        """.strip()
        
        return formatted
    
    def _send_email_alert(self, subject: str, message: str, level: str):
        """Send email alert."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = settings.email_user
            msg['To'] = settings.email_user  # Send to self or configure recipients
            msg['Subject'] = f"[{level}] Bot Quantum Max: {subject}"
            
            # Convert markdown to HTML for better formatting
            html_message = self._markdown_to_html(message)
            msg.attach(MIMEText(html_message, 'html'))
            
            # Send email
            with smtplib.SMTP(settings.smtp_server, settings.smtp_port) as server:
                server.starttls()
                server.login(settings.email_user, settings.email_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            raise
    
    def _send_telegram_alert(self, subject: str, message: str, level: str):
        """Send Telegram alert."""
        try:
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            
            # Telegram message formatting
            telegram_message = f"*{level}*: {subject}\n\n{message}"
            
            payload = {
                'chat_id': settings.telegram_chat_id,
                'text': telegram_message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, data=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Telegram alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            raise
    
    def _log_alert(self, level: str, subject: str, message: str):
        """Log alert to application logs."""
        log_message = f"ALERT - {subject}: {message}"
        
        if level == 'CRITICAL':
            logger.critical(log_message)
        elif level == 'ERROR':
            logger.error(log_message)
        elif level == 'WARNING':
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _markdown_to_html(self, text: str) -> str:
        """Convert simple markdown to HTML."""
        # Simple conversions
        html = text.replace('**', '<strong>').replace('**', '</strong>')
        html = html.replace('*', '<em>').replace('*', '</em>')
        html = html.replace('\n', '<br>')
        
        # Wrap in HTML structure
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
                {html}
            </div>
        </body>
        </html>
        """
    
    def send_prediction_alerts(self, predictions: List[dict]):
        """Send alerts for high-confidence predictions."""
        high_confidence_preds = [
            pred for pred in predictions
            if pred.get('confidence', 0) >= settings.confidence_threshold
        ]
        
        if not high_confidence_preds:
            return
        
        # Format predictions for alert
        pred_text = []
        for pred in high_confidence_preds[:5]:  # Limit to top 5
            match_name = f"{pred.get('home_team', 'Home')} vs {pred.get('away_team', 'Away')}"
            outcome = pred.get('predicted_outcome', 'Unknown')
            confidence = pred.get('confidence', 0)
            
            pred_text.append(f"• {match_name}: {outcome} ({confidence:.1%})")
        
        message = f"""
Found {len(high_confidence_preds)} high-confidence predictions:

{chr(10).join(pred_text)}

Check the dashboard for full details and betting recommendations.
        """.strip()
        
        self.send_info_alert(
            f"{len(high_confidence_preds)} High-Confidence Predictions",
            message,
            channels=['telegram']
        )
    
    def send_arbitrage_alert(self, opportunities: List[dict]):
        """Send alert for arbitrage opportunities."""
        if not opportunities:
            return
        
        arb_text = []
        for opp in opportunities[:3]:  # Top 3 opportunities
            profit_margin = opp.get('profit_margin', 0)
            market_type = opp.get('market_type', 'Unknown')
            match_info = opp.get('match_info', {})
            match_name = f"{match_info.get('home_team', 'Home')} vs {match_info.get('away_team', 'Away')}"
            
            arb_text.append(f"• {match_name} ({market_type}): {profit_margin:.2%} profit")
        
        message = f"""
🚨 ARBITRAGE OPPORTUNITIES DETECTED 🚨

{chr(10).join(arb_text)}

Act quickly as these opportunities may disappear fast!
        """.strip()
        
        self.send_warning_alert(
            f"{len(opportunities)} Arbitrage Opportunities",
            message,
            channels=['telegram', 'email']
        )
    
    def send_portfolio_alert(self, portfolio_metrics: dict):
        """Send portfolio performance alerts."""
        roi = portfolio_metrics.get('roi', 0)
        drawdown = portfolio_metrics.get('max_drawdown', 0)
        sharpe = portfolio_metrics.get('sharpe_ratio', 0)
        
        # Alert conditions
        alerts_to_send = []
        
        if roi < -0.1:  # -10% ROI
            alerts_to_send.append(('ERROR', 'Severe Portfolio Decline', f'ROI: {roi:.1%}'))
        elif roi < -0.05:  # -5% ROI
            alerts_to_send.append(('WARNING', 'Portfolio Decline', f'ROI: {roi:.1%}'))
        elif roi > 0.2:  # +20% ROI
            alerts_to_send.append(('INFO', 'Strong Portfolio Performance', f'ROI: {roi:.1%}'))
        
        if abs(drawdown) > 0.15:  # 15% drawdown
            alerts_to_send.append(('ERROR', 'High Drawdown Alert', f'Drawdown: {drawdown:.1%}'))
        elif abs(drawdown) > 0.1:  # 10% drawdown
            alerts_to_send.append(('WARNING', 'Moderate Drawdown', f'Drawdown: {drawdown:.1%}'))
        
        if sharpe < 0.5:  # Poor risk-adjusted returns
            alerts_to_send.append(('WARNING', 'Poor Risk-Adjusted Returns', f'Sharpe: {sharpe:.2f}'))
        
        # Send alerts
        for level, subject, detail in alerts_to_send:
            message = f"""
Portfolio Performance Alert:

Current Metrics:
• ROI: {roi:.1%}
• Max Drawdown: {drawdown:.1%}
• Sharpe Ratio: {sharpe:.2f}
• Win Rate: {portfolio_metrics.get('win_rate', 0):.1%}

{detail}

Review your position sizes and betting strategy.
            """.strip()
            
            if level == 'ERROR':
                self.send_error_alert(subject, message)
            elif level == 'WARNING':
                self.send_warning_alert(subject, message)
            else:
                self.send_info_alert(subject, message)
    
    def send_model_drift_alert(self, model_name: str, current_accuracy: float, 
                              historical_accuracy: float):
        """Send alert for model performance drift."""
        drift_percentage = (current_accuracy - historical_accuracy) / historical_accuracy
        
        if abs(drift_percentage) > 0.1:  # 10% drift
            level = 'ERROR' if drift_percentage < -0.1 else 'WARNING'
            
            message = f"""
Model Drift Detected:

Model: {model_name}
Current Accuracy: {current_accuracy:.1%}
Historical Average: {historical_accuracy:.1%}
Drift: {drift_percentage:.1%}

{'Performance has significantly degraded. Consider retraining.' if drift_percentage < 0 else 'Performance has improved significantly.'}
            """.strip()
            
            if level == 'ERROR':
                self.send_error_alert(f"Model Drift: {model_name}", message)
            else:
                self.send_warning_alert(f"Model Drift: {model_name}", message)
    
    def send_system_health_alert(self, health_metrics: dict):
        """Send system health alerts."""
        issues = []
        
        # Check various health metrics
        if health_metrics.get('disk_usage', 0) > 85:
            issues.append(f"High disk usage: {health_metrics['disk_usage']:.1f}%")
        
        if health_metrics.get('memory_usage', 0) > 85:
            issues.append(f"High memory usage: {health_metrics['memory_usage']:.1f}%")
        
        if not health_metrics.get('database_connected', True):
            issues.append("Database connection failed")
        
        if health_metrics.get('api_failures', 0) > 5:
            issues.append(f"Multiple API failures: {health_metrics['api_failures']}")
        
        if issues:
            message = f"""
System Health Issues Detected:

{chr(10).join(f'• {issue}' for issue in issues)}

Please investigate and resolve these issues promptly.
            """.strip()
            
            level = 'CRITICAL' if len(issues) > 2 else 'WARNING'
            
            if level == 'CRITICAL':
                self.send_critical_alert("System Health Critical", message)
            else:
                self.send_warning_alert("System Health Warning", message)
    
    def test_alerts(self):
        """Test all alert channels."""
        test_subject = "Test Alert - Bot Quantum Max"
        test_message = """
This is a test alert to verify that the notification system is working correctly.

If you receive this message, the alert system is functioning properly.

Test conducted at: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Test each channel
        channels_tested = []
        
        if self.email_enabled:
            try:
                self._send_email_alert(test_subject, test_message, 'INFO')
                channels_tested.append('Email ✅')
            except Exception as e:
                channels_tested.append(f'Email ❌ ({e})')
        
        if self.telegram_enabled:
            try:
                self._send_telegram_alert(test_subject, test_message, 'INFO')
                channels_tested.append('Telegram ✅')
            except Exception as e:
                channels_tested.append(f'Telegram ❌ ({e})')
        
        # Log test results
        logger.info(f"Alert test completed: {', '.join(channels_tested)}")
        
        return channels_tested


# Global alert manager instance
alert_manager = AlertManager()