# Database Configuration
DATABASE_URL=postgresql://quantum:password@localhost:5432/quantum_bot
POSTGRES_USER=quantum
POSTGRES_PASSWORD=password
POSTGRES_DB=quantum_bot

# API Keys
FOOTBALL_DATA_API_KEY=your_football_data_key
UNDERSTAT_API_KEY=your_understat_key
OPENWEATHER_API_KEY=your_weather_key
ODDS_API_KEY=your_odds_key

# Telegram Bot (optional)
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id

# Email Alerts (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Redis (for caching)
REDIS_URL=redis://localhost:6379

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Model Configuration
MAX_MONTE_CARLO_RUNS=10000
CONFIDENCE_THRESHOLD=0.65
MIN_EDGE=0.05