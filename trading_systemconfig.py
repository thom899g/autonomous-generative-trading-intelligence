"""
Configuration management for the Autonomous Generative Trading Intelligence system.
Centralized configuration with environment variable support and validation.
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Firebase configuration (CRITICAL - Ecosystem requirement)
FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY", ""),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", ""),
    "projectId": os.getenv("FIREBASE_PROJECT_ID", ""),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", ""),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", ""),
    "appId": os.getenv("FIREBASE_APP_ID", ""),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL", "")
}

@dataclass
class TradingConfig:
    """Centralized configuration for trading system"""
    
    # Data ingestion settings
    data_sources: Dict[str, str] = field(default_factory=lambda: {
        "crypto": "ccxt",
        "stocks": "yfinance",
        "forex": "yfinance"
    })
    
    # Feature engineering
    technical_indicators: list = field(default_factory=lambda: [
        "sma_20", "sma_50", "ema_12", "ema_26",
        "rsi_14", "macd", "bollinger_upper", "bollinger_lower",
        "atr_14", "obv"
    ])
    
    # Model parameters
    pattern_recognition_window: int = 60  # 60-minute windows for pattern detection
    lstm_units: int = 128
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    
    # Reinforcement learning
    rl_episodes: int = 1000
    rl_gamma: float = 0.99
    rl_epsilon: float = 0.1
    rl_memory_size: int = 10000
    
    # Risk management
    max_position_size: float = 0.1  # Max 10% of portfolio per trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    
    # Execution settings
    paper_trading: bool = True  # Start with paper trading
    trade_slippage: float = 0.001  # 0.1% slippage assumption
    
    # Real-time adaptation
    adaptation_frequency_minutes: int = 15
    model_retrain_threshold: float = 0.85  # Retrain if accuracy drops below 85%

def validate_config(config: TradingConfig) -> bool:
    """Validate configuration parameters"""
    try:
        assert config.max_position_size > 0 and config.max_position_size <= 1.0, "Invalid position size"
        assert config.stop_loss_pct > 0, "Stop loss must be positive"
        assert config.take_profit_pct > config.stop_loss_pct, "Take profit must exceed stop loss"
        assert config.pattern_recognition_window >= 30, "Window too small for meaningful patterns"
        assert config.rl_episodes >= 100, "Insufficient RL episodes"
        return True
    except AssertionError as e:
        logging.error(f"Configuration validation failed: {e}")
        return False

def load_environment_config() -> TradingConfig:
    """Load configuration from environment variables with defaults"""
    config = TradingConfig()
    
    # Override with environment variables if present
    if os.getenv("PAPER_TRADING"):
        config.paper_trading = os.getenv("PAPER_TRADING").lower() == "true"
    
    if os.getenv("MAX_POSITION_SIZE"):
        config.max_position_size = float(os.getenv("MAX_POSITION_SIZE"))
    
    if os.getenv("MODEL_RETRAIN_THRESHOLD"):
        config.model_retrain_threshold = float(os.getenv("MODEL_RETRAIN_THRESHOLD"))
    
    return config