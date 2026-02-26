"""
Real-time market data ingestion module.
Supports multiple asset classes with fault tolerance and data validation.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import ccxt
import yfinance as yf
from firebase_admin import firestore, initialize_app
from firebase_admin.exceptions import FirebaseError
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Structured market data container"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    asset_class: str
    source: str
    
class DataIngestionEngine:
    """Orchestrates data collection from multiple sources with error handling"""
    
    def __init__(self, config: Any, firestore_client: Optional[firestore.Client] = None):
        self.config = config
        self.firestore_client = firestore_client
        self.ccxt_exchanges: Dict[str, Any] = {}
        self.data_buffer: Dict[str, List[MarketData]] = {}
        self._initialize_exchanges()
        
    def _initialize_exchanges(self) -> None:
        """Initialize exchange connections with error handling"""
        try:
            # Initialize CCXT exchanges for crypto
            self.ccxt_exchanges['binance'] = ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 1200,
                'options