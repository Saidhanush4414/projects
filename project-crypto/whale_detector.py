"""
Whale Detection Algorithm for Cryptocurrency Trading
Detects large volume spikes and price movements that indicate whale activity
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class WhaleDetector:
    """Detects whale trading events in cryptocurrency markets"""
    
    def __init__(self):
        self.volume_threshold = 3.0  # Z-score threshold for volume spikes
        self.price_impact_threshold = 0.02  # 2% price impact threshold
        self.lookback_window = 24  # Hours to look back for volume baseline
        
    def detect_events(self):
        """Detect whale events in the market"""
        try:
            # Generate synthetic market data with whale events
            market_data = self._generate_market_data()
            
            # Detect whale events
            whale_events = self._analyze_whale_activity(market_data)
            
            # Generate predictions based on whale activity
            predictions = self._generate_whale_predictions(whale_events)
            
            return {
                'events': whale_events,
                'market_data': market_data,
                'predictions': predictions,
                'detection_summary': self._get_detection_summary(whale_events)
            }
        except Exception as e:
            print(f"Error in whale detection: {e}")
            return {
                'events': [],
                'market_data': [],
                'predictions': [],
                'detection_summary': {
                    'total_events': 0,
                    'buying_events': 0,
                    'selling_events': 0,
                    'high_severity': 0,
                    'medium_severity': 0,
                    'low_severity': 0,
                    'avg_confidence': 0
                }
            }
    
    def _generate_market_data(self):
        """Generate synthetic market data with whale events"""
        # Generate 7 days of hourly data
        hours = 7 * 24
        timestamps = [datetime.now() - timedelta(hours=hours-i) for i in range(hours)]
        
        market_data = []
        base_price = 45000  # Bitcoin base price
        base_volume = 1000
        
        for i, timestamp in enumerate(timestamps):
            # Normal price movement
            price_change = np.random.normal(0, 0.01)
            base_price *= (1 + price_change)
            
            # Normal volume with some randomness
            volume = base_volume * (1 + np.random.normal(0, 0.3))
            
            # Occasionally add whale events
            is_whale = random.random() < 0.05  # 5% chance of whale event
            
            if is_whale:
                # Whale buying or selling
                whale_direction = random.choice(['buy', 'sell'])
                volume_spike = random.uniform(5, 15)  # 5x to 15x volume
                price_impact = random.uniform(0.02, 0.08)  # 2% to 8% price impact
                
                if whale_direction == 'buy':
                    base_price *= (1 + price_impact)
                else:
                    base_price *= (1 - price_impact)
                
                volume *= volume_spike
                
                whale_type = 'buying' if whale_direction == 'buy' else 'selling'
            else:
                whale_type = None
            
            market_data.append({
                'timestamp': timestamp.isoformat(),
                'price': round(base_price, 2),
                'volume': round(volume, 2),
                'whale_type': whale_type,
                'hour': i
            })
        
        return market_data
    
    def _analyze_whale_activity(self, market_data):
        """Analyze market data to detect whale events"""
        df = pd.DataFrame(market_data)
        
        # Calculate volume z-scores
        df['volume_mean'] = df['volume'].rolling(window=self.lookback_window, min_periods=1).mean()
        df['volume_std'] = df['volume'].rolling(window=self.lookback_window, min_periods=1).std()
        df['volume_zscore'] = (df['volume'] - df['volume_mean']) / (df['volume_std'] + 1e-9)
        
        # Calculate price changes
        df['price_change'] = df['price'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Detect whale events
        whale_events = []
        
        for i, row in df.iterrows():
            if (row['volume_zscore'] > self.volume_threshold and 
                row['price_change_abs'] > self.price_impact_threshold):
                
                event_type = 'buying' if row['price_change'] > 0 else 'selling'
                
                whale_event = {
                    'timestamp': row['timestamp'],
                    'price': row['price'],
                    'volume': row['volume'],
                    'volume_zscore': round(row['volume_zscore'], 2),
                    'price_change': round(row['price_change'] * 100, 2),
                    'event_type': event_type,
                    'confidence': min(1.0, (row['volume_zscore'] / 5.0) * (row['price_change_abs'] / 0.05)),
                    'severity': self._get_severity(row['volume_zscore'], row['price_change_abs'])
                }
                
                whale_events.append(whale_event)
        
        return whale_events
    
    def _get_severity(self, volume_zscore, price_impact):
        """Determine severity of whale event"""
        if volume_zscore > 8 and price_impact > 0.05:
            return 'High'
        elif volume_zscore > 5 and price_impact > 0.03:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_detection_summary(self, whale_events):
        """Get summary of whale detection results"""
        if not whale_events:
            return {
                'total_events': 0,
                'buying_events': 0,
                'selling_events': 0,
                'high_severity': 0,
                'medium_severity': 0,
                'low_severity': 0,
                'avg_confidence': 0
            }
        
        buying_events = sum(1 for event in whale_events if event['event_type'] == 'buying')
        selling_events = sum(1 for event in whale_events if event['event_type'] == 'selling')
        
        high_severity = sum(1 for event in whale_events if event['severity'] == 'High')
        medium_severity = sum(1 for event in whale_events if event['severity'] == 'Medium')
        low_severity = sum(1 for event in whale_events if event['severity'] == 'Low')
        
        avg_confidence = sum(event['confidence'] for event in whale_events) / len(whale_events)
        
        return {
            'total_events': len(whale_events),
            'buying_events': buying_events,
            'selling_events': selling_events,
            'high_severity': high_severity,
            'medium_severity': medium_severity,
            'low_severity': low_severity,
            'avg_confidence': round(avg_confidence, 2)
        }
    
    def _generate_whale_predictions(self, whale_events):
        """Generate predictions based on whale activity patterns"""
        if not whale_events:
            return {
                'market_outlook': 'Neutral - No significant whale activity detected',
                'price_prediction': 'Stable price movement expected',
                'volatility_forecast': 'Low volatility expected',
                'trading_recommendation': 'HOLD - Wait for clearer signals'
            }
        
        # Analyze whale activity patterns
        buying_events = [e for e in whale_events if e['event_type'] == 'buying']
        selling_events = [e for e in whale_events if e['event_type'] == 'selling']
        
        # Calculate average confidence and severity
        avg_confidence = sum(e['confidence'] for e in whale_events) / len(whale_events)
        high_severity_count = sum(1 for e in whale_events if e['severity'] == 'High')
        
        # Generate predictions based on patterns
        if len(buying_events) > len(selling_events) * 1.5:
            market_outlook = "Bullish - Whale accumulation detected"
            price_prediction = "Coin price may rise due to whale buying pressure"
            trading_recommendation = "BUY - Follow whale accumulation"
        elif len(selling_events) > len(buying_events) * 1.5:
            market_outlook = "Bearish - Whale distribution detected"
            price_prediction = "Coin price may drop due to whale selling pressure"
            trading_recommendation = "SELL - Avoid whale distribution"
        else:
            market_outlook = "Mixed - Balanced whale activity"
            price_prediction = "Sideways movement expected with increased volatility"
            trading_recommendation = "HOLD - Wait for clearer direction"
        
        # Volatility forecast based on severity
        if high_severity_count > 0:
            volatility_forecast = "High volatility expected due to significant whale activity"
        elif avg_confidence > 0.7:
            volatility_forecast = "Medium volatility expected"
        else:
            volatility_forecast = "Low to medium volatility expected"
        
        return {
            'market_outlook': market_outlook,
            'price_prediction': price_prediction,
            'volatility_forecast': volatility_forecast,
            'trading_recommendation': trading_recommendation,
            'whale_activity_summary': {
                'total_events': len(whale_events),
                'buying_events': len(buying_events),
                'selling_events': len(selling_events),
                'avg_confidence': round(avg_confidence, 2),
                'high_severity_events': high_severity_count
            }
        }
    
    def should_trade(self, whale_event):
        """Determine if we should trade based on whale event"""
        if whale_event['event_type'] == 'buying' and whale_event['confidence'] > 0.6:
            return 'BUY'
        elif whale_event['event_type'] == 'selling' and whale_event['confidence'] > 0.6:
            return 'SELL'
        else:
            return 'HOLD'
