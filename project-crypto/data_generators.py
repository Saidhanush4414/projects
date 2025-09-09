"""
Data Generators for Crypto Portfolio AI Dashboard
Generates synthetic cryptocurrency and news data for offline demonstration
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os

class CryptoDataGenerator:
    """Generates synthetic cryptocurrency price data"""
    
    def __init__(self):
        self.crypto_data = {
            'BTC': {'name': 'Bitcoin', 'price': 45000, 'volatility': 0.02},
            'ETH': {'name': 'Ethereum', 'price': 3200, 'volatility': 0.025},
            'SOL': {'name': 'Solana', 'price': 95, 'volatility': 0.03},
            'ADA': {'name': 'Cardano', 'price': 0.45, 'volatility': 0.035},
            'DOT': {'name': 'Polkadot', 'price': 6.8, 'volatility': 0.04},
            'MATIC': {'name': 'Polygon', 'price': 0.85, 'volatility': 0.045},
            'AVAX': {'name': 'Avalanche', 'price': 25, 'volatility': 0.04},
            'LINK': {'name': 'Chainlink', 'price': 12.5, 'volatility': 0.035},
            'UNI': {'name': 'Uniswap', 'price': 6.2, 'volatility': 0.04},
            'ATOM': {'name': 'Cosmos', 'price': 8.5, 'volatility': 0.03}
        }
        self.price_history = {}
        
    def get_current_prices(self):
        """Get current prices for all cryptocurrencies"""
        prices = []
        for symbol, data in self.crypto_data.items():
            # Add some random fluctuation
            change = random.uniform(-0.05, 0.05)
            current_price = data['price'] * (1 + change)
            
            prices.append({
                'symbol': symbol,
                'name': data['name'],
                'price': round(current_price, 2),
                'change_24h': round(change * 100, 2),
                'volume': random.randint(1000000, 50000000),
                'market_cap': round(current_price * random.randint(1000000, 100000000), 0)
            })
        return prices
    
    def get_price_history(self, symbol, days=30):
        """Generate historical price data for a cryptocurrency"""
        if symbol not in self.crypto_data:
            return []
            
        base_price = self.crypto_data[symbol]['price']
        volatility = self.crypto_data[symbol]['volatility']
        
        # Generate price history using random walk
        prices = [base_price]
        dates = []
        
        for i in range(days):
            # Random walk with drift
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # Ensure positive prices
            
            date = datetime.now() - timedelta(days=days-i)
            dates.append(date.strftime('%Y-%m-%d'))
        
        history = []
        for i, (date, price) in enumerate(zip(dates, prices[1:])):
            # Generate OHLC data
            high = price * (1 + abs(np.random.normal(0, volatility/2)))
            low = price * (1 - abs(np.random.normal(0, volatility/2)))
            open_price = prices[i] if i > 0 else price
            volume = random.randint(100000, 1000000)
            
            history.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'volume': volume
            })
        
        return history
    
    def get_chart_data(self, symbol, days=30):
        """Get chart data formatted for visualization"""
        history = self.get_price_history(symbol, days)
        
        chart_data = {
            'labels': [item['date'] for item in history],
            'datasets': [{
                'label': f'{symbol} Price',
                'data': [item['close'] for item in history],
                'borderColor': '#3b82f6',
                'backgroundColor': 'rgba(59, 130, 246, 0.1)',
                'fill': True
            }]
        }
        
        return chart_data

class NewsDataGenerator:
    """Generates synthetic cryptocurrency news data"""
    
    def __init__(self):
        self.news_templates = {
            'positive': [
                "Major institutional adoption of {coin} announced by leading financial firm",
                "{coin} network upgrade shows promising results with 50% faster transactions",
                "New partnership between {coin} and major tech company drives market optimism",
                "Regulatory clarity boosts {coin} adoption in key markets",
                "Innovative DeFi protocol on {coin} network attracts $100M in TVL"
            ],
            'negative': [
                "Security concerns raised about {coin} network after minor incident",
                "Regulatory uncertainty weighs on {coin} market sentiment",
                "Technical issues cause temporary delays in {coin} network",
                "Market volatility affects {coin} trading volumes",
                "Competition from new blockchain projects impacts {coin} dominance"
            ],
            'neutral': [
                "Weekly market analysis shows mixed signals for {coin}",
                "Technical indicators suggest sideways movement for {coin}",
                "Market experts remain cautious about {coin} short-term outlook",
                "Trading volume analysis reveals normal patterns for {coin}",
                "Network statistics show stable performance for {coin}"
            ]
        }
        
        self.coins = ['Bitcoin', 'Ethereum', 'Solana', 'Cardano', 'Polkadot', 'Polygon']
        
    def generate_news(self, count=10):
        """Generate synthetic news articles"""
        news_items = []
        
        for i in range(count):
            sentiment = random.choice(['positive', 'negative', 'neutral'])
            coin = random.choice(self.coins)
            template = random.choice(self.news_templates[sentiment])
            
            news_item = {
                'id': i + 1,
                'title': template.format(coin=coin),
                'content': f"This is a detailed analysis of the recent developments affecting {coin}. " +
                          f"The market has shown {'positive' if sentiment == 'positive' else 'negative' if sentiment == 'negative' else 'mixed'} " +
                          f"signals in recent trading sessions.",
                'coin': coin,
                'sentiment': sentiment,
                'timestamp': (datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat(),
                'source': random.choice(['CryptoNews', 'Blockchain Daily', 'DeFi Times', 'Market Watch', 'CoinDesk'])
            }
            
            news_items.append(news_item)
        
        return news_items
    
    def get_news_with_sentiment(self):
        """Get news with AI sentiment analysis"""
        news_items = self.generate_news(15)
        
        # Add market prediction based on overall sentiment
        positive_count = sum(1 for item in news_items if item['sentiment'] == 'positive')
        negative_count = sum(1 for item in news_items if item['sentiment'] == 'negative')
        
        if positive_count > negative_count:
            market_prediction = 'Bullish'
        elif negative_count > positive_count:
            market_prediction = 'Bearish'
        else:
            market_prediction = 'Neutral'
        
        return {
            'news': news_items,
            'market_prediction': market_prediction,
            'sentiment_summary': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': len(news_items) - positive_count - negative_count
            }
        }
