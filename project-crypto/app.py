"""
Crypto Portfolio AI Dashboard - Flask Backend
A comprehensive cryptocurrency analytics platform with AI-powered features
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import random
from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore")

# Import our custom modules
try:
    from data_generators import CryptoDataGenerator, NewsDataGenerator
    from whale_detector import WhaleDetector
    from trading_simulator import TradingSimulator
    from sentiment_analyzer import SentimentAnalyzer
    # Use real dataset loaders when available
    from utils.data import (
        get_coins as load_coins,
        get_ohlcv as load_ohlcv,
        get_news as load_news,
        expand_news_to_count,
        filter_news_by_coin,
    )
    from utils.sentiment_analyzer import NewsSentimentAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all Python files are in the same directory as app.py")
    exit(1)

app = Flask(__name__)

# Initialize data generators and AI components
crypto_gen = CryptoDataGenerator()
news_gen = NewsDataGenerator()
whale_detector = WhaleDetector()
trading_sim = TradingSimulator()
sentiment_analyzer = SentimentAnalyzer()
news_sentiment_analyzer = NewsSentimentAnalyzer()

# Global portfolio state
portfolio_state = {
    'balance': 10000.0,
    'positions': {},
    'equity_history': [],
    'trades': [],
    'realized_profit': 0.0
}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/coins')
def coins_page():
    """All Coins Analytics Page"""
    return render_template('coins.html')

@app.route('/coins/<symbol>')
def coin_details_page(symbol):
    """Coin Details Page"""
    return render_template('coin_detail.html', symbol=symbol.upper())

@app.route('/icon/<path:filename>')
def serve_coin_icon(filename):
    """Serve coin SVG icons from the project 'icon' directory"""
    return send_from_directory('icon', filename)

@app.route('/news')
def news_page():
    """AI News Monitoring Page"""
    return render_template('news.html')

@app.route('/whale')
def whale_page():
    """Whale Detection & Trading Page"""
    return render_template('whale.html')

# API Endpoints

@app.route('/api/crypto/prices')
def get_crypto_prices():
    """Get current prices for all major cryptocurrencies from dataset if available"""
    try:
        coins = load_coins()
        if coins:
            # Ensure consistent response shape with optional change_24h
            normalized = []
            for c in coins:
                normalized.append({
                    'symbol': c.get('symbol'),
                    'name': c.get('name'),
                    'price': c.get('price'),
                    'change_24h': c.get('change_24h', 0.0),
                    'volume': c.get('volume'),
                    'market_cap': c.get('market_cap'),
                    'logo': c.get('logo')
                })
            return jsonify(normalized)
    except Exception as e:
        print(f"Failed to load coins from dataset: {e}")

    # Fallback to synthetic data
    prices = crypto_gen.get_current_prices()
    return jsonify(prices)

@app.route('/api/crypto/history/<symbol>')
def get_crypto_history(symbol):
    """Get historical price data for a specific cryptocurrency"""
    days = request.args.get('days', 30, type=int)
    try:
        history = load_ohlcv(symbol, days)
        # Truncate to last `days` entries if dataset is longer
        if isinstance(history, list) and len(history) > days:
            history = history[-days:]
        return jsonify(history)
    except Exception as e:
        print(f"Failed to load OHLCV for {symbol} from dataset: {e}")
        history = crypto_gen.get_price_history(symbol, days)
        return jsonify(history)

@app.route('/api/crypto/chart/<symbol>')
def get_crypto_chart(symbol):
    """Get chart data for a specific cryptocurrency"""
    days = request.args.get('days', 30, type=int)
    try:
        history = load_ohlcv(symbol, days)
        if isinstance(history, list) and len(history) > days:
            history = history[-days:]
        chart_data = {
            'labels': [item.get('date') for item in history],
            'datasets': [{
                'label': f'{symbol.upper()} Price',
                'data': [item.get('close') for item in history],
                'borderColor': '#3b82f6',
                'backgroundColor': 'rgba(59, 130, 246, 0.1)',
                'fill': True
            }]
        }
        return jsonify(chart_data)
    except Exception as e:
        print(f"Failed to build chart from dataset for {symbol}: {e}")
        chart_data = crypto_gen.get_chart_data(symbol, days)
        return jsonify(chart_data)

@app.route('/api/news')
def get_news():
    """Get crypto news with AI sentiment analysis and coin predictions"""
    coin = request.args.get('coin')
    try:
        seed = load_news()
        items = filter_news_by_coin(seed, coin) if coin else seed
        items = expand_news_to_count(items or seed, target=500)
        
        # Analyze sentiment for top coins
        top_coins_analysis = news_sentiment_analyzer.get_top_coins_analysis(items, top_n=6)
        
        # Map to frontend format with sentiment analysis
        mapped = []
        for it in items:
            # Analyze sentiment for this news item
            sentiment_analysis = news_sentiment_analyzer.analyze_news_sentiment(
                it.get('title', ''), it.get('content', '')
            )
            
            # Determine sentiment label
            if sentiment_analysis['sentiment_score'] > 0.1:
                sentiment = 'positive'
            elif sentiment_analysis['sentiment_score'] < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            mapped.append({
                'id': it.get('id'),
                'title': it.get('title'),
                'content': it.get('title'),
                'coin': (coin or (it.get('coins') or '').split(';')[0].split(',')[0]).strip() or 'Market-wide',
                'sentiment': sentiment,
                'sentiment_score': sentiment_analysis['sentiment_score'],
                'timestamp': it.get('date'),
                'source': it.get('source'),
            })

        # Generate prediction based on top coins analysis
        if top_coins_analysis:
            bullish_coins = [c for c in top_coins_analysis if c['prediction'] == 'bullish']
            bearish_coins = [c for c in top_coins_analysis if c['prediction'] == 'bearish']
            
            if bullish_coins and bearish_coins:
                prediction_text = f"AI Analysis: {', '.join([c['coin'].upper() for c in bullish_coins[:3]])} showing bullish trends, {', '.join([c['coin'].upper() for c in bearish_coins[:2]])} showing bearish trends."
            elif bullish_coins:
                prediction_text = f"AI Analysis: {', '.join([c['coin'].upper() for c in bullish_coins[:3]])} showing strong bullish momentum."
            elif bearish_coins:
                prediction_text = f"AI Analysis: {', '.join([c['coin'].upper() for c in bearish_coins[:3]])} showing bearish pressure."
            else:
                prediction_text = "AI Analysis: Market showing mixed signals with neutral sentiment."
        else:
            prediction_text = "AI Analysis: Insufficient data for comprehensive analysis."

        # Count sentiments
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for item in mapped:
            sentiment_counts[item['sentiment']] += 1

        resp = {
            'news': mapped,
            'market_prediction': prediction_text,
            'top_coins_analysis': top_coins_analysis,
            'sentiment_summary': sentiment_counts
        }
        return jsonify(resp)
    except Exception as e:
        print(f"Failed to load news: {e}")
        news_data = news_gen.get_news_with_sentiment()
        return jsonify(news_data)

@app.route('/api/whale/detect')
def detect_whale_events():
    """Detect whale events in the market"""
    events = whale_detector.detect_events()
    return jsonify(events)

@app.route('/api/whale/simulate')
def simulate_trading():
    """Run trading simulation based on whale detection"""
    global portfolio_state
    # Optional params: coin symbol and invest amount
    coin = request.args.get('coin', default='BTC', type=str)
    amount = request.args.get('amount', default=None, type=float)
    sim_params = {'coin': coin}
    if amount is not None:
        sim_params['amount'] = amount
    result = trading_sim.run_simulation(portfolio_state, sim_params)
    portfolio_state = result['portfolio_state']
    return jsonify(result)

@app.route('/api/portfolio/status')
def get_portfolio_status():
    """Get current portfolio status"""
    return jsonify(portfolio_state)

@app.route('/api/portfolio/reset')
def reset_portfolio():
    """Reset portfolio to initial state"""
    global portfolio_state
    portfolio_state = {
        'balance': 10000.0,
        'positions': {},
        'equity_history': [],
        'trades': [],
        'realized_profit': 0.0
    }
    return jsonify({'status': 'reset', 'portfolio': portfolio_state})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 Starting Crypto Portfolio AI Dashboard...")
    print("📊 Available pages:")
    print("   - http://localhost:5000/ (Dashboard)")
    print("   - http://localhost:5000/coins (Crypto Analytics)")
    print("   - http://localhost:5000/news (AI News Monitoring)")
    print("   - http://localhost:5000/whale (Whale Detection & Trading)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
