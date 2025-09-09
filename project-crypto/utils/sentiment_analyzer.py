import re
from typing import Dict, List, Tuple
from textblob import TextBlob


class NewsSentimentAnalyzer:
    """Analyzes news sentiment for cryptocurrency predictions"""
    
    def __init__(self):
        # Keywords that indicate positive sentiment
        self.positive_keywords = [
            'rise', 'surge', 'bullish', 'gain', 'increase', 'up', 'moon', 'pump',
            'breakthrough', 'adoption', 'partnership', 'upgrade', 'breakout',
            'rally', 'soar', 'explode', 'boom', 'success', 'growth', 'profit',
            'investment', 'buy', 'purchase', 'hodl', 'diamond hands'
        ]
        
        # Keywords that indicate negative sentiment
        self.negative_keywords = [
            'fall', 'drop', 'bearish', 'decline', 'down', 'crash', 'dump',
            'correction', 'sell', 'loss', 'bear', 'dip', 'plunge', 'tank',
            'volatility', 'risk', 'concern', 'warning', 'bubble', 'overvalued',
            'sell-off', 'panic', 'fear', 'uncertainty', 'regulation'
        ]
        
        # Coin-specific positive indicators
        self.coin_positive_patterns = {
            'bitcoin': ['bitcoin', 'btc', 'halving', 'institutional', 'etf'],
            'ethereum': ['ethereum', 'eth', 'defi', 'smart contract', 'upgrade'],
            'cardano': ['cardano', 'ada', 'staking', 'governance'],
            'solana': ['solana', 'sol', 'fast', 'scalable', 'defi'],
            'ripple': ['ripple', 'xrp', 'payment', 'banking', 'partnership'],
            'polkadot': ['polkadot', 'dot', 'parachain', 'interoperability']
        }
    
    def analyze_news_sentiment(self, title: str, content: str = "") -> Dict[str, float]:
        """Analyze sentiment of news article"""
        text = f"{title} {content}".lower()
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity  # -1 to 1
        
        # Keyword-based sentiment
        positive_count = sum(1 for word in self.positive_keywords if word in text)
        negative_count = sum(1 for word in self.negative_keywords if word in text)
        
        keyword_sentiment = 0
        if positive_count + negative_count > 0:
            keyword_sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        
        # Combined sentiment (weighted average)
        combined_sentiment = (textblob_sentiment * 0.6) + (keyword_sentiment * 0.4)
        
        return {
            'sentiment_score': combined_sentiment,
            'textblob_score': textblob_sentiment,
            'keyword_score': keyword_sentiment,
            'positive_keywords': positive_count,
            'negative_keywords': negative_count
        }
    
    def analyze_coin_sentiment(self, coin_symbol: str, news_items: List[Dict]) -> Dict[str, any]:
        """Analyze sentiment for a specific coin across multiple news items"""
        coin_news = [item for item in news_items if self._is_coin_related(coin_symbol, item)]
        
        if not coin_news:
            return {
                'coin': coin_symbol,
                'sentiment_score': 0.0,
                'prediction': 'neutral',
                'confidence': 0.0,
                'news_count': 0,
                'trend': 'stable'
            }
        
        # Analyze each news item
        sentiments = []
        for news in coin_news:
            sentiment = self.analyze_news_sentiment(news.get('title', ''), news.get('content', ''))
            sentiments.append(sentiment['sentiment_score'])
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Determine prediction
        if avg_sentiment > 0.1:
            prediction = 'bullish'
            trend = 'rising'
        elif avg_sentiment < -0.1:
            prediction = 'bearish'
            trend = 'falling'
        else:
            prediction = 'neutral'
            trend = 'stable'
        
        # Calculate confidence based on sentiment variance
        variance = sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)
        confidence = max(0, 1 - variance)  # Higher confidence with lower variance
        
        return {
            'coin': coin_symbol,
            'sentiment_score': avg_sentiment,
            'prediction': prediction,
            'confidence': confidence,
            'news_count': len(coin_news),
            'trend': trend,
            'news_items': coin_news[:5]  # Top 5 news items
        }
    
    def _is_coin_related(self, coin_symbol: str, news_item: Dict) -> bool:
        """Check if news item is related to specific coin"""
        coin_lower = coin_symbol.lower()
        title = (news_item.get('title', '') or '').lower()
        coins_field = (news_item.get('coins', '') or '').lower()
        
        # Check exact match in coins field first (most reliable)
        if coins_field:
            # Split by common delimiters and check for exact matches
            coins_list = re.split(r'[;,|]', coins_field)
            for coin in coins_list:
                if coin.strip().lower() == coin_lower:
                    return True
        
        # Check in title for exact symbol match
        if coin_lower in title:
            return True
        
        # Check coin-specific patterns for broader matching
        patterns = self.coin_positive_patterns.get(coin_lower, [])
        for pattern in patterns:
            if pattern in title or pattern in coins_field:
                return True
        
        return False
    
    def get_top_coins_analysis(self, news_items: List[Dict], top_n: int = 6) -> List[Dict]:
        """Get sentiment analysis for top coins with most news coverage"""
        # Count news per coin
        coin_counts = {}
        for news in news_items:
            coins_field = news.get('coins', '')
            if coins_field:
                # Split by common delimiters
                coins = re.split(r'[;,|]', coins_field)
                for coin in coins:
                    coin = coin.strip().lower()
                    if coin and coin != 'market-wide':
                        coin_counts[coin] = coin_counts.get(coin, 0) + 1
        
        # Get top coins by news count
        top_coins = sorted(coin_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Analyze sentiment for each top coin
        analyses = []
        for coin_symbol, count in top_coins:
            analysis = self.analyze_coin_sentiment(coin_symbol, news_items)
            analyses.append(analysis)
        
        return sorted(analyses, key=lambda x: abs(x['sentiment_score']), reverse=True)
