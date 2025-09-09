"""
AI Sentiment Analysis for Crypto News
Uses TextBlob for natural language processing and sentiment analysis
"""

from textblob import TextBlob
import re

class SentimentAnalyzer:
    """AI-powered sentiment analysis for cryptocurrency news"""
    
    def __init__(self):
        # Crypto-specific sentiment keywords
        self.positive_keywords = [
            'bullish', 'surge', 'rally', 'breakthrough', 'adoption', 'partnership',
            'upgrade', 'innovation', 'growth', 'profit', 'gain', 'rise', 'increase',
            'success', 'milestone', 'achievement', 'breakthrough', 'launch', 'expansion'
        ]
        
        self.negative_keywords = [
            'bearish', 'crash', 'decline', 'fall', 'drop', 'loss', 'concern',
            'risk', 'volatility', 'uncertainty', 'regulation', 'ban', 'hack',
            'security', 'issue', 'problem', 'delay', 'failure', 'rejection'
        ]
        
        self.crypto_terms = [
            'bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi', 'nft',
            'trading', 'market', 'price', 'volume', 'mining', 'staking'
        ]
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of a given text"""
        # Clean and preprocess text
        text = self._preprocess_text(text)
        
        # Get TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Enhance with crypto-specific keywords
        enhanced_score = self._enhance_with_keywords(text, polarity)
        
        # Determine sentiment category
        if enhanced_score > 0.1:
            sentiment = 'positive'
        elif enhanced_score < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': enhanced_score,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'confidence': abs(enhanced_score)
        }
    
    def _preprocess_text(self, text):
        """Clean and preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _enhance_with_keywords(self, text, base_polarity):
        """Enhance sentiment analysis with crypto-specific keywords"""
        words = text.split()
        
        positive_count = sum(1 for word in words if word in self.positive_keywords)
        negative_count = sum(1 for word in words if word in self.negative_keywords)
        
        # Weight the keyword analysis
        keyword_bias = (positive_count - negative_count) * 0.1
        
        # Combine with TextBlob polarity
        enhanced_score = base_polarity + keyword_bias
        
        # Normalize to [-1, 1] range
        return max(-1, min(1, enhanced_score))
    
    def analyze_news_batch(self, news_items):
        """Analyze sentiment for a batch of news items"""
        analyzed_news = []
        
        for item in news_items:
            # Analyze title and content
            title_sentiment = self.analyze_sentiment(item.get('title', ''))
            content_sentiment = self.analyze_sentiment(item.get('content', ''))
            
            # Combine title and content sentiment (title weighted more)
            combined_score = (title_sentiment['score'] * 0.7 + content_sentiment['score'] * 0.3)
            
            if combined_score > 0.1:
                final_sentiment = 'positive'
            elif combined_score < -0.1:
                final_sentiment = 'negative'
            else:
                final_sentiment = 'neutral'
            
            analyzed_item = item.copy()
            analyzed_item.update({
                'ai_sentiment': final_sentiment,
                'sentiment_score': combined_score,
                'confidence': abs(combined_score),
                'title_sentiment': title_sentiment,
                'content_sentiment': content_sentiment
            })
            
            analyzed_news.append(analyzed_item)
        
        return analyzed_news
    
    def get_market_prediction(self, analyzed_news):
        """Generate market prediction based on news sentiment"""
        if not analyzed_news:
            return 'Neutral'
        
        positive_count = sum(1 for item in analyzed_news if item['ai_sentiment'] == 'positive')
        negative_count = sum(1 for item in analyzed_news if item['ai_sentiment'] == 'negative')
        total_count = len(analyzed_news)
        
        positive_ratio = positive_count / total_count
        negative_ratio = negative_count / total_count
        
        # Determine prediction with confidence
        if positive_ratio > 0.6:
            return 'Bullish'
        elif negative_ratio > 0.6:
            return 'Bearish'
        elif abs(positive_ratio - negative_ratio) < 0.2:
            return 'Neutral'
        else:
            return 'Bullish' if positive_ratio > negative_ratio else 'Bearish'
