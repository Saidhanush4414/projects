from textblob import TextBlob

def analyze_text(text: str):
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    if score > 0.1:
        label = 'Positive'
    elif score < -0.1:
        label = 'Negative'
    else:
        label = 'Neutral'
    return {'label': label, 'score': float(score)}

def reason_from_text(title: str, content: str):
    title_lower = title.lower()
    if 'tesla' in title_lower:
        return 'Market reacts to Tesla-related developments'
    if 'upgrade' in title_lower or 'update' in title_lower:
        return 'Network improvement drives sentiment'
    if 'congestion' in title_lower or 'issue' in title_lower:
        return 'Technical issues cause caution'
    return 'Headline-driven sentiment'

def aggregate_prediction(items):
    pos = sum(1 for it in items if it['ai']['label'] == 'Positive')
    neg = sum(1 for it in items if it['ai']['label'] == 'Negative')
    if pos > neg:
        return 'Bullish'
    if neg > pos:
        return 'Bearish'
    return 'Neutral'


