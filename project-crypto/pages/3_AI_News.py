import streamlit as st
from utils.data import get_news
from utils.sentiment import analyze_text, reason_from_text, aggregate_prediction

st.set_page_config(page_title="AI News", page_icon="📰", layout="wide")
st.title("📰 AI News Monitoring")
st.caption("Offline news dataset with AI sentiment analysis and aggregated prediction.")

news = get_news()
analyzed = []
for item in news:
    ai = analyze_text(item['title'] + '. ' + item.get('content',''))
    reason = reason_from_text(item['title'], item.get('content',''))
    analyzed.append({**item, 'ai': ai, 'reason': reason})

prediction = aggregate_prediction(analyzed)

st.subheader("Market Prediction")
st.metric("Prediction", prediction)

pos = sum(1 for a in analyzed if a['ai']['label'] == 'Positive')
neg = sum(1 for a in analyzed if a['ai']['label'] == 'Negative')
neu = sum(1 for a in analyzed if a['ai']['label'] == 'Neutral')
st.write(f"Positive: {pos} | Negative: {neg} | Neutral: {neu}")

st.divider()
st.subheader("Headlines")
for a in analyzed:
    with st.container(border=True):
        st.markdown(f"**{a['title']}**")
        st.caption(a.get('content',''))
        st.write(f"Sentiment: {a['ai']['label']} (score {a['ai']['score']:.2f})")
        st.write(f"Reason: {a['reason']}")


