# 🚀 Crypto Portfolio AI Dashboard

A comprehensive cryptocurrency analytics platform that demonstrates AI + Data Science capabilities for cryptocurrency market analysis, whale detection, and sentiment analysis.

## 🌟 Features

### 📊 All Coins Page (Crypto Analytics Dashboard)
- **Real-time cryptocurrency tracking** for major coins (Bitcoin, Ethereum, Solana, etc.)
- **Interactive price charts** using Chart.js with 30-day historical data
- **Market data visualization** including volume, market cap, and price changes
- **Responsive grid layout** with hover effects and modern UI

### 🤖 AI News Monitoring Page
- **AI-powered sentiment analysis** using TextBlob and custom NLP algorithms
- **Real-time news processing** with positive/negative/neutral classification
- **Market prediction indicators** (Bullish/Bearish/Neutral) based on news sentiment
- **Crypto-specific keyword enhancement** for improved accuracy

### 🐋 Whale Detection & Trading Page
- **Advanced whale detection algorithms** that identify large volume spikes and price movements
- **AI trading simulation** with buy/sell/hold logic based on whale events
- **Portfolio management** with real-time balance tracking and performance metrics
- **Equity curve visualization** showing portfolio growth over time

## 🛠️ Technology Stack

### Backend
- **Python Flask** - Web framework
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **TextBlob** - Natural language processing and sentiment analysis

### Frontend
- **HTML5 & CSS3** - Modern web standards
- **Tailwind CSS** - Utility-first CSS framework
- **JavaScript (ES6+)** - Interactive functionality
- **Chart.js** - Data visualization and charts
- **Font Awesome** - Icons and visual elements

### AI & Data Science
- **Synthetic Data Generation** - Offline-compatible demo data
- **Sentiment Analysis** - NLP for news classification
- **Whale Detection** - Volume and price impact analysis
- **Trading Simulation** - AI-powered trading strategies

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd crypto-portfolio-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app (recommended)**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
crypto-portfolio-ai/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── data_generators.py    # Synthetic data generation
├── sentiment_analyzer.py # AI sentiment analysis
├── whale_detector.py     # Whale detection algorithms
├── trading_simulator.py  # Trading simulation logic
├── templates/            # HTML templates
│   ├── base.html        # Base template
│   ├── index.html       # Dashboard page
│   ├── coins.html       # Crypto analytics page
│   ├── news.html        # AI news monitoring page
│   └── whale.html       # Whale detection page
└── static/              # Static assets
├── streamlit_app.py      # Streamlit home + floating chatbot
├── pages/                # Streamlit pages
│   ├── 1_All_Coins.py
│   ├── 2_Coin_Analytics.py
│   ├── 3_AI_News.py
│   └── 4_Whales_Auto_Trading.py
├── utils/                # Shared utils for Streamlit
│   ├── data.py
│   └── sentiment.py
├── data/                 # Offline datasets
│   ├── coins.json
│   └── news.json
└── assets/
    └── coins/           # Coin logos (add pngs: btc.png, eth.png, ...)
    ├── css/
    │   └── style.css    # Custom styles
    ├── js/
    │   └── main.js      # Main JavaScript
    └── data/            # Data files
```

## 🎯 Key Features Explained

### AI Sentiment Analysis
- Uses TextBlob for natural language processing
- Enhanced with crypto-specific keywords
- Provides confidence scores and market predictions
- Processes news headlines and content for sentiment classification

### Whale Detection Algorithm
- Monitors volume spikes using z-score analysis
- Detects price impact thresholds (2%+ moves)
- Classifies events by severity (High/Medium/Low)
- Provides confidence scores for trading decisions

### Trading Simulation
- AI-powered buy/sell/hold logic
- Risk management with stop-loss and take-profit
- Portfolio tracking with equity curves
- Performance metrics (win rate, total return, max drawdown)

### Synthetic Data Generation
- **Completely offline** - no API keys required
- Realistic cryptocurrency price movements
- Simulated whale events and news articles
- Easy to replace with live data sources

## 🔧 Configuration

The application uses synthetic data by default. To customize:

1. **Modify data generators** in `data_generators.py`
2. **Adjust whale detection parameters** in `whale_detector.py`
3. **Update trading simulation settings** in `trading_simulator.py`
4. **Customize sentiment analysis** in `sentiment_analyzer.py`

## 🌐 API Endpoints

### Cryptocurrency Data
- `GET /api/crypto/prices` - Current prices for all coins
- `GET /api/crypto/history/<symbol>` - Historical price data
- `GET /api/crypto/chart/<symbol>` - Chart data for visualization

### News & Sentiment
- `GET /api/news` - News with AI sentiment analysis

### Whale Detection & Trading
- `GET /api/whale/detect` - Detect whale events
- `GET /api/whale/simulate` - Run trading simulation
- `GET /api/portfolio/status` - Portfolio status
- `GET /api/portfolio/reset` - Reset portfolio

## 🎨 UI/UX Features

- **Modern, responsive design** with Tailwind CSS
- **Interactive charts** with Chart.js
- **Real-time updates** with auto-refresh
- **Loading states** and error handling
- **Mobile-friendly** responsive layout
- **Professional appearance** suitable for portfolios

## 🔮 Future Enhancements

- **Real API integration** (CoinGecko, Binance, etc.)
- **More sophisticated ML models** (LSTM, Transformer)
- **Additional trading strategies** (DCA, Grid, etc.)
- **Portfolio optimization** algorithms
- **Risk management** tools
- **Backtesting** capabilities
- **Real-time notifications**

## 📊 Performance Metrics

The application demonstrates:
- **AI accuracy** in sentiment analysis
- **Trading performance** with win rates and returns
- **Whale detection** effectiveness
- **Real-time processing** capabilities

## 🛡️ Security & Privacy

- **No external API calls** in demo mode
- **Local data processing** - all analysis happens offline
- **No personal data collection**
- **Safe for demonstration** and portfolio use

## 🤝 Contributing

This project is designed as a portfolio demonstration. Feel free to:
- Fork and customize for your needs
- Add new features and algorithms
- Improve the UI/UX
- Optimize performance
- Add real API integrations

## 📄 License

This project is open source and available under the MIT License.

## 🎓 Educational Value

Perfect for demonstrating:
- **Full-stack development** skills
- **AI/ML implementation** in real applications
- **Data science** methodologies
- **Financial technology** concepts
- **Modern web development** practices

---

**Built with ❤️ for the crypto and AI community**

*This project showcases the intersection of artificial intelligence, data science, and financial technology in a practical, portfolio-ready application.*
