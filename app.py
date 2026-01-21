from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import numpy as np
import os
import textblob 
import joblib
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

# Ensure the directory exists
os.makedirs('static/plots', exist_ok=True)
 


app = Flask(__name__)

# Load model
model = joblib.load('stock_model.pkl')

# Set your Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = 'MO1KFL9X94FZ01VR'
MARKETAUX_API_KEY = '9zeMnYnU8JAqXgxQAqm0KWN8vZGn9vnV2Q91AeRY'


# ====================== DATA & FEATURE FUNCTIONS ======================

# Mappings
INDIAN_STOCKS = {
    'reliance': 'RELIANCE.BSE',
    'tcs': 'TCS.BSE',
    'infosys': 'INFY.BSE',
    'hdfc bank': 'HDFCBANK.BSE',
    'icici bank': 'ICICIBANK.BSE',
    'wipro': 'WIPRO.BSE',
    'bharti airtel': 'BHARTIARTL.BSE',
    'itc': 'ITC.BSE',
    'hul': 'HINDUNILVR.BSE',
    'hindustan unilever': 'HINDUNILVR.BSE',
    'maruti suzuki': 'MARUTI.BSE',
    'bajaj finance': 'BAJFINANCE.BSE',
    'axis bank': 'AXISBANK.BSE',
    'kotak mahindra': 'KOTAKBANK.BSE',
    'l&t': 'LT.BSE',
    'larsen & toubro': 'LT.BSE',
    'sun pharma': 'SUNPHARMA.BSE',
    'ntpc': 'NTPC.BSE',
    'power grid': 'POWERGRID.BSE',
    'ongc': 'ONGC.BSE',
    'tatasteel': 'TATASTEEL.BSE',
    'tata steel': 'TATASTEEL.BSE',
    'adani enterprises': 'ADANIENT.BSE',
    'adani ports': 'ADANIPORTS.BSE'
}

US_STOCKS = {
    'apple': 'AAPL',
    'microsoft': 'MSFT',
    'google': 'GOOGL',
    'alphabet': 'GOOGL',
    'amazon': 'AMZN',
    'tesla': 'TSLA',
    'meta': 'META',
    'facebook': 'META',
    'nvidia': 'NVDA',
    'netflix': 'NFLX',
    'disney': 'DIS',
    'coca cola': 'KO',
    'pepsi': 'PEP',
    'walmart': 'WMT',
    'johnson & johnson': 'JNJ',
    'procter & gamble': 'PG',
    'visa': 'V',
    'mastercard': 'MA',
    'jpmorgan': 'JPM',
    'goldman sachs': 'GS',
    'intel': 'INTC',
    'cisco': 'CSCO',
    'oracle': 'ORCL',
    'ibm': 'IBM',
    'adobe': 'ADBE',
    'salesforce': 'CRM',
    'uber': 'UBER',
    'airbnb': 'ABNB',
    'spotify': 'SPOT',
    'zoom': 'ZM',
    'twitter': 'TWTR',
    'snap': 'SNAP',
    'pinterest': 'PINS',
    'paypal': 'PYPL',
    'square': 'SQ',
    'robinhood': 'HOOD',
    'coinbase': 'COIN'
}

def get_ticker_by_name(company_name):
    company_lower = company_name.lower().strip()
    if company_lower in INDIAN_STOCKS:
        return INDIAN_STOCKS[company_lower], 'Indian'
    if company_lower in US_STOCKS:
        return US_STOCKS[company_lower], 'US'
    for key, val in INDIAN_STOCKS.items():
        if company_lower in key or key in company_lower:
            return val, 'Indian'
    for key, val in US_STOCKS.items():
        if company_lower in key or key in company_lower:
            return val, 'US'
    return None, None

def validate_ticker_symbol(symbol):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY,
        'outputsize': 'compact'
    }
    try:
        response = requests.get('https://www.alphavantage.co/query', params=params, timeout=10)
        data = response.json()
        return 'Time Series (Daily)' in data
    except Exception:
        return False


def get_alpha_vantage_data(symbol, days=90):
    """Get stock price data from Alpha Vantage"""
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY,
        'outputsize': 'full'
    }
    
    try:
        response = requests.get('https://www.alphavantage.co/query', params=params, timeout=30)
        
        if response.status_code != 200:
            raise ValueError(f"API request failed with status {response.status_code}")
        
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            raise ValueError(f"API Error: {data['Error Message']}")
        if 'Note' in data:
            raise ValueError("API call frequency limit reached. Please wait 1 minute and try again.")
        if 'Time Series (Daily)' not in data:
            raise ValueError('Invalid ticker symbol or API response')
        
        # Convert to DataFrame
        ts_data = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(ts_data, orient='index')
        df = df.astype(float)
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.tail(days)
        
        return df
        
    except requests.RequestException as e:
        raise ValueError(f"Network error: {e}")

def calculate_technical_indicators(df):
    df = df.copy()
    df['price_change'] = df['close'].pct_change()
    df['price_change_2d'] = df['close'].pct_change(periods=2)
    df['price_change_5d'] = df['close'].pct_change(periods=5)
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_5_signal'] = (df['close'] > df['ma_5']).astype(int)
    df['ma_10_signal'] = (df['close'] > df['ma_10']).astype(int)
    df['ma_20_signal'] = (df['close'] > df['ma_20']).astype(int)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # ### FIX 1: Prevent division by zero in RSI calculation ###
    # Add a small epsilon value to the denominator
    epsilon = 1e-10
    rs = gain / (loss + epsilon)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_5'] + epsilon) # Also add epsilon here for safety
    df['volatility'] = df['close'].rolling(window=10).std()
    df['hl_spread'] = (df['high'] - df['low']) / df['close']
    df['next_day_direction'] = (df['close'].shift(-1) > df['close']).astype(int)

    # ### FIX 2: Replace any lingering infinity values ###
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df


def analyze_sentiment(news_data):
    sentiments = []
    for article in news_data.get('data', []):
        text = f"{article.get('title', '')} {article.get('description', '')}"
        if text.strip():
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)
    return np.mean(sentiments) if sentiments else 0.0


def get_news_sentiment(company_name):
    url = f'https://api.marketaux.com/v1/news/all?api_token={MARKETAUX_API_KEY}&language=en&limit=10&filter_entities=true&entities={company_name}'
    response = requests.get(url)
    news_data = response.json()
    return analyze_sentiment(news_data)


def prepare_features(df, sentiment_score):
    feature_columns = [
        'price_change', 'price_change_2d', 'price_change_5d',
        'ma_5_signal', 'ma_10_signal', 'ma_20_signal',
        'rsi', 'volume_change', 'volume_ratio',
        'volatility', 'hl_spread'
    ]
    features_df = df[feature_columns + ['next_day_direction']].copy()
    features_df['sentiment'] = sentiment_score
    features_df = features_df.dropna()
    return features_df
def plot_comprehensive_analysis(stock_df, results_df, ticker, sentiment_score, market_type):
    """Create comprehensive plotting dashboard"""
    fig = plt.figure(figsize=(20, 16))
    
    # Determine currency symbol
    currency = "₹" if market_type == "Indian" else "$"
    
    # Plot 1: Stock Price with Moving Averages
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(stock_df.index, stock_df['close'], label='Close Price', linewidth=2, color='blue')
    ax1.plot(stock_df.index, stock_df['ma_5'], label='MA 5', alpha=0.7, color='red')
    ax1.plot(stock_df.index, stock_df['ma_10'], label='MA 10', alpha=0.7, color='green')
    ax1.plot(stock_df.index, stock_df['ma_20'], label='MA 20', alpha=0.7, color='orange')
    ax1.set_title(f'{ticker} - Price with Moving Averages', fontsize=14, fontweight='bold')
    ax1.set_ylabel(f'Price ({currency})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volume Analysis
    ax2 = plt.subplot(3, 3, 2)
    ax2.bar(stock_df.index, stock_df['volume'], alpha=0.7, color='purple')
    ax2.plot(stock_df.index, stock_df['volume_ma_5'], label='Volume MA 5', color='red', linewidth=2)
    ax2.set_title(f'{ticker} - Volume Analysis', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Volume')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RSI
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(stock_df.index, stock_df['rsi'], label='RSI', color='purple', linewidth=2)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax3.set_title(f'{ticker} - RSI Indicator', fontsize=14, fontweight='bold')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Actual vs Predicted Directions
    ax4 = plt.subplot(3, 3, 4)
    dates = results_df.index
    actual_directions = results_df['next_day_direction'].values
    predicted_directions = results_df['predicted_direction'].values
    
    # Create comparison plot
    x_pos = np.arange(len(dates))
    width = 0.35
    
    ax4.bar(x_pos - width/2, actual_directions, width, label='Actual Direction', alpha=0.7, color='blue')
    ax4.bar(x_pos + width/2, predicted_directions, width, label='Predicted Direction', alpha=0.7, color='red')
    
    ax4.set_title('Actual vs Predicted Directions', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Direction (0=Down, 1=Up)')
    ax4.set_xlabel('Trading Days')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Prediction Confidence Over Time
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(dates, results_df['predicted_probability'], label='Prediction Confidence', color='green', linewidth=2)
    ax5.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Neutral (50%)')
    ax5.set_title('Prediction Confidence Over Time', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Probability of Price Increase')
    ax5.set_ylim(0, 1)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Price Change Distribution
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(stock_df['price_change'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax6.set_title('Price Change Distribution', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Daily Price Change (%)')
    ax6.set_ylabel('Frequency')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Accuracy Analysis
    ax7 = plt.subplot(3, 3, 7)
    correct_predictions = (results_df['next_day_direction'] == results_df['predicted_direction']).astype(int)
    rolling_accuracy = pd.Series(correct_predictions).rolling(window=10).mean()
    
    ax7.plot(dates, rolling_accuracy, label='10-Day Rolling Accuracy', color='purple', linewidth=2)
    ax7.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Chance (50%)')
    ax7.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax7.set_ylabel('Accuracy')
    ax7.set_ylim(0, 1)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Volatility Analysis
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(stock_df.index, stock_df['volatility'], label='Price Volatility', color='orange', linewidth=2)
    ax8.set_title('Price Volatility Over Time', fontsize=14, fontweight='bold')
    ax8.set_ylabel('Volatility')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Sentiment and Performance Summary
    ax9 = plt.subplot(3, 3, 9)
    
    # Calculate some summary statistics
    total_predictions = len(results_df)
    correct_predictions = (results_df['next_day_direction'] == results_df['predicted_direction']).sum()
    accuracy = correct_predictions / total_predictions
    
    # Create text summary
    summary_text = f"""
    PREDICTION SUMMARY
    ==================
    
    Ticker: {ticker}
    Market: {market_type}
    Currency: {currency}
    Total Predictions: {total_predictions}
    Correct Predictions: {correct_predictions}
    Accuracy: {accuracy:.2%}
    
    Current Sentiment: {sentiment_score:.3f}
    Latest Close: {currency}{stock_df['close'].iloc[-1]:.2f}
    Latest RSI: {stock_df['rsi'].iloc[-1]:.1f}
    
    Recent Performance:
    - 5-day change: {stock_df['price_change_5d'].iloc[-1]:.2%}
    - Volume ratio: {stock_df['volume_ratio'].iloc[-1]:.2f}
    """
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    plt.tight_layout()
    plot_filename = f'static/plots/{ticker}_analysis.png'
    plt.savefig(plot_filename)
    plt.close(fig)
    return plot_filename


# ====================== ROUTES ======================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    company_name = request.form.get('company')
    if not company_name:
        return jsonify({'error': 'Company name is required'}), 400

    try:
        ticker, market_type = get_ticker_by_name(company_name)
        if not ticker:
            return jsonify({'error': f"Could not find a ticker for '{company_name}'"}), 400

        df = get_alpha_vantage_data(ticker)
        df = calculate_technical_indicators(df)

        # ### THIS IS THE UPDATED PART ###
        # Get live sentiment score from the news API
        sentiment_score = get_news_sentiment(company_name)

        features_df = prepare_features(df, sentiment_score)
        if features_df.empty:
            return jsonify({'error': 'Not enough historical data to make a prediction.'}), 400
        
        X = features_df.drop('next_day_direction', axis=1).values

        latest_features = X[-1].reshape(1, -1)
        prediction = model.predict(latest_features)[0]
        proba = model.predict_proba(latest_features)[0]
        up_prob, down_prob = round(float(proba[1]), 3), round(float(proba[0]), 3)

        results_df = features_df.copy()
        results_df['predicted_direction'] = model.predict(X)
        results_df['predicted_probability'] = model.predict_proba(X)[:, 1]

        plot_path = plot_comprehensive_analysis(df, results_df, ticker, sentiment_score, market_type)

        return jsonify({
            'direction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': round(up_prob if prediction == 1 else down_prob, 2),
            'probabilities': {'up': up_prob, 'down': down_prob},
            'plot': '/' + plot_path
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=5000)