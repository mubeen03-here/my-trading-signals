import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz
from twelvedata import TDClient

st.set_page_config(page_title="Pro Trading Signals", layout="wide", initial_sidebar_state="expanded")

# ==================== CONFIG ====================
TWELVE_DATA_API_KEY = "04686c9409744e3d8453e3a371796a3c"   # Tumhari key
td = TDClient(apikey=TWELVE_DATA_API_KEY)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .main-header { font-size: 2.2rem; font-weight: 700; background: linear-gradient(90deg, #00ff9f, #00b8ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
    .symbol-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 14px; padding: 1rem; margin-bottom: 1rem; }
    .signal-badge { padding: 0.3rem 0.9rem; border-radius: 20px; font-weight: 700; font-size: 0.9rem; display: inline-block; }
    .strong-buy { background-color: #00c853; color: white; }
    .buy { background-color: #4caf50; color: white; }
    .neutral { background-color: #ff9800; color: white; }
    .sell { background-color: #f44336; color: white; }
    .strong-sell { background-color: #d32f2f; color: white; }
    .metric-value { font-size: 1.7rem; font-weight: 700; }
    .trade-box { background-color: #161b22; border: 2px solid #00b8ff; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
    .wait-box { background-color: #2d2d2d; border: 2px solid #ff9800; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None

MAIN_SYMBOLS = {
    "Bitcoin (BTC)": {"ticker": "BTC/USD", "yf_ticker": "BTC-USD", "display": "BTC/USD", "category": "Crypto"},
    "USD/JPY": {"ticker": "USD/JPY", "yf_ticker": "USDJPY=X", "display": "USD/JPY", "category": "Forex"},
    "NAS100": {"ticker": "IXIC", "yf_ticker": "NQ=F", "display": "NAS100 (NQ)", "category": "Index"},
}

def get_pakistan_time():
    tz = pytz.timezone('Asia/Karachi')
    return datetime.now(tz).strftime("%d %b %Y  |  %I:%M:%S %p PKT")

@st.cache_data(ttl=40, show_spinner=False)
def fetch_ohlcv(symbol_info, interval="15m", period="5d"):
    ticker = symbol_info.get("yf_ticker", symbol_info["ticker"])
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty: return None
        df = df.reset_index()
        df.columns = [str(c[0]).capitalize() if isinstance(c, tuple) else str(c).capitalize() for c in df.columns]
        rename_map = {}
        for col in df.columns:
            if "datetime" in col.lower() or "date" in col.lower(): rename_map[col] = "Datetime"
            elif col.lower() in ["close", "open", "high", "low"]: rename_map[col] = col.capitalize()
        df = df.rename(columns=rename_map)
        if "Close" not in df.columns: return None
        return df[["Datetime", "Open", "High", "Low", "Close"]].dropna()
    except:
        return None

def calculate_signal_and_levels(df, tf="15m"):
    if df is None or len(df) < 35: return None
    
    df = df.copy()
    close = df['Close']
    
    # Indicators
    df['EMA_9'] = close.ewm(span=9, adjust=False).mean()
    df['EMA_21'] = close.ewm(span=21, adjust=False).mean()
    df['RSI'] = ta.rsi(close, length=14) if 'ta' in dir() else (100 - (100 / (1 + (close.diff().where(close.diff() > 0, 0).rolling(14).mean() / 
                           close.diff().where(close.diff() < 0, 0).rolling(14).mean().abs()))))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd_line - signal_line
    
    # Bollinger
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_Upper'] = sma20 + (std20 * 2)
    df['BB_Lower'] = sma20 - (std20 * 2)
    
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df = df.dropna()
    if len(df) < 15: return None
    
    last = df.iloc[-1]
    price = float(last['Close'])
    
    # Scoring
    score = 0
    reasons = []
    
    # Trend
    if price > last['EMA_9'] > last['EMA_21']:
        score += 2; reasons.append("✅ Strong bullish structure")
    elif price > last['EMA_9']:
        score += 1; reasons.append("✅ Price above EMA9")
    elif price < last['EMA_9'] < last['EMA_21']:
        score -= 2; reasons.append("❌ Bearish structure")
    
    # Momentum
    rsi_val = float(last['RSI']) if 'RSI' in last else 50
    if rsi_val > 58: score += 1; reasons.append("✅ RSI bullish")
    elif rsi_val < 42: score -= 1; reasons.append("❌ RSI bearish")
    
    if last['MACD_Hist'] > 0: score += 1; reasons.append("✅ MACD positive")
    else: score -= 1; reasons.append("❌ MACD negative")
    
    # Volatility
    if price <= last['BB_Lower'] * 1.01: score += 1; reasons.append("✅ Near lower band")
    elif price >= last['BB_Upper'] * 0.99: score -= 1; reasons.append("❌ Near upper band")
    
    # Final Decision
    if score >= 4:
        signal = "STRONG BUY"
        badge = "strong-buy"
    elif score >= 2:
        signal = "BUY"
        badge = "buy"
    elif score <= -4:
        signal = "STRONG SELL"
        badge = "strong-sell"
    elif score <= -2:
        signal = "SELL"
        badge = "sell"
    else:
        signal = "WAIT"
        badge = "neutral"
    
    # Next Candle Prediction
    if signal in ["STRONG BUY", "BUY"]:
        expected = "Next candle likely bullish (green)"
        pullback = "Watch for small red pullback then continuation"
    elif signal in ["STRONG SELL", "SELL"]:
        expected = "Next candle likely bearish (red)"
        pullback = "Watch for small green pullback then continuation"
    else:
        expected = "Next candle direction unclear"
        pullback = "Better to wait for clear structure"
    
    return {
        "signal": signal,
        "badge_class": badge,
        "score": score,
        "reasons": reasons,
        "expected_candles": expected,
        "pullback": pullback,
        "last_price": round(price, 2),
        "rsi": round(rsi_val, 1),
        "atr": round(float(last['ATR']), 2)
    }

def build_chart(df, analysis, symbol_name, tf):
    if df is None or analysis is None: return None
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Datetime'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Price",
        increasing_line_color="#00c853", decreasing_line_color="#f44336"))
    
    last_price = float(df['Close'].iloc[-1])
    if "BUY" in analysis['signal']:
        fig.add_annotation(x=df['Datetime'].iloc[-1], y=last_price*0.99, text="▲ LONG", showarrow=True,
            arrowhead=2, arrowcolor="#00c853", font=dict(color="#00c853", size=14))
    elif "SELL" in analysis['signal']:
        fig.add_annotation(x=df['Datetime'].iloc[-1], y=last_price*1.01, text="▼ SHORT", showarrow=True,
            arrowhead=2, arrowcolor="#f44336", font=dict(color="#f44336", size=14))
    
    fig.update_layout(title=f"{symbol_name} — {tf}", template="plotly_dark", height=380,
        margin=dict(l=10, r=10, t=40, b=10), xaxis_rangeslider_visible=False)
    return fig

# ==================== UI ====================
st.markdown('<h1 class="main-header">📈 Pro Trading Signals</h1>', unsafe_allow_html=True)
st.caption(f"Pakistan Time: {get_pakistan_time()}  |  Next Candle Focus + Wait Mode")

if st.button("🔄 Refresh All Data"):
    st.cache_data.clear()
    st.rerun()

# Grid
cols = st.columns(3)
for idx, (disp_name, meta) in enumerate(MAIN_SYMBOLS.items()):
    col = cols[idx % 3]
    with col:
        quick_df = fetch_ohlcv(meta, interval="60m", period="2d")
        price, pct, sig, badge = 0, 0, "NEUTRAL", "neutral"
        if quick_df is not None and len(quick_df) > 1:
            price = float(quick_df['Close'].iloc[-1])
            pct = ((price - float(quick_df['Close'].iloc[0])) / float(quick_df['Close'].iloc[0])) * 100
            anal = calculate_signal_and_levels(quick_df)
            if anal: 
                sig = anal["signal"]
                badge = anal["badge_class"]
        
        st.markdown(f"""
        <div class="symbol-card">
            <strong>{meta['display']}</strong><br>
            <span class="metric-value">{price:,.2f}</span>
            <span style="color:{'#00c853' if pct >= 0 else '#f44336'};"> {pct:+.2f}%</span><br>
            <span class="signal-badge {badge}">{sig}</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"View Analysis", key=f"btn_{disp_name}"):
            st.session_state.selected_symbol = disp_name
            st.rerun()

# Detailed View
if st.session_state.selected_symbol:
    selected = st.session_state.selected_symbol
    meta = MAIN_SYMBOLS[selected]
    st.divider()
    st.subheader(f"📊 {selected}")
    
    tf = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "4h"], index=2)
    df = fetch_ohlcv(meta, interval=tf, period="5d" if tf in ["5m","15m"] else "10d")
    analysis = calculate_signal_and_levels(df, tf)
    
    if analysis:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", analysis['last_price'])
        c2.metric("Signal", analysis['signal'])
        c3.metric("RSI", analysis['rsi'])
        c4.metric("ATR", analysis['atr'])
        
        # Chart at bottom
        fig = build_chart(df, analysis, selected, tf)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        if analysis['signal'] == "WAIT":
            st.markdown(f"""
            <div class="wait-box">
            <h3>⏳ WAIT MODE</h3>
            <p>Market structure is not clear. Better to wait for next candle confirmation.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("### 🎯 Trade Setup")
            st.code(f"Entry: {analysis['last_price']}\nSuggested SL & TP based on ATR & Structure")
        
        st.markdown("### 🕯️ Next Candle Expectation")
        st.info(analysis['expected_candles'])
        if analysis.get('pullback'):
            st.warning(analysis['pullback'])
        
        st.markdown("### 🧠 Why this decision?")
        for r in analysis['reasons']:
            st.write(r)
    else:
        st.error("Not enough data. Try higher timeframe.")

st.caption("Professional Next-Candle Focus • Free Tier • Data via yfinance + Twelve Data")
