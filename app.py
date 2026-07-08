import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from groq import Groq

# ==================== GROQ CLIENT ====================
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.set_page_config(page_title="Pro Trading Signals", layout="wide", initial_sidebar_state="expanded")

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
    .ai-box { background-color: #1a1f2e; border: 1px solid #4a90e2; border-radius: 12px; padding: 1rem; margin-top: 1rem; }
    .current-candle-box { background-color: #1f2a3d; border: 1px solid #4a90e2; border-radius: 12px; padding: 1rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None

MAIN_SYMBOLS = {
    "Bitcoin (BTC)": {"yf_ticker": "BTC-USD", "display": "BTC/USD"},
    "USD/JPY": {"yf_ticker": "USDJPY=X", "display": "USD/JPY"},
    "NAS100": {"yf_ticker": "NQ=F", "display": "NAS100 (NQ)"},
}

def get_pakistan_time():
    tz = pytz.timezone('Asia/Karachi')
    return datetime.now(tz).strftime("%d %b %Y  |  %I:%M:%S %p PKT")

@st.cache_data(ttl=35, show_spinner=False)
def fetch_ohlcv(ticker, interval="15m", period="5d"):
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

def calculate_technical_signal(df):
    if df is None or len(df) < 35: return None
    df = df.copy()
    close = df['Close']
    
    df['EMA_9'] = close.ewm(span=9, adjust=False).mean()
    df['EMA_21'] = close.ewm(span=21, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (close.diff().where(close.diff() > 0, 0).rolling(14).mean() / 
                           close.diff().where(close.diff() < 0, 0).rolling(14).mean().abs())))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd_hist
    
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_Upper'] = sma20 + std20 * 2
    df['BB_Lower'] = sma20 - std20 * 2
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    df = df.dropna()
    if len(df) < 15: return None
    
    last = df.iloc[-1]
    price = float(last['Close'])
    score = 0
    reasons = []
    
    if price > last['EMA_9'] > last['EMA_21']: score += 2; reasons.append("✅ Strong bullish structure")
    elif price > last['EMA_9']: score += 1; reasons.append("✅ Price above EMA9")
    elif price < last['EMA_9'] < last['EMA_21']: score -= 2; reasons.append("❌ Bearish structure")
    
    rsi = float(last['RSI'])
    if rsi > 58: score += 1; reasons.append("✅ RSI bullish")
    elif rsi < 42: score -= 1; reasons.append("❌ RSI bearish")
    
    if last['MACD_Hist'] > 0: score += 1; reasons.append("✅ MACD positive")
    else: score -= 1; reasons.append("❌ MACD negative")
    
    if price <= last['BB_Lower'] * 1.01: score += 1; reasons.append("✅ Near lower band")
    elif price >= last['BB_Upper'] * 0.99: score -= 1; reasons.append("❌ Near upper band")
    
    if score >= 4: signal, badge = "STRONG BUY", "strong-buy"
    elif score >= 2: signal, badge = "BUY", "buy"
    elif score <= -4: signal, badge = "STRONG SELL", "strong-sell"
    elif score <= -2: signal, badge = "SELL", "sell"
    else: signal, badge = "WAIT", "neutral"
    
    # Next Candle Prediction
    if "BUY" in signal:
        expected = "Next candle likely bullish (green)"
        pullback = "Possible small red pullback then continuation"
    elif "SELL" in signal:
        expected = "Next candle likely bearish (red)"
        pullback = "Possible small green pullback then continuation"
    else:
        expected = "Next candle direction unclear - better to wait"
        pullback = ""
    
    return {
        "signal": signal, "badge_class": badge, "score": score, "reasons": reasons,
        "last_price": round(price, 2), "rsi": round(rsi, 1), "atr": round(float(last['ATR']), 2),
        "expected_candles": expected, "pullback": pullback
    }

def get_current_candle_status(df):
    """Real-time current forming candle"""
    if df is None or len(df) < 2: return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    last_color = "🟢 Green" if last['Close'] > last['Open'] else "🔴 Red"
    current_price = float(last['Close'])
    current_open = float(last['Open'])
    
    if current_price > current_open:
        forming = "🟢 Bullish forming right now"
    else:
        forming = "🔴 Bearish forming right now"
    
    return {
        "last_closed": last_color,
        "forming_now": forming
    }

def get_ai_insight(symbol, tf, technical_signal, recent_data):
    prompt = f"""
    You are a professional price action trader.
    Symbol: {symbol}
    Timeframe: {tf}
    Technical Signal: {technical_signal}
    
    Recent behavior: {recent_data}
    
    Give short honest answer:
    1. Probability that NEXT candle will be bullish or bearish?
    2. Should we wait or take trade?
    3. Any important observation?
    
    Max 4-5 lines.
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except:
        return "AI insight not available right now."

# ==================== UI ====================
st.markdown('<h1 class="main-header">📈 Pro Trading Signals</h1>', unsafe_allow_html=True)
st.caption(f"Pakistan Time: {get_pakistan_time()}  |  Technical Primary + AI Second Opinion")

if st.button("🔄 Refresh All Data"):
    st.cache_data.clear()
    st.rerun()

# Grid Layout
cols = st.columns(3)
for idx, (disp_name, meta) in enumerate(MAIN_SYMBOLS.items()):
    col = cols[idx % 3]
    with col:
        quick_df = fetch_ohlcv(meta["yf_ticker"], interval="60m", period="2d")
        price, pct, sig, badge = 0, 0, "NEUTRAL", "neutral"
        if quick_df is not None and len(quick_df) > 1:
            price = float(quick_df['Close'].iloc[-1])
            pct = ((price - float(quick_df['Close'].iloc[0])) / float(quick_df['Close'].iloc[0])) * 100
            anal = calculate_technical_signal(quick_df)
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
    df = fetch_ohlcv(meta["yf_ticker"], interval=tf, period="5d")
    analysis = calculate_technical_signal(df)
    
    if analysis:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", analysis['last_price'])
        c2.metric("Technical Signal", analysis['signal'])
        c3.metric("RSI", analysis['rsi'])
        c4.metric("ATR", analysis['atr'])
        
        # Trade Setup
        if analysis['signal'] == "WAIT":
            st.markdown(f"""
            <div class="wait-box">
            <h3>⏳ WAIT - No Clear Setup</h3>
            <p>Technical confluence is low. Better to wait for clear structure.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("### 🎯 Trade Setup")
            st.code(f"Entry around: {analysis['last_price']}\nUse ATR for SL & TP")
        
        # NEW: Current Forming Candle (Real-time)
        current_candle = get_current_candle_status(df)
        if current_candle:
            st.markdown("### 📍 Current Market Candle (Real-time)")
            st.markdown(f"""
            <div class="current-candle-box">
            <b>Last Closed Candle:</b> {current_candle['last_closed']}<br>
            <b>Currently Forming:</b> {current_candle['forming_now']}
            </div>
            """, unsafe_allow_html=True)
        
        # Technical Next Candle Expectation
        st.markdown("### 🕯️ Next Candle Expectation (Technical)")
        st.info(analysis['expected_candles'])
        if analysis.get('pullback'):
            st.warning(analysis['pullback'])
        
        # AI Second Opinion
        st.markdown("### 🤖 AI Insight (Second Opinion Only)")
        with st.spinner("Getting AI perspective..."):
            recent_summary = f"Last 10 candles - RSI: {analysis['rsi']}, Price: {analysis['last_price']}"
            ai_response = get_ai_insight(selected, tf, analysis['signal'], recent_summary)
        
        st.markdown(f"""
        <div class="ai-box">
        {ai_response}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🧠 Technical Reasons")
        for r in analysis['reasons']:
            st.write(r)
        
        st.caption("⚡ Signals update when new candle closes. Click 'Refresh All Data' to see latest.")
    else:
        st.error("Not enough data for this timeframe.")

st.caption("Technical Primary + AI Second Opinion • Free Tier • Data via yfinance")
