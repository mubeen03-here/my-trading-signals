import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from groq import Groq
import google.generativeai as genai
from PIL import Image
import math

# ==================== API KEYS SETUP ====================
if "GROQ_API_KEY" in st.secrets:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("GROQ_API_KEY is missing in Streamlit Secrets!")
    st.stop()

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

st.set_page_config(page_title="Quantum AI Signal Engine", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: #e6edf3; }
    .main-header { font-size: 2.4rem; font-weight: 800; background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
    .symbol-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 1rem; }
    .signal-badge { padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 700; font-size: 0.85rem; display: inline-block; }
    .strong-buy { background-color: #00c853; color: white; }
    .buy { background-color: #2e7d32; color: white; }
    .neutral { background-color: #ff9800; color: white; }
    .sell { background-color: #c62828; color: white; }
    .strong-sell { background-color: #d32f2f; color: white; }
    .quant-box { background-color: #131924; border: 1px solid #1f6feb; border-radius: 12px; padding: 1.2rem; margin: 1rem 0; }
    .entropy-high { border-left: 5px solid #f44336; }
    .entropy-low { border-left: 5px solid #00c853; }
    .ai-box { background-color: #1a1f2e; border: 1px solid #4a90e2; border-radius: 12px; padding: 1rem; margin-top: 1rem; }
    .gemini-box { background-color: #1c2833; border: 1px solid #00ff9f; border-radius: 12px; padding: 1rem; margin-top: 1rem; }
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
    return datetime.now(tz).strftime("%d %b %Y | %I:%M:%S %p PKT")

@st.cache_data(ttl=35, show_spinner=False)
def fetch_ohlcv(ticker, interval="15m", period="30d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.reset_index()
        df.columns = [str(c).capitalize() for c in df.columns]
        
        rename_map = {}
        for col in df.columns:
            if "datetime" in col.lower() or "date" in col.lower(): rename_map[col] = "Datetime"
            elif col.lower() in ["close", "open", "high", "low"]: rename_map[col] = col.capitalize()
        df = df.rename(columns=rename_map)
        return df[["Datetime", "Open", "High", "Low", "Close"]].dropna()
    except:
        return None

# ==================== QUANT MATHEMATICAL MODELS ====================

def calculate_shannon_entropy(series, bins=10):
    """Calculates Market Noise (Entropy). High Entropy = Chaos/No Trade."""
    returns = np.diff(np.log(series))
    hist, _ = np.histogram(returns, bins=bins)
    prob = hist / float(np.sum(hist))
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    max_entropy = np.log2(bins)
    normalized_entropy = entropy / max_entropy
    return round(normalized_entropy, 3)

def fast_dtw_distance(s1, s2):
    """Dynamic Time Warping: Non-linear pattern similarity matching."""
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), fill_value=np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
    return dtw_matrix[n, m]

def monte_carlo_simulation(last_price, returns_std, num_sims=300, steps=5):
    """Simulates 300 price trajectories to compute win probability."""
    sims = np.zeros((num_sims, steps))
    for i in range(num_sims):
        path = [last_price]
        for s in range(steps - 1):
            shock = np.random.normal(0, returns_std)
            path.append(path[-1] * (1 + shock))
        sims[i] = path
    bullish_paths = np.sum(sims[:, -1] > last_price)
    win_rate = (bullish_paths / num_sims) * 100
    return round(win_rate, 1)

def calculate_quant_signals(df):
    if df is None or len(df) < 50: return None
    df = df.copy()
    close = df['Close'].astype(float).values
    
    # 1. Entropy Check (Noise Detection)
    entropy_score = calculate_shannon_entropy(close[-40:])
    
    # 2. Return volatility
    log_returns = np.diff(np.log(close[-50:]))
    volatility = np.std(log_returns)
    
    # 3. Monte Carlo Win Rate
    monte_carlo_bull_prob = monte_carlo_simulation(close[-1], volatility, num_sims=400, steps=5)
    
    # Technical Indicators
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    last = df.iloc[-1]
    price = float(last['Close'])
    
    # Base Signal
    score = 0
    if price > last['EMA_9'] > last['EMA_21']: score += 2
    elif price < last['EMA_9'] < last['EMA_21']: score -= 2
    
    if monte_carlo_bull_prob > 58: score += 2
    elif monte_carlo_bull_prob < 42: score -= 2
    
    if score >= 3: signal, badge = "STRONG BUY", "strong-buy"
    elif score >= 1: signal, badge = "BUY", "buy"
    elif score <= -3: signal, badge = "STRONG SELL", "strong-sell"
    elif score <= -1: signal, badge = "SELL", "sell"
    else: signal, badge = "WAIT", "neutral"
    
    # Override signal if Market Entropy (Noise) is extremely high
    is_noisy = entropy_score > 0.88
    if is_noisy:
        signal = "WAIT (High Noise)"
        badge = "neutral"
        
    return {
        "signal": signal, "badge_class": badge, "score": score, "price": round(price, 2),
        "entropy": entropy_score, "is_noisy": is_noisy, "mc_bull_prob": monte_carlo_bull_prob
    }

def dtw_sequence_predictor(df, pattern_len=8, predict_len=6):
    if df is None or len(df) < 200: return None
    
    close = df['Close'].astype(float).values
    open_p = df['Open'].astype(float).values
    
    # Normalize current pattern shape
    curr = (close[-pattern_len:] - open_p[-pattern_len:])
    curr_norm = (curr - np.mean(curr)) / (np.std(curr) + 1e-8)
    
    best_dist = float('inf')
    best_idx = -1
    
    search_range = min(800, len(df) - pattern_len - predict_len - 2)
    for idx in range(len(df) - search_range - pattern_len - predict_len, len(df) - pattern_len - predict_len):
        hist = (close[idx:idx+pattern_len] - open_p[idx:idx+pattern_len])
        hist_norm = (hist - np.mean(hist)) / (np.std(hist) + 1e-8)
        
        dist = fast_dtw_distance(curr_norm, hist_norm)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
            
    if best_idx == -1: return None
    
    sequence = []
    g_count, r_count = 0, 0
    for k in range(predict_len):
        f_idx = best_idx + pattern_len + k
        if f_idx < len(df):
            if close[f_idx] >= open_p[f_idx]:
                sequence.append("🟢 Green")
                g_count += 1
            else:
                sequence.append("🔴 Red")
                r_count += 1
                
    return {"sequence": sequence, "green": g_count, "red": r_count, "match_quality": round(100 - (best_dist * 5), 1)}

def get_grok_analysis(symbol, tf, signal, metrics):
    prompt = f"Symbol: {symbol}, TF: {tf}, Signal: {signal}, Quant Metrics: {metrics}. Provide a direct 5-line market execution plan."
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4, max_tokens=200
        )
        return res.choices[0].message.content.strip()
    except Exception as e: return f"Grok Error: {str(e)}"

def analyze_chart_with_gemini(image, symbol, tf):
    if "GEMINI_API_KEY" not in st.secrets: return "Gemini Key Missing"
    prompt = f"Analyze this chart image for {symbol} ({tf}). Predict next candle direction (Bullish/Bearish) with clear reasoning."
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        res = model.generate_content([prompt, image])
        return res.text
    except Exception as e: return f"Gemini Error: {str(e)}"

# ==================== STREAMLIT UI ====================
st.markdown('<h1 class="main-header">⚡ Quantum AI Signal Engine</h1>', unsafe_allow_html=True)
st.caption(f"PKT: {get_pakistan_time()} | Entropy + DTW Non-Linear Fractals + Monte Carlo")

cols = st.columns(3)
for idx, (disp, meta) in enumerate(MAIN_SYMBOLS.items()):
    with cols[idx % 3]:
        q_df = fetch_ohlcv(meta["yf_ticker"], interval="60m", period="5d")
        p, q_sig, badge = 0.0, "WAIT", "neutral"
        if q_df is not None:
            anal = calculate_quant_signals(q_df)
            if anal:
                p = anal["price"]
                q_sig = anal["signal"]
                badge = anal["badge_class"]
        st.markdown(f"""
        <div class="symbol-card">
            <strong>{meta['display']}</strong><br>
            <span style="font-size:1.5rem; font-weight:700;">{p:,.2f}</span><br>
            <span class="signal-badge {badge}">{q_sig}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Analyze {disp}", key=f"s_{disp}"):
            st.session_state.selected_symbol = disp
            st.rerun()

if st.session_state.selected_symbol:
    sel = st.session_state.selected_symbol
    meta = MAIN_SYMBOLS[sel]
    st.divider()
    st.subheader(f"🧠 Quantitative Deep Dive: {sel}")
    
    tf = st.selectbox("Select Timeframe", ["5m", "15m", "1h", "4h"], index=1)
    df = fetch_ohlcv(meta["yf_ticker"], interval=tf, period="30d")
    
    q_res = calculate_quant_signals(df)
    if q_res:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Live Price", f"{q_res['price']:,}")
        c2.metric("Quantum Signal", q_res['signal'])
        c3.metric("Shannon Noise Entropy", f"{q_res['entropy']} / 1.0")
        c4.metric("Monte Carlo Bullish Prob.", f"{q_res['mc_bull_prob']}%")
        
        # Entropy Warning
        ent_class = "entropy-high" if q_res['is_noisy'] else "entropy-low"
        st.markdown(f"""
        <div class="quant-box {ent_class}">
            <h4>🔬 Shannon Market Noise Analysis:</h4>
            <p>Market Entropy is <b>{q_res['entropy']}</b>. {'⚠️ High Chaos Detected: Technical signals disabled to prevent fakeouts.' if q_res['is_noisy'] else '✅ Clean Market Structure: Signal confidence is high.'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # DTW Sequence Predictor
        st.markdown("### 🌀 Dynamic Time Warping (DTW) Sequence Forecast")
        seq = dtw_sequence_predictor(df, pattern_len=8, predict_len=6)
        if seq:
            st.write(f"Matched Historical Pattern Shape Quality: **{seq['match_quality']}%**")
            scols = st.columns(len(seq['sequence']))
            for i, step in enumerate(seq['sequence']):
                with scols[i]:
                    st.markdown(f"**Candle {i+1}**\n\n{step}")
                    
        # Grok & Gemini
        st.markdown("---")
        st.markdown("### 🤖 Grok AI Strategic Execution")
        if st.button("Generate Grok Tactical Plan"):
            metrics = f"Entropy: {q_res['entropy']}, MC Prob: {q_res['mc_bull_prob']}%"
            st.info(get_grok_analysis(sel, tf, q_res['signal'], metrics))
            
        st.markdown("---")
        st.markdown("### 📸 Gemini AI Vision Chart Analysis")
        up_file = st.file_uploader("Upload Chart Screenshot", type=["png", "jpg", "jpeg"])
        if up_file:
            img = Image.open(up_file)
            st.image(img, use_container_width=True)
            if st.button("Analyze Chart Image"):
                st.success(analyze_chart_with_gemini(img, sel, tf))
    
