import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from openai import OpenAI
from PIL import Image
import base64
import io

# ==================== MULTI-KEY & FAST AI ROUTING ====================
OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", None)

if OPENROUTER_KEY:
    ai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
    )
else:
    st.error("⚠️ OPENROUTER_API_KEY is missing in Streamlit Secrets! Please add it.")
    st.stop()

# Page Setup
st.set_page_config(page_title="Quantum AI Signal Engine", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: #e6edf3; }
    .main-header { font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem; }
    .symbol-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 1rem; text-align: center; }
    .signal-badge { padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 700; font-size: 0.85rem; display: inline-block; margin-top: 5px; }
    .strong-buy { background-color: #00c853; color: white; }
    .buy { background-color: #2e7d32; color: white; }
    .neutral { background-color: #ff9800; color: white; }
    .sell { background-color: #c62828; color: white; }
    .strong-sell { background-color: #d32f2f; color: white; }
    .quant-box { background-color: #131924; border: 1px solid #1f6feb; border-radius: 12px; padding: 1.2rem; margin: 1rem 0; }
    .entropy-high { border-left: 5px solid #f44336; }
    .entropy-low { border-left: 5px solid #00c853; }
</style>
""", unsafe_allow_html=True)

MAIN_SYMBOLS = {
    "Bitcoin (BTC)": {"yf_ticker": "BTC-USD", "display": "BTC/USD"},
    "USD/JPY": {"yf_ticker": "USDJPY=X", "display": "USD/JPY"},
    "NAS100": {"yf_ticker": "NQ=F", "display": "NAS100 (NQ)"},
}

if "selected_symbol" not in st.session_state or st.session_state.selected_symbol is None:
    st.session_state.selected_symbol = "Bitcoin (BTC)"

def get_pakistan_time():
    tz = pytz.timezone('Asia/Karachi')
    return datetime.now(tz).strftime("%d %b %Y | %I:%M:%S %p PKT")

@st.cache_data(ttl=20, show_spinner=False)
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
    returns = np.diff(np.log(series))
    hist, _ = np.histogram(returns, bins=bins)
    prob = hist / float(np.sum(hist))
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    max_entropy = np.log2(bins)
    return round(entropy / max_entropy, 3)

def fast_dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), fill_value=np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
    return dtw_matrix[n, m]

def monte_carlo_simulation(last_price, returns_std, num_sims=300, steps=5):
    sims = np.zeros((num_sims, steps))
    for i in range(num_sims):
        path = [last_price]
        for s in range(steps - 1):
            shock = np.random.normal(0, returns_std)
            path.append(path[-1] * (1 + shock))
        sims[i] = path
    bullish_paths = np.sum(sims[:, -1] > last_price)
    return round((bullish_paths / num_sims) * 100, 1)

def calculate_quant_signals(df):
    if df is None or len(df) < 50: return None
    df = df.copy()
    close = df['Close'].astype(float).values
    
    entropy_score = calculate_shannon_entropy(close[-40:])
    log_returns = np.diff(np.log(close[-50:]))
    volatility = np.std(log_returns)
    monte_carlo_bull_prob = monte_carlo_simulation(close[-1], volatility, num_sims=300, steps=5)
    
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    last = df.iloc[-1]
    price = float(last['Close'])
    
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
    
    is_noisy = entropy_score > 0.88
    if is_noisy:
        signal = "WAIT (High Noise)"
        badge = "neutral"
        
    return {
        "signal": signal, "badge_class": badge, "score": score, "price": round(price, 2),
        "entropy": entropy_score, "is_noisy": is_noisy, "mc_bull_prob": monte_carlo_bull_prob
    }

# STABILIZED DTW SEQUENCE PREDICTOR (Top 3 Historical Consensus)
def dtw_sequence_predictor(df, pattern_len=8, predict_len=6):
    if df is None or len(df) < 200: return None
    close = df['Close'].astype(float).values
    open_p = df['Open'].astype(float).values
    
    # Use closed candles to stabilize pattern
    curr = (close[-pattern_len-1:-1] - open_p[-pattern_len-1:-1])
    curr_norm = (curr - np.mean(curr)) / (np.std(curr) + 1e-8)
    
    matches = []
    search_range = min(700, len(df) - pattern_len - predict_len - 5)
    
    for idx in range(len(df) - search_range - pattern_len - predict_len, len(df) - pattern_len - predict_len):
        hist = (close[idx:idx+pattern_len] - open_p[idx:idx+pattern_len])
        hist_norm = (hist - np.mean(hist)) / (np.std(hist) + 1e-8)
        dist = fast_dtw_distance(curr_norm, hist_norm)
        matches.append((dist, idx))
        
    if not matches: return None
    
    # Sort and take Top 3 closest patterns
    matches.sort(key=lambda x: x[0])
    top_3 = matches[:3]
    
    sequence = []
    for k in range(predict_len):
        bull_votes = 0
        for dist, best_idx in top_3:
            f_idx = best_idx + pattern_len + k
            if f_idx < len(df) and close[f_idx] >= open_p[f_idx]:
                bull_votes += 1
        sequence.append("🟢 Green" if bull_votes >= 2 else "🔴 Red")
        
    avg_dist = np.mean([m[0] for m in top_3])
    quality = round(max(0, 100 - (avg_dist * 4)), 1)
    
    return {"sequence": sequence, "match_quality": quality}

# ==================== ULTRA-FAST MULTI-MODEL ROUTER ====================

def get_openrouter_text_analysis(symbol, tf, signal, metrics):
    prompt = f"Symbol: {symbol}, Timeframe: {tf}, Signal: {signal}, Metrics: {metrics}. Provide 4 ultra-concise bullet points: Bias, Entry Zone, Target, and Next Candle Expectation."
    
    # High-speed models list
    fast_models = [
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "qwen/qwen-2.5-72b-instruct:free",
        "deepseek/deepseek-r1:free"
    ]
    
    for model_name in fast_models:
        try:
            response = ai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                timeout=12
            )
            return response.choices[0].message.content.strip()
        except Exception:
            continue
            
    return "⚡ Response took too long. Click button again for instant retry."

def analyze_chart_with_openrouter_vision(image, symbol, tf):
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        prompt = f"Analyze this chart screenshot for {symbol} ({tf}). Predict the NEXT immediate candle direction (Bullish Green / Bearish Red) with clear price action reason."

        response = ai_client.chat.completions.create(
            model="google/gemini-2.0-flash-exp:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_str}"}
                        }
                    ]
                }
            ],
            timeout=18
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Vision AI Error: {str(e)}"

# ==================== STREAMLIT UI ====================

# Top Header
head_col, btn_col = st.columns([3, 1])
with head_col:
    st.markdown('<h1 class="main-header">⚡ Quantum AI Signal Engine</h1>', unsafe_allow_html=True)
    st.caption(f"PKT: {get_pakistan_time()} | Active Focus: {st.session_state.selected_symbol}")
with btn_col:
    st.write("")
    if st.button("🔄 Refresh Market Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()

# Pair Cards Top Bar
cols = st.columns(3)
for idx, (disp, meta) in enumerate(MAIN_SYMBOLS.items()):
    with cols[idx % 3]:
        q_df = fetch_ohlcv(meta["yf_ticker"], interval="60m", period="5d")
        p, q_sig, badge = 0.0, "WAIT", "neutral"
        if q_df is not None:
            anal = calculate_quant_signals(q_df)
            if anal:
                p, q_sig, badge = anal["price"], anal["signal"], anal["badge_class"]
        
        is_active = (st.session_state.selected_symbol == disp)
        border_style = "border: 2px solid #00f2fe; background-color: #1c2333;" if is_active else ""
        
        st.markdown(f"""
        <div class="symbol-card" style="{border_style}">
            <strong>{meta['display']}</strong><br>
            <span style="font-size:1.4rem; font-weight:700;">{p:,.2f}</span><br>
            <span class="signal-badge {badge}">{q_sig}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Focus {disp}", key=f"s_{disp}"):
            st.session_state.selected_symbol = disp
            st.rerun()

# Deep Dive Section
sel = st.session_state.selected_symbol
meta = MAIN_SYMBOLS[sel]

st.divider()
st.subheader(f"🧠 Next Candle & Market Structure Deep Dive: {sel}")

tf = st.selectbox("Select Timeframe", ["5m", "15m", "1h", "4h"], index=1)
df = fetch_ohlcv(meta["yf_ticker"], interval=tf, period="30d")

q_res = calculate_quant_signals(df)
if q_res:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live Price", f"{q_res['price']:,}")
    c2.metric("Quantum Signal", q_res['signal'])
    c3.metric("Shannon Noise Entropy", f"{q_res['entropy']} / 1.0")
    c4.metric("Monte Carlo Next Candle Bull Prob.", f"{q_res['mc_bull_prob']}%")
    
    ent_class = "entropy-high" if q_res['is_noisy'] else "entropy-low"
    st.markdown(f"""
    <div class="quant-box {ent_class}">
        <h4>🔬 Shannon Market Chaos Check:</h4>
        <p>Market Noise Level: <b>{q_res['entropy']}</b>. {'⚠️ High Chaos/Squeeze: Wait for breakout before taking next candle trades.' if q_res['is_noisy'] else '✅ Clean Market Trend: High probability setup for next candle prediction.'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🌀 DTW 6-Candle Pattern Forecast (Stabilized Consensus)")
    seq = dtw_sequence_predictor(df, pattern_len=8, predict_len=6)
    if seq:
        st.write(f"Historical Top 3 Shape Consensus Quality: **{seq['match_quality']}%**")
        scols = st.columns(len(seq['sequence']))
        for i, step in enumerate(seq['sequence']):
            with scols[i]:
                st.markdown(f"**Candle +{i+1}**\n\n{step}")
                
    st.markdown("---")
    st.markdown("### 🤖 Fast Tactical AI Analysis")
    if st.button("Generate Immediate Execution Plan"):
        metrics = f"Entropy: {q_res['entropy']}, Bull Prob: {q_res['mc_bull_prob']}%"
        with st.spinner("Analyzing Next Candle via AI..."):
            st.info(get_openrouter_text_analysis(sel, tf, q_res['signal'], metrics))
        
    st.markdown("---")
    st.markdown("### 📸 Gemini AI Chart Screenshot Analysis")
    up_file = st.file_uploader("Upload Chart Screenshot", type=["png", "jpg", "jpeg"])
    if up_file:
        img = Image.open(up_file)
        st.image(img, use_container_width=True)
        if st.button("Predict Next Candle via Gemini Vision"):
            with st.spinner("Analyzing chart image for Next Candle Direction..."):
                st.success(analyze_chart_with_openrouter_vision(img, sel, tf))
    
