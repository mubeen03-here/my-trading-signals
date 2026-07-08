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
import requests

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
    .entropy-high { border-left: 6px solid #f44336; }
    .entropy-low { border-left: 6px solid #00c853; }
    
    .confluence-container { background-color: #0d1117; border: 2px solid #238636; border-radius: 16px; padding: 1.5rem; margin-top: 2rem; }
    .ind-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Main Supported Assets
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

@st.cache_data(ttl=30, show_spinner=False)
def fetch_ohlcv(ticker, interval="15m"):
    try:
        if interval == "5m":
            period = "5d"
        elif interval == "15m":
            period = "14d"
        elif interval == "1h":
            period = "60d"
        else:
            period = "120d"
            
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.reset_index()
        df.columns = [str(c).capitalize() for c in df.columns]
        
        rename_map = {}
        for col in df.columns:
            if "datetime" in col.lower() or "date" in col.lower():
                rename_map[col] = "Datetime"
            elif col.lower() in ["close", "open", "high", "low"]:
                rename_map[col] = col.capitalize()
        df = df.rename(columns=rename_map)
        
        if not {"Datetime", "Open", "High", "Low", "Close"}.issubset(df.columns):
            return None
            
        return df[["Datetime", "Open", "High", "Low", "Close"]].dropna()
    except Exception:
        return None

# ==================== QUANT MATHEMATICAL MODELS ====================

def calculate_shannon_entropy(series, bins=10):
    returns = np.diff(np.log(series))
    hist, _ = np.histogram(returns, bins=bins)
    prob = hist / float(np.sum(hist))
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    max_entropy = np.log2(bins)
    return round(float(entropy / max_entropy), 3)

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
    return round(float((bullish_paths / num_sims) * 100), 1)

def calculate_quant_signals(df):
    if df is None or len(df) < 30:
        return None
    df = df.copy()
    close = df['Close'].astype(float).values
    
    entropy_score = calculate_shannon_entropy(close[-30:])
    log_returns = np.diff(np.log(close[-30:]))
    volatility = np.std(log_returns) if len(log_returns) > 0 else 0.01
    monte_carlo_bull_prob = monte_carlo_simulation(close[-1], volatility, num_sims=300, steps=5)
    
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    last = df.iloc[-1]
    price = float(last['Close'])
    
    score = 0
    if price > last['EMA_9'] > last['EMA_21']:
        score += 2
    elif price < last['EMA_9'] < last['EMA_21']:
        score -= 2
    
    if monte_carlo_bull_prob > 58:
        score += 2
    elif monte_carlo_bull_prob < 42:
        score -= 2
    
    if score >= 3:
        signal, badge = "STRONG BUY", "strong-buy"
    elif score >= 1:
        signal, badge = "BUY", "buy"
    elif score <= -3:
        signal, badge = "STRONG SELL", "strong-sell"
    elif score <= -1:
        signal, badge = "SELL", "sell"
    else:
        signal, badge = "WAIT", "neutral"
    
    is_noisy = entropy_score > 0.88
    if is_noisy:
        signal = "WAIT (High Noise)"
        badge = "neutral"
        
    return {
        "signal": signal, "badge_class": badge, "score": score, "price": round(price, 2),
        "entropy": entropy_score, "is_noisy": is_noisy, "mc_bull_prob": monte_carlo_bull_prob
    }

def dtw_sequence_predictor(df, pattern_len=8, predict_len=6):
    if df is None or len(df) < 100:
        return None
    close = df['Close'].astype(float).values
    open_p = df['Open'].astype(float).values
    
    curr = (close[-pattern_len-1:-1] - open_p[-pattern_len-1:-1])
    curr_norm = (curr - np.mean(curr)) / (np.std(curr) + 1e-8)
    
    matches = []
    search_range = min(400, len(df) - pattern_len - predict_len - 5)
    if search_range < 10:
        return None
        
    for idx in range(len(df) - search_range - pattern_len - predict_len, len(df) - pattern_len - predict_len):
        hist = (close[idx:idx+pattern_len] - open_p[idx:idx+pattern_len])
        hist_norm = (hist - np.mean(hist)) / (np.std(hist) + 1e-8)
        dist = fast_dtw_distance(curr_norm, hist_norm)
        matches.append((dist, idx))
        
    if not matches:
        return None
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
    quality = round(float(max(0, 100 - (avg_dist * 4))), 1)
    return {"sequence": sequence, "match_quality": quality}

# ==================== 3-INDICATOR CONFLUENCE ENGINE ====================

def calculate_supertrend(df, period=10, multiplier=3.0):
    high = df['High'].astype(float).values
    low = df['Low'].astype(float).values
    close = df['Close'].astype(float).values
    n = len(df)
    
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, high[0] - low[0])
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    
    hl2 = (high + low) / 2.0
    up = hl2 - (multiplier * atr)
    dn = hl2 + (multiplier * atr)
    
    trend = np.ones(n)
    final_up = np.zeros(n)
    final_dn = np.zeros(n)
    
    for i in range(1, n):
        final_up[i] = max(up[i], final_up[i-1]) if close[i-1] > final_up[i-1] else up[i]
        final_dn[i] = min(dn[i], final_dn[i-1]) if close[i-1] < final_dn[i-1] else dn[i]
        
        if trend[i-1] == -1 and close[i] > final_dn[i-1]:
            trend[i] = 1
        elif trend[i-1] == 1 and close[i] < final_up[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
    return "BULLISH" if trend[-1] == 1 else "BEARISH"

def calculate_ut_bot(df, key_value=1, atr_period=10):
    high = df['High'].astype(float).values
    low = df['Low'].astype(float).values
    close = df['Close'].astype(float).values
    n = len(df)
    
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, high[0] - low[0])
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False).mean().values
    n_loss = key_value * atr
    
    trail = np.zeros(n)
    for i in range(1, n):
        if close[i] > trail[i-1] and close[i-1] > trail[i-1]:
            trail[i] = max(trail[i-1], close[i] - n_loss[i])
        elif close[i] < trail[i-1] and close[i-1] < trail[i-1]:
            trail[i] = min(trail[i-1], close[i] + n_loss[i])
        elif close[i] > trail[i-1]:
            trail[i] = close[i] - n_loss[i]
        else:
            trail[i] = close[i] + n_loss[i]
            
    return "BUY / BULLISH" if close[-1] > trail[-1] else "SELL / BEARISH"

def calculate_market_structure(df, length=5):
    high = df['High'].astype(float).values
    low = df['Low'].astype(float).values
    close = df['Close'].astype(float).values
    n = len(df)
    
    p = length // 2
    upper_val, lower_val = np.nan, np.nan
    upper_crossed, lower_crossed = True, True
    bias = "NEUTRAL"
    
    for i in range(length, n):
        if high[i-p] == np.max(high[i-length+1 : i+1]):
            upper_val, upper_crossed = high[i-p], False
            
        if low[i-p] == np.min(low[i-length+1 : i+1]):
            lower_val, lower_crossed = low[i-p], False
            
        if not np.isnan(upper_val) and not upper_crossed and close[i] > upper_val:
            bias = "BULLISH (BOS/CHoCH)"
            upper_crossed = True
            
        if not np.isnan(lower_val) and not lower_crossed and close[i] < lower_val:
            bias = "BEARISH (BOS/CHoCH)"
            lower_crossed = True
            
    return bias

def render_confluence_hub_section(df_data, symbol_name, current_tf):
    st.markdown('<div class="confluence-container">', unsafe_allow_html=True)
    st.markdown(f"### 🛡️ Independent Multi-Indicator Confluence Matrix ({symbol_name} - {current_tf})")
    st.caption("Custom indicator logic parsed from PineScript into Python Engine.")
    
    if df_data is None or len(df_data) < 30:
        st.warning("Insufficient data to calculate indicator signals.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
        
    st_sig = calculate_supertrend(df_data)
    ut_sig = calculate_ut_bot(df_data)
    ms_sig = calculate_market_structure(df_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color_st = "#00c853" if "BULLISH" in st_sig else "#c62828"
        st.markdown(f"""<div class="ind-card"><h4>1. Supertrend Indicator</h4><h3 style="color:{color_st};">{st_sig}</h3><p>ATR Period: 10 | Multiplier: 3.0</p></div>""", unsafe_allow_html=True)
        
    with col2:
        color_ut = "#00c853" if "BUY" in ut_sig else "#c62828"
        st.markdown(f"""<div class="ind-card"><h4>2. UT Bot Alerts</h4><h3 style="color:{color_ut};">{ut_sig}</h3><p>Key Value: 1 | ATR Period: 10</p></div>""", unsafe_allow_html=True)
        
    with col3:
        color_ms = "#00c853" if "BULLISH" in ms_sig else ("#c62828" if "BEARISH" in ms_sig else "#ff9800")
        st.markdown(f"""<div class="ind-card"><h4>3. LuxAlgo Market Structure</h4><h3 style="color:{color_ms};">{ms_sig}</h3><p>Fractal Length: 5 (BOS/CHoCH)</p></div>""", unsafe_allow_html=True)
        
    bull_count = sum(1 for s in [st_sig, ut_sig, ms_sig] if "BULLISH" in s or "BUY" in s)
    bear_count = sum(1 for s in [st_sig, ut_sig, ms_sig] if "BEARISH" in s or "SELL" in s)
    
    st.write("")
    if bull_count >= 2:
        final_conf = "🔥 STRONG CONFLUENCE BUY SIGNAL"
        conf_badge = "strong-buy"
    elif bear_count >= 2:
        final_conf = "🔻 STRONG CONFLUENCE SELL SIGNAL"
        conf_badge = "strong-sell"
    else:
        final_conf = "⚠️ MIXED SIGNALS / WAIT"
        conf_badge = "neutral"
        
    st.markdown(f"""<div style="text-align:center; padding: 1rem; background-color:#161b22; border-radius:12px; margin-top:10px;"><h2>Aggregated Signal: <span class="signal-badge {conf_badge}" style="font-size:1.3rem;">{final_conf}</span></h2><p style="color:#8b949e; margin-top:5px;">Confluence Score: {bull_count}/3 Bullish | {bear_count}/3 Bearish</p></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== NATIVE GOOGLE GEMINI VISION ENGINE ====================

def analyze_chart_vision_native_google(image, symbol, tf):
    gemini_key = st.secrets.get("GEMINI_API_KEY")
    if not gemini_key:
        return "⚠️ Please add GEMINI_API_KEY in Streamlit Secrets."
        
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    prompt = f"Analyze chart for {symbol} ({tf}). Predict NEXT candle (Bullish Green or Bearish Red) with a brief 2-line price action reason."
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": img_b64}}
            ]
        }]
    }
    
    candidate_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    last_error = ""
    for model_name in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=12)
            res_json = response.json()
            if response.status_code == 200 and 'candidates' in res_json:
                text_out = res_json['candidates'][0]['content']['parts'][0]['text']
                return f"⚡ **Gemini Vision Result ({model_name}):**\n\n{text_out}"
            else:
                last_error = res_json.get('error', {}).get('message', str(res_json))
        except Exception as e:
            last_error = str(e)
                
    return f"❌ Gemini Native API Error: {last_error}"

# ==================== DIRECT MULTI-PROVIDER AI ROUTING ====================

def get_ai_next_candle_opinion(provider_name, symbol, tf, signal, metrics):
    prompt = f"Asset: {symbol}, TF: {tf}, Signal: {signal}, Metrics: {metrics}. Predict NEXT immediate candle (Green/Red) with a 1-sentence price action reason."
    
    try:
        if "Gemini" in provider_name:
            gemini_key = st.secrets.get("GEMINI_API_KEY")
            if not gemini_key:
                return False, "GEMINI_API_KEY Missing in Secrets"
            for m_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={gemini_key}"
                try:
                    res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=8).json()
                    if 'candidates' in res:
                        return True, res['candidates'][0]['content']['parts'][0]['text'].strip()
                except Exception:
                    continue
            return False, "Gemini API failed."

        elif "Groq" in provider_name or "Llama" in provider_name:
            groq_key = st.secrets.get("GROQ_API_KEY")
            if not groq_key:
                return False, "GROQ_API_KEY Missing in Secrets"
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
            res = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], timeout=8)
            return True, res.choices[0].message.content.strip()

        elif "DeepSeek" in provider_name:
            ds_key = st.secrets.get("DEEPSEEK_API_KEY")
            if not ds_key:
                return False, "DEEPSEEK_API_KEY Missing in Secrets"
            client = OpenAI(base_url="https://api.deepseek.com", api_key=ds_key)
            res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], timeout=10)
            return True, res.choices[0].message.content.strip()

        elif "Mistral" in provider_name:
            mistral_key = st.secrets.get("MISTRAL_API_KEY")
            if not mistral_key:
                return False, "MISTRAL_API_KEY Missing in Secrets"
            headers = {"Authorization": f"Bearer {mistral_key}", "Content-Type": "application/json"}
            data = {"model": "mistral-small-latest", "messages": [{"role": "user", "content": prompt}]}
            res = requests.post("https://api.mistral.ai/v1/chat/completions", json=data, headers=headers, timeout=10).json()
            if 'choices' in res:
                return True, res['choices'][0]['message']['content'].strip()
            return False, f"Mistral Error: {res.get('message', 'Failed')}"

        elif "HuggingFace" in provider_name or "Qwen" in provider_name:
            hf_key = st.secrets.get("HF_API_KEY")
            if not hf_key:
                return False, "HF_API_KEY Missing in Secrets"
            headers = {"Authorization": f"Bearer {hf_key}"}
            hf_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"
            res = requests.post(hf_url, json={"inputs": prompt}, headers=headers, timeout=10).json()
            if isinstance(res, list) and len(res) > 0:
                return True, res[0].get('generated_text', str(res)).strip()
            return True, str(res).strip()

    except Exception as e:
        return False, f"Provider Error ({str(e)[:35]}...)"

# ==================== STREAMLIT UI ====================

head_col, btn_col = st.columns([3, 1])
with head_col:
    st.markdown('<h1 class="main-header">⚡ Quantum AI Signal Engine</h1>', unsafe_allow_html=True)
    st.caption(f"PKT: {get_pakistan_time()} | Active Focus: {st.session_state.selected_symbol}")
with btn_col:
    st.write("")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()

# Pair Selection Top Cards
cols = st.columns(3)
for idx, (disp, meta) in enumerate(MAIN_SYMBOLS.items()):
    with cols[idx % 3]:
        q_df = fetch_ohlcv(meta["yf_ticker"], interval="15m")
        p, q_sig, badge = 0.0, "WAIT", "neutral"
        if q_df is not None:
            anal = calculate_quant_signals(q_df)
            if anal:
                p, q_sig, badge = anal["price"], anal["signal"], anal["badge_class"]
        
        is_active = (st.session_state.selected_symbol == disp)
        border_style = "border: 2px solid #00f2fe; background-color: #1c2333;" if is_active else ""
        
        card_html = f"""<div class="symbol-card" style="{border_style}">
            <strong>{meta['display']}</strong><br>
            <span style="font-size:1.4rem; font-weight:700;">{p:,.2f}</span><br>
            
