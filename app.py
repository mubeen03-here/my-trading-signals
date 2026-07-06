# Free Professional Trading Signals Dashboard
# Built with Grok AI — Best free personal scalping signal tool
# Data: yfinance (free, reliable, close to TradingView)
# Run locally: streamlit run app.py
# Deploy free 24/7: See README.md (GitHub + Streamlit Cloud — 5-10 min, no coding needed)

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime
import time

# ==================== PAGE CONFIG & THEME ====================
st.set_page_config(
    page_title="Free Pro Scalping Signals | Gold • BTC • USDJPY • NAS100",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Trading Theme (clean & modern)
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .main-header { 
        font-size: 2.2rem; font-weight: 700; 
        background: linear-gradient(90deg, #00ff9f, #00b8ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .symbol-card {
        background-color: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 1rem; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .signal-badge {
        padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 700;
        font-size: 0.9rem; display: inline-block; margin: 0.3rem 0;
    }
    .strong-buy { background-color: #00c853; color: white; }
    .buy { background-color: #4caf50; color: white; }
    .neutral { background-color: #ff9800; color: white; }
    .sell { background-color: #f44336; color: white; }
    .strong-sell { background-color: #d32f2f; color: white; }
    .metric-value { font-size: 1.6rem; font-weight: 700; }
    .section-header { 
        font-size: 1.4rem; font-weight: 600; color: #00ff9f; 
        border-bottom: 2px solid #30363d; padding-bottom: 0.3rem; margin: 1rem 0 0.5rem 0;
    }
    .trade-box {
        background-color: #161b22; border: 2px solid #00b8ff; border-radius: 10px;
        padding: 1rem; margin: 0.5rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00b8ff, #00ff9f); color: black; font-weight: 700;
        border: none; border-radius: 8px; padding: 0.5rem 1.5rem;
    }
    .stButton>button:hover { filter: brightness(1.1); }
    .info-text { color: #8b949e; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None
if "custom_symbols" not in st.session_state:
    st.session_state.custom_symbols = {}
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")

# ==================== MAIN SYMBOLS (Always Top) ====================
MAIN_SYMBOLS = {
    "Gold (XAUUSD)": {"ticker": "XAUUSD=X", "display": "XAU/USD", "category": "Commodity"},
    "Bitcoin (BTC)": {"ticker": "BTC-USD", "display": "BTC/USD", "category": "Crypto"},
    "USD/JPY": {"ticker": "USDJPY=X", "display": "USD/JPY", "category": "Forex"},
    "NAS100": {"ticker": "NQ=F", "display": "NAS100 (NQ Futures)", "category": "Index"},
}

def get_all_symbols():
    symbols = MAIN_SYMBOLS.copy()
    symbols.update(st.session_state.custom_symbols)
    return symbols

# ==================== DATA FETCH (Cached) ====================
@st.cache_data(ttl=45, show_spinner=False)
def fetch_ohlcv(ticker: str, interval: str = "15m", period: str = "5d"):
    """Fetch OHLCV data from yfinance. Handles column cleaning."""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=True
        )
        if df is None or df.empty:
            return None
        
        df = df.reset_index()
        
        # Normalize column names (handle multi-index from yfinance)
        df.columns = [str(c[0]).capitalize() if isinstance(c, tuple) else str(c).capitalize() 
                      for c in df.columns]
        
        # Standardize common names
        rename_map = {}
        for col in df.columns:
            if "datetime" in col.lower() or "date" in col.lower():
                rename_map[col] = "Datetime"
            elif col.lower() == "close":
                rename_map[col] = "Close"
            elif col.lower() == "open":
                rename_map[col] = "Open"
            elif col.lower() == "high":
                rename_map[col] = "High"
            elif col.lower() == "low":
                rename_map[col] = "Low"
            elif col.lower() == "volume":
                rename_map[col] = "Volume"
        
        df = df.rename(columns=rename_map)
        
        # Ensure we have required columns
        required = ["Close", "High", "Low", "Open"]
        for req in required:
            if req not in df.columns:
                if "Adj Close" in df.columns:
                    df[req] = df["Adj Close"]
                else:
                    return None
        
        if "Datetime" not in df.columns:
            df["Datetime"] = pd.to_datetime(df.index)
            
        return df[["Datetime", "Open", "High", "Low", "Close", "Volume"]].dropna()
        
    except Exception as e:
        st.warning(f"Data fetch issue for {ticker}: {str(e)[:80]}")
        return None

# ==================== SIGNAL ENGINE (Professional Confluence) ====================
def calculate_signal_and_levels(df: pd.DataFrame, current_price: float = None):
    """
    Professional rule-based signal engine.
    Score based on multiple confluences → Weak / Medium / Strong.
    Realistic SL/TP using ATR + recent swings + classic pivots.
    """
    if df is None or len(df) < 25:
        return None
    
    df = df.copy()
    
    # === INDICATORS ===
    df.ta.ema(close="Close", length=9, append=True)
    df.ta.ema(close="Close", length=21, append=True)
    df.ta.rsi(close="Close", length=14, append=True)
    df.ta.macd(close="Close", fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(close="Close", length=20, std=2, append=True)
    df.ta.atr(close="Close", length=14, append=True)
    
    df = df.dropna()
    if len(df) < 10:
        return None
    
    last = df.iloc[-1]
    price = current_price or float(last["Close"])
    
    ema9 = float(last.get("EMA_9", price))
    ema21 = float(last.get("EMA_21", price))
    rsi = float(last.get("RSI_14", 50))
    macd_hist = float(last.get("MACDh_12_26_9", 0))
    bb_upper = float(last.get("BBU_20_2.0", price * 1.02))
    bb_lower = float(last.get("BBL_20_2.0", price * 0.98))
    atr = float(last.get("ATRr_14", price * 0.01))
    
    # Recent swing levels (last ~20-30 candles)
    lookback = min(25, len(df))
    recent_high = float(df["High"].tail(lookback).max())
    recent_low = float(df["Low"].tail(lookback).min())
    
    # Classic Pivot Points (from last completed bar)
    high = float(last["High"])
    low = float(last["Low"])
    close_p = float(last["Close"])
    pp = (high + low + close_p) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    
    # === CONFLUENCE SCORING (Bullish = positive) ===
    score = 0
    reasons = []
    
    # 1. EMA Trend Alignment (strong weight)
    if price > ema9 > ema21:
        score += 2
        reasons.append("✅ Strong bullish EMA alignment (Price > EMA9 > EMA21)")
    elif price > ema9:
        score += 1
        reasons.append("✅ Price trading above short-term EMA9")
    elif price < ema9 < ema21:
        score -= 2
        reasons.append("❌ Bearish EMA alignment (Price < EMA9 < EMA21)")
    
    # 2. RSI Momentum
    if rsi > 55:
        score += 1
        reasons.append("✅ RSI showing bullish momentum (>55)")
    elif rsi < 45:
        score -= 1
        reasons.append("❌ RSI showing bearish momentum (<45)")
    
    # 3. MACD Histogram
    if macd_hist > 0:
        score += 1
        reasons.append("✅ MACD histogram positive → bullish momentum building")
    else:
        score -= 1
        reasons.append("❌ MACD histogram negative → bearish pressure")
    
    # 4. Bollinger Band position (scalping edge)
    if price <= bb_lower * 1.005:
        score += 1
        reasons.append("✅ Price near/ at lower Bollinger Band (bounce potential)")
    elif price >= bb_upper * 0.995:
        score -= 1
        reasons.append("❌ Price near upper Bollinger Band (pullback risk)")
    
    # 5. Price vs recent structure
    if price > recent_high * 0.995:
        score += 1
        reasons.append("✅ Breaking or near recent swing high (momentum continuation)")
    elif price < recent_low * 1.005:
        score -= 1
        reasons.append("❌ Breaking or near recent swing low")
    
    # === FINAL SIGNAL ===
    if score >= 4:
        signal_type = "STRONG BUY"
        badge_class = "strong-buy"
    elif score >= 2:
        signal_type = "BUY"
        badge_class = "buy"
    elif score <= -4:
        signal_type = "STRONG SELL"
        badge_class = "strong-sell"
    elif score <= -2:
        signal_type = "SELL"
        badge_class = "sell"
    else:
        signal_type = "NEUTRAL"
        badge_class = "neutral"
    
    # === TRADE PLAN (Realistic) ===
    if "BUY" in signal_type:
        entry = round(price, 2)
        # SL below recent structure or S1, buffered by 0.6 ATR
        sl = round(min(recent_low, s1) - (atr * 0.6), 2)
        risk = max(entry - sl, atr * 0.8)
        tp1 = round(entry + risk * 1.6, 2)
        tp2 = round(entry + risk * 2.5, 2)
        tp3 = round(max(r1, entry + risk * 3.5), 2)
        rr_text = "1 : 1.6+  |  Good RR"
    elif "SELL" in signal_type:
        entry = round(price, 2)
        sl = round(max(recent_high, r1) + (atr * 0.6), 2)
        risk = max(sl - entry, atr * 0.8)
        tp1 = round(entry - risk * 1.6, 2)
        tp2 = round(entry - risk * 2.5, 2)
        tp3 = round(min(s1, entry - risk * 3.5), 2)
        rr_text = "1 : 1.6+  |  Good RR"
    else:
        entry = sl = tp1 = tp2 = tp3 = round(price, 2)
        rr_text = "Wait for clearer setup"
    
    return {
        "signal": signal_type,
        "badge_class": badge_class,
        "score": score,
        "reasons": reasons,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rr": rr_text,
        "support": round(s1, 2),
        "resistance": round(r1, 2),
        "recent_support": round(recent_low, 2),
        "recent_resistance": round(recent_high, 2),
        "rsi": round(rsi, 1),
        "ema9": round(ema9, 2),
        "ema21": round(ema21, 2),
        "atr": round(atr, 2),
        "last_price": round(price, 2),
    }

# ==================== CHART BUILDER ====================
def build_chart(df: pd.DataFrame, analysis: dict, symbol_name: str, tf: str):
    if df is None or analysis is None:
        return None
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["Datetime"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
        increasing_line_color="#00c853",
        decreasing_line_color="#f44336",
        increasing_fillcolor="#00c853",
        decreasing_fillcolor="#f44336",
    ))
    
    # EMAs
    if "EMA_9" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["EMA_9"],
            line=dict(color="#00b8ff", width=2),
            name="EMA 9"
        ))
    if "EMA_21" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["EMA_21"],
            line=dict(color="#ff9800", width=2),
            name="EMA 21"
        ))
    
    # Key Levels
    price = analysis["last_price"]
    fig.add_hline(y=analysis["support"], line_dash="dash", line_color="#00c853",
                  annotation_text="Support", annotation_position="bottom right")
    fig.add_hline(y=analysis["resistance"], line_dash="dash", line_color="#f44336",
                  annotation_text="Resistance", annotation_position="top right")
    fig.add_hline(y=analysis["recent_support"], line_dash="dot", line_color="#4caf50", 
                  annotation_text="Recent Low", annotation_position="bottom left")
    fig.add_hline(y=analysis["recent_resistance"], line_dash="dot", line_color="#ff5252",
                  annotation_text="Recent High", annotation_position="top left")
    
    # Entry / SL / TP markers (if not neutral)
    if "BUY" in analysis["signal"]:
        fig.add_hline(y=analysis["entry"], line_color="#00ff9f", line_width=2,
                      annotation_text="Entry", annotation_position="bottom right")
        fig.add_hline(y=analysis["sl"], line_color="#f44336", line_width=1.5, line_dash="dash",
                      annotation_text="SL", annotation_position="bottom left")
        fig.add_hline(y=analysis["tp1"], line_color="#00c853", line_width=1.5,
                      annotation_text="TP1", annotation_position="top right")
    elif "SELL" in analysis["signal"]:
        fig.add_hline(y=analysis["entry"], line_color="#ff5252", line_width=2,
                      annotation_text="Entry", annotation_position="top right")
        fig.add_hline(y=analysis["sl"], line_color="#00c853", line_width=1.5, line_dash="dash",
                      annotation_text="SL", annotation_position="top left")
        fig.add_hline(y=analysis["tp1"], line_color="#f44336", line_width=1.5,
                      annotation_text="TP1", annotation_position="bottom right")
    
    fig.update_layout(
        title=f"{symbol_name} — {tf} | Signal: {analysis['signal']}",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )
    
    return fig

# ==================== UI: HEADER ====================
st.markdown('<h1 class="main-header">📈 Free Pro Scalping Signals</h1>', unsafe_allow_html=True)
st.caption("Gold • BTC • USDJPY • NAS100 + Custom Pairs  |  Live Technical Analysis  |  Built with Grok AI")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown(f"**Last Updated:** {st.session_state.last_refresh} PKT")
with col2:
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
        st.rerun()
with col3:
    st.markdown('<span class="info-text">Data via yfinance • Verify with your broker</span>', unsafe_allow_html=True)

st.divider()

# ==================== SIDEBAR: CUSTOM PAIRS ====================
with st.sidebar:
    st.header("➕ Add Custom Pair")
    st.caption("Any symbol supported by yfinance (e.g. EURUSD=X, AAPL, GC=F, ETH-USD)")
    
    custom_ticker = st.text_input("yFinance Ticker", placeholder="EURUSD=X")
    custom_display = st.text_input("Display Name (optional)", placeholder="EUR/USD")
    
    if st.button("Add to Dashboard", use_container_width=True):
        if custom_ticker:
            disp = custom_display.strip() or custom_ticker
            st.session_state.custom_symbols[disp] = {
                "ticker": custom_ticker.strip(),
                "display": disp,
                "category": "Custom"
            }
            st.success(f"✅ Added {disp}")
            time.sleep(0.6)
            st.rerun()
        else:
            st.warning("Please enter a ticker")
    
    if st.session_state.custom_symbols:
        st.divider()
        st.subheader("Your Custom Pairs")
        for disp, meta in list(st.session_state.custom_symbols.items()):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"• {disp}")
            with c2:
                if st.button("🗑️", key=f"del_{disp}"):
                    del st.session_state.custom_symbols[disp]
                    st.rerun()
    
    st.divider()
    st.caption("💡 Tip: Use 5m/15m for scalping, 1h/4h for swing confirmation")

# ==================== MAIN SYMBOLS GRID ====================
st.markdown('<div class="section-header">🔥 Main Symbols — Quick View</div>', unsafe_allow_html=True)

all_symbols = get_all_symbols()
cols = st.columns(4)

for idx, (disp_name, meta) in enumerate(list(MAIN_SYMBOLS.items()) + list(st.session_state.custom_symbols.items())):
    col = cols[idx % 4]
    
    with col:
        ticker = meta["ticker"]
        display = meta.get("display", disp_name)
        
        # Quick data for card
        quick_df = fetch_ohlcv(ticker, interval="60m", period="2d")
        
        price = 0
        pct_change = 0
        signal_badge = "NEUTRAL"
        badge_cls = "neutral"
        
        if quick_df is not None and len(quick_df) > 1:
            price = float(quick_df["Close"].iloc[-1])
            first_close = float(quick_df["Close"].iloc[0])
            pct_change = ((price - first_close) / first_close) * 100 if first_close != 0 else 0
            
            # Quick signal (15m)
            quick_analysis = calculate_signal_and_levels(quick_df)
            if quick_analysis:
                signal_badge = quick_analysis["signal"]
                badge_cls = quick_analysis["badge_class"]
        
        # Card UI
        st.markdown(f"""
        <div class="symbol-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong style="font-size:1.1rem;">{display}</strong><br>
                    <span class="info-text">{meta['category']}</span>
                </div>
                <div style="text-align:right;">
                    <div class="metric-value">{price:,.2f}</div>
                    <div style="color:{'#00c853' if pct_change >= 0 else '#f44336'}; font-weight:600;">
                        {pct_change:+.2f}%
                    </div>
                </div>
            </div>
            <div style="margin-top:0.6rem;">
                <span class="signal-badge {badge_cls}">{signal_badge}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"📊 View Full Analysis", key=f"btn_{disp_name}", use_container_width=True):
            st.session_state.selected_symbol = disp_name
            st.rerun()

# ==================== DETAILED ANALYSIS VIEW ====================
if st.session_state.selected_symbol:
    selected = st.session_state.selected_symbol
    meta = all_symbols.get(selected, {})
    ticker = meta.get("ticker", "")
    display = meta.get("display", selected)
    
    st.divider()
    st.markdown(f'<div class="section-header">📊 Detailed Analysis — {display}</div>', unsafe_allow_html=True)
    
    # Timeframe selector
    tf_col1, tf_col2 = st.columns([1, 3])
    with tf_col1:
        tf = st.selectbox(
            "Timeframe",
            options=["5m", "15m", "1h", "4h"],
            index=1,
            help="5m/15m = Scalping | 1h/4h = Higher timeframe confirmation"
        )
    
    period_map = {"5m": "3d", "15m": "5d", "1h": "14d", "4h": "30d"}
    
    with st.spinner(f"Fetching {tf} data for {display}..."):
        df = fetch_ohlcv(ticker, interval=tf, period=period_map[tf])
    
    if df is None or len(df) < 20:
        st.error("Not enough data for this symbol/timeframe. Try another timeframe or symbol.")
    else:
        analysis = calculate_signal_and_levels(df)
        
        if analysis is None:
            st.error("Could not calculate analysis. Try refreshing data.")
        else:
            # Top summary bar
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Current Price", f"{analysis['last_price']:,.2f}")
            with c2:
                st.metric("Signal", analysis["signal"])
            with c3:
                st.metric("RSI (14)", analysis["rsi"])
            with c4:
                st.metric("ATR (14)", analysis["atr"])
            
            # Main content: Chart + Trade Setup
            chart_col, setup_col = st.columns([3, 2])
            
            with chart_col:
                fig = build_chart(df, analysis, display, tf)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            with setup_col:
                st.markdown("### 🎯 Trade Setup")
                
                # Signal badge
                st.markdown(f"""
                <div style="text-align:center; margin:0.5rem 0;">
                    <span class="signal-badge {analysis['badge_class']}" style="font-size:1.3rem; padding:0.4rem 1.5rem;">
                        {analysis['signal']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Confluence Score:** {analysis['score']} / 5")
                
                # Trade levels
                st.markdown(f"""
                <div class="trade-box">
                    <strong>Entry:</strong> {analysis['entry']}<br>
                    <strong>Stop Loss (SL):</strong> {analysis['sl']}<br><br>
                    <strong>Take Profit 1:</strong> {analysis['tp1']}<br>
                    <strong>Take Profit 2:</strong> {analysis['tp2']}<br>
                    <strong>Take Profit 3:</strong> {analysis['tp3']}<br><br>
                    <strong>Risk : Reward</strong> → {analysis['rr']}
                </div>
                """, unsafe_allow_html=True)
                
                # Copy friendly
                st.code(f"""Entry: {analysis['entry']}
SL: {analysis['sl']}
TP1: {analysis['tp1']}   |   TP2: {analysis['tp2']}   |   TP3: {analysis['tp3']}
Support: {analysis['support']}   |   Resistance: {analysis['resistance']}""", language="text")
                
                if st.button("📋 Copy Levels to Clipboard (select & Ctrl+C)", use_container_width=True):
                    st.toast("Levels ready — select text above and copy!")
            
            # Why this signal
            st.markdown("### 🧠 Why This Signal?")
            for reason in analysis["reasons"]:
                st.write(reason)
            
            if analysis["signal"] == "NEUTRAL":
                st.info("Market is consolidating. Wait for clearer direction or lower timeframe confirmation.")
            
            # Key Levels & Indicators
            st.markdown("### 📌 Key Levels & Indicators")
            lcol1, lcol2 = st.columns(2)
            
            with lcol1:
                st.write("**Support & Resistance**")
                st.write(f"• Pivot Support (S1): {analysis['support']}")
                st.write(f"• Pivot Resistance (R1): {analysis['resistance']}")
                st.write(f"• Recent Swing Low: {analysis['recent_support']}")
                st.write(f"• Recent Swing High: {analysis['recent_resistance']}")
            
            with lcol2:
                st.write("**Technical Summary**")
                st.write(f"• EMA 9 / EMA 21: {analysis['ema9']} / {analysis['ema21']}")
                st.write(f"• RSI (14): {analysis['rsi']}")
                st.write(f"• ATR (14) — Volatility: {analysis['atr']}")
            
            st.caption("⚠️ This is educational/technical analysis only. Not financial advice. Always do your own research and manage risk properly. Past performance ≠ future results.")

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style="text-align:center; color:#8b949e; font-size:0.8rem;">
    Free Professional Trading Signals Dashboard • Powered by Grok AI<br>
    Data source: yfinance (free public data) • Close to TradingView prices but always verify on your broker<br>
    Best used for personal learning & decision support • Not a signal service
</div>
""", unsafe_allow_html=True)

# Auto refresh hint
st.caption("Tip: Click 'Refresh All Data' button after some time for latest prices & signals.")