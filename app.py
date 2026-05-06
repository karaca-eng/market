import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# ==================== CONFIGURATION ====================
MIN_VOL_3M = 40000
MIN_CHG_3M = 1.0
CONFIRM_CHG_15M = 1.3
FAST_STRIKE_CHG = 1.0
TRI_WINDOW = 180
MAX_DISPLAY_ROWS = 100
FETCH_INTERVAL = 10
PUMP_DUMP_THRESHOLD = 2.2

# MACD Paralel Ayarları
MACD_MIN_CANDLES = 3
MACD_MAX_CANDLES = 8
MACD_COOLDOWN = 180
MACD_EXECUTOR = ThreadPoolExecutor(max_workers=15)

# BIG MOVE AYARLARI
BIGMOVE_EXECUTOR = ThreadPoolExecutor(max_workers=20)
BIGMOVE_COOLDOWN = 600
BB_SQUEEZE_LOOKBACK = 100
BB_SQUEEZE_PERCENTILE = 5
MA200_MIN_BARS_BELOW = 20
MACD_RESISTANCE_LOOKBACK = 20

BINANCE_REST_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
]

# ==================== SESSION & RETRY SETUP ====================
def create_session():
    """418 ve rate-limit hatalarına karşı retry mekanizmalı session oluşturur."""
    session = requests.Session()
    
    # Gerçek tarayıcı header'ları
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9,tr;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.tradingview.com/',
        'Origin': 'https://www.tradingview.com',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'cross-site',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
    })
    
    # Retry stratejisi: 418, 429, 502, 503, 504 için tekrar dene
    retry_strategy = Retry(
        total=3,
        backoff_factor=1.5,  # 1.5s, 3s, 6s bekleme
        status_forcelist=[418, 429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

# ==================== HELPERS ====================
def get_signal_label(direction: str, chg: float) -> str:
    if abs(chg) >= PUMP_DUMP_THRESHOLD:
        return "PUMP" if direction == "up" else "DUMP"
    return "BUY" if direction == "up" else "SELL"


# ==================== CORE CLASS ====================
class MarketRadar:
    def __init__(self):
        self.history = {}
        self.signals = []
        self.stats_hourly = {}
        self.stats_4h = {}
        self.lock = threading.RLock()
        self.last_heartbeat = 0
        self.total_pairs = 0
        self.last_reset_hour = datetime.now().hour
        self.last_reset_4h_block = datetime.now().hour // 4
        self.rest_base_url = BINANCE_REST_URLS[0]
        self.price_cache_15m = {}
        self.debug_log = []
        self.session = create_session()  # YENİ: Session tabanlı istekler

        # MACD state
        self.macd_sent = {}
        self.macd_sent_keys = {}
        self.macd_candidates = {}
        self.macd_last_trigger = {}

        # BIG MOVE state
        self.bigmove_signals = []
        self.bigmove_sent = {}
        self.bigmove_candidates = {}
        self.bigmove_last_trigger = {}

    def log(self, msg):
        t = datetime.now().strftime("%H:%M:%S")
        full = f"[{t}] {msg}"
        print(full, flush=True)
        with self.lock:
            self.debug_log.insert(0, full)
            if len(self.debug_log) > 50:
                self.debug_log.pop()

    def get_working_rest_url(self):
        for url in BINANCE_REST_URLS:
            try:
                # Session kullan, timeout artır
                r = self.session.get(f"{url}/fapi/v1/ping", timeout=5)
                if r.status_code == 200:
                    self.rest_base_url = url
                    self.log(f"✅ Binance bağlantısı OK: {url}")
                    return url
                else:
                    self.log(f"⚠️ {url} → status {r.status_code}")
            except Exception as e:
                self.log(f"❌ {url} → HATA: {e}")
            time.sleep(0.5)  # URL'ler arası bekle
        self.log("🔴 Hiçbir Binance URL'e bağlanılamadı!")
        return self.rest_base_url

    def _safe_request(self, url, timeout=5):
        """418 hatalarını önlemek için güvenli istek wrapper'ı."""
        try:
            # İstek öncesi kısa bekleme (rate limit koruması)
            time.sleep(0.05)
            response = self.session.get(url, timeout=timeout)
            if response.status_code == 418:
                self.log(f"⚠️ 418 TEAPOT alındı, 2sn bekleniyor... → {url[:60]}")
                time.sleep(2.0)
                response = self.session.get(url, timeout=timeout)
            return response
        except Exception as e:
            self.log(f"❌ İstek hatası: {e}")
            return None

    def check_resets(self):
        now = datetime.now()
        if now.hour != self.last_reset_hour:
            self.stats_hourly.clear()
            self.last_reset_hour = now.hour
        if (now.hour // 4) != self.last_reset_4h_block:
            self.stats_4h.clear()
            self.last_reset_4h_block = now.hour // 4

    def process_ticker(self, data):
        now = time.time()
        with self.lock:
            self.check_resets()
            self.last_heartbeat = now
            self.total_pairs = len(data)
            for item in data:
                symbol = item['s']
                if not symbol.endswith('USDT'):
                    continue
                price, quote_vol = float(item['c']), float(item['q'])
                if symbol not in self.history:
                    self.history[symbol] = deque(maxlen=400)
                self.history[symbol].append((now, price, quote_vol))
                self.check_logic(symbol, now)
                self._maybe_trigger_macd(symbol, price, now)
                self._maybe_trigger_bigmove(symbol, price, now)

    def check_logic(self, symbol, now):
        hist = list(self.history[symbol])
        if len(hist) < 5:
            return

        current = hist[-1]
        past_1m = next((x for x in reversed(hist) if now - x[0] >= 60), hist[0])
        past_3m = next((x for x in reversed(hist) if now - x[0] >= TRI_WINDOW), hist[0])

        c1 = ((current[1] - past_1m[1]) / past_1m[1]) * 100
        c3 = ((current[1] - past_3m[1]) / past_3m[1]) * 100
        vol_3m = current[2] - past_3m[2]
        vol_1m = current[2] - past_1m[2]

        if abs(c1) >= FAST_STRIKE_CHG and vol_1m >= 50000:
            direction = "up" if c1 > 0 else "down"
            label = get_signal_label(direction, c1)
            self.add_signal(symbol, current[1], c1, 0, vol_1m, label, "⚡ FLASH", score=40)
            return

        if vol_3m >= MIN_VOL_3M and abs(c3) >= MIN_CHG_3M:
            price_15m_ago = self.get_15m_price(symbol)
            if price_15m_ago:
                c15 = ((current[1] - price_15m_ago) / price_15m_ago) * 100
                is_consistent = (c3 > 0 and c15 > 0) or (c3 < 0 and c15 < 0)
                if is_consistent and abs(c15) >= CONFIRM_CHG_15M:
                    direction = "up" if c3 > 0 else "down"
                    label = get_signal_label(direction, c3)
                    self.add_signal(symbol, current[1], c3, c15, vol_3m, label, "💎 CONFIRMED", score=55)

    def get_15m_price(self, symbol):
        now = time.time()
        if symbol in self.price_cache_15m:
            cache_time, price = self.price_cache_15m[symbol]
            if now - cache_time < 300:
                return price
        try:
            url = f"{self.rest_base_url}/fapi/v1/klines?symbol={symbol}&interval=15m&limit=2"
            response = self._safe_request(url, timeout=3)  # YENİ: safe_request kullan
            if response and response.status_code == 200:
                price = float(response.json()[0][1])
                self.price_cache_15m[symbol] = (now, price)
                return price
        except Exception as e:
            self.log(f"15m price hata ({symbol}): {e}")
        return None

    def add_signal(self, symbol, price, chg_main, chg_ref, vol, s_type, mode, score=50, macd_tag=None):
        t_str = datetime.now().strftime("%H:%M:%S")
        sym_clean = symbol.replace("USDT", "")
        is_up = s_type in ("PUMP", "BUY")
        stat_key = "PUMP" if is_up else "DUMP"

        with self.lock:
            for s in self.signals[:10]:
                if s.get('Symbol') == sym_clean and s.get('Mode') == mode:
                    return

            if sym_clean not in self.stats_hourly:
                self.stats_hourly[sym_clean] = {"PUMP": 0, "DUMP": 0}
            self.stats_hourly[sym_clean][stat_key] += 1

            if sym_clean not in self.stats_4h:
                self.stats_4h[sym_clean] = {"PUMP": 0, "DUMP": 0}
            self.stats_4h[sym_clean][stat_key] += 1

            self.signals.insert(0, {
                "Time": t_str,
                "Symbol": sym_clean,
                "Price": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                "Chg": chg_main,
                "Ref": chg_ref,
                "Vol": vol,
                "P/D": s_type,
                "Mode": mode,
                "Score": score,
                "SnapP": self.stats_4h[sym_clean]["PUMP"],
                "SnapD": self.stats_4h[sym_clean]["DUMP"],
                "MACD": macd_tag or "",
            })
            self.log(f"🚨 SİNYAL: {sym_clean} {s_type} {mode} {chg_main:+.2f}%" +
                     (f" | {macd_tag}" if macd_tag else ""))
            if len(self.signals) > MAX_DISPLAY_ROWS:
                self.signals.pop()


# ==================== STREAMLIT UI ====================
st.set_page_config(layout="wide", page_title="Market Radar Pro")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .status-live { color: #00ff88; font-weight: bold; border: 1px solid #00ff88; padding: 2px 10px; border-radius: 15px; font-size: 0.8rem; }
    .status-offline { color: #ff4b4b; font-weight: bold; border: 1px solid #ff4b4b; padding: 2px 10px; border-radius: 15px; font-size: 0.8rem; }
    .pump-label  { background-color: #00ff88; color: black;  padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .dump-label  { background-color: #ff4b4b; color: white;  padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .buy-label   { background-color: #1a7f4b; color: #afffcf; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .sell-label  { background-color: #7f1a1a; color: #ffcfcf; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .mode-confirmed { background-color: #1abc9c; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .mode-flash { background-color: #e67e22; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .mode-macd  { background-color: #8e44ad; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .mode-bigmove { background-color: #f39c12; color: #000; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .macd-tag   { background-color: #2c1654; color: #c39bd3; padding: 2px 7px; border-radius: 4px; font-size: 0.78rem; font-weight: bold; border: 1px solid #8e44ad; }
    .bigmove-tag { background-color: #3d2208; color: #f5b041; padding: 2px 7px; border-radius: 4px; font-size: 0.78rem; font-weight: bold; border: 1px solid #f39c12; }
    .stat-card { background-color: #1e2127; padding: 10px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #f1c40f; }
    .debug-box { background-color: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 10px; font-family: monospace; font-size: 0.75rem; color: #aaa; max-height: 200px; overflow-y: auto; }
    table { width: 100%; border-collapse: collapse; }
    th, td { white-space: nowrap; padding: 12px 15px; text-align: left; border-bottom: 1px solid #222; }
    .sym-link { color: #f1c40f; text-decoration: none; font-weight: bold; font-size: 1.1rem; }
    .sym-link:hover { color: #fff; }
    .green-arrow { color: #00ff88; font-weight: bold; }
    .red-arrow   { color: #ff4b4b; font-weight: bold; }
    .row-flash-pump { background-color: rgba(0, 255, 136, 0.22) !important; border-left: 5px solid #00ff88 !important; }
    .row-flash-dump { background-color: rgba(255, 75,  75,  0.22) !important; border-left: 5px solid #ff4b4b !important; }
    .row-conf-pump  { background-color: rgba(0, 255, 136, 0.08) !important; }
    .row-conf-dump  { background-color: rgba(255, 75,  75,  0.08) !important; }
    .row-macd       { background-color: rgba(142, 68, 173, 0.12) !important; border-left: 3px solid #8e44ad !important; }
    .row-bigmove    { background-color: rgba(243, 156, 18, 0.15) !important; border-left: 4px solid #f39c12 !important; }
    .macd-radar-card { background: #1a1030; border: 1px solid #8e44ad; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; }
    .macd-radar-sym { color: #c39bd3; font-weight: bold; font-size: 1rem; }
    .macd-radar-tag { color: #f0c3ff; font-size: 0.82rem; }
    .macd-radar-time { color: #666; font-size: 0.72rem; }
    .bigmove-card { background: #2a1d0a; border: 1px solid #f39c12; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }
    .bigmove-sym { color: #f5b041; font-weight: bold; font-size: 1.1rem; }
    .bigmove-cond { color: #f8c471; font-size: 0.85rem; }
    .bigmove-score { color: #fff; font-weight: bold; font-size: 0.9rem; }
    .bigmove-time { color: #888; font-size: 0.72rem; }
    .bigmove-radar-card { background: #1a1508; border: 1px solid #7f8c8d; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; }
    .bigmove-radar-sym { color: #d5dbdb; font-weight: bold; font-size: 1rem; }
    .bigmove-radar-cond { color: #aab7b8; font-size: 0.82rem; }
    </style>
""", unsafe_allow_html=True)

radar = get_radar_instance()
if "thread_started" not in st.session_state:
    t = threading.Thread(target=binance_worker, args=(radar,), daemon=True)
    t.start()
    st.session_state.thread_started = True
    radar.log(">>> UI: Thread baslatildi")

# ==================== NAVIGATION ====================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Sayfa Sec",
    ["📡 Normal Sinyaller", "📊 MACD Radar", "🎯 Big Move Hunter"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption("v2.0 | Market Radar Pro")

# Header ortak
h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
h1.title("📡 Market Radar Pro")

elapsed = time.time() - radar.last_heartbeat
status_html = (
    '<span class="status-live">● SYSTEM LIVE</span>'
    if elapsed < 10
    else '<span class="status-offline">● RECONNECTING</span>'
)
h2.markdown(f"<div style='margin-top:10px;'>{status_html}</div>", unsafe_allow_html=True)
h2.markdown(
    '<a href="https://x.com/SinyalEngineer" target="_blank" style="color:white; text-decoration:none;">𝕏 @SinyalEngineer</a>',
    unsafe_allow_html=True,
)
h3.metric("Pairs Tracked", radar.total_pairs)
h3.metric("Total Signals", len(radar.signals))
h4.metric("Big Moves", len(radar.bigmove_signals))

st.divider()

# DEBUG PANEL (ortak)
with st.expander("🔧 Debug Log", expanded=False):
    with radar.lock:
        logs = list(radar.debug_log)
    if logs:
        log_html = "<div class='debug-box'>" + "<br>".join(logs) + "</div>"
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.info("Henüz log yok...")

st.divider()

# ================================================================
# SAYFA 1: NORMAL SINYALLER
# ================================================================
if page == "📡 Normal Sinyaller":
    h1.caption("⚡ Flash: Anlık hareket | 💎 Confirmed: 3dk+15dk | 📊 MACD: Paralel yukselis (3-8 mum)")

    col_filters = st.columns([1, 1, 1, 1])
    mode_filter = col_filters[0].multiselect(
        "Sinyal Modu",
        ["⚡ FLASH", "💎 CONFIRMED", "📊 MACD"],
        default=["⚡ FLASH", "💎 CONFIRMED", "📊 MACD"],
        key="mode_filter"
    )
    pd_filter = col_filters[1].multiselect(
        "Yon",
        ["PUMP", "BUY", "DUMP", "SELL"],
        default=["PUMP", "BUY", "DUMP", "SELL"],
        key="pd_filter"
    )
    search_query = col_filters[2].text_input("🔍 Symbol Filter", placeholder="BTC...", key="search").upper()
    macd_only = col_filters[3].checkbox("Sadece MACD etiketli sinyaller", value=False)

    st.divider()

    col_side, col_main, col_macd = st.columns([1, 4, 1])

    with col_side:
        st.subheader("🔥 Top 5 Activity")
        side_placeholder = st.empty()

    with col_main:
        st.subheader("📡 Intelligence Stream")
        main_placeholder = st.empty()

    def get_mode_css_class(mode):
        if "CONFIRMED" in mode:
            return "mode-confirmed"
        if "MACD" in mode:
            return "mode-macd"
        return "mode-flash"

    def label_css(s_type):
        mapping = {
            "PUMP": "pump-label",
            "DUMP": "dump-label",
            "BUY": "buy-label",
            "SELL": "sell-label",
        }
        return mapping.get(s_type, "buy-label")

    def row_css(s_type, mode):
        if "MACD" in mode:
            return "row-macd"
        is_up = s_type in ("PUMP", "BUY")
        if "FLASH" in mode:
            return "row-flash-pump" if is_up else "row-flash-dump"
        return "row-conf-pump" if is_up else "row-conf-dump"

    def render_table(display_data, placeholder):
        with placeholder.container():
            with radar.lock:
                if display_data:
                    html = (
                        "<table><tr>"
                        "<th>Time</th><th>Symbol (4H ↑/↓)</th><th>Price</th>"
                        "<th>Momentum</th><th>15m Ref</th><th>Vol</th>"
                        "<th>Status</th><th>Type</th><th>MACD Pattern</th>"
                        "</tr>"
                    )
                    for row in display_data:
                        sym = row['Symbol']
                        tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
                        p_type = row['P/D']
                        mode = row['Mode']
                        r_cls = row_css(p_type, mode)
                        lbl = label_css(p_type)
                        mode_cls = get_mode_css_class(mode)
                        macd_val = row.get('MACD', '')
                        macd_html = f"<span class='macd-tag'>{macd_val}</span>" if macd_val else "—"
                        vol_display = f"{row['Vol'] / 1000:.0f}k" if row['Vol'] > 0 else "—"
                        ref_display = f"{row['Ref']:+.4f}" if row['Ref'] != 0 else "—"

                        html += (
                            f"<tr class='{r_cls}'>"
                            f"<td>{row['Time']}</td>"
                            f"<td><a href='{tv_url}' target='_blank' class='sym-link'>{sym}</a> "
                            f"<small class='green-arrow'>↑{row['SnapP']}</small> "
                            f"<small class='red-arrow'>↓{row['SnapD']}</small></td>"
                            f"<td>{row['Price']}</td>"
                            f"<td style='font-weight:bold;'>{row['Chg']:+.2f}%</td>"
                            f"<td>{ref_display}</td>"
                            f"<td>{vol_display}</td>"
                            f"<td><span class='{mode_cls}'>{mode}</span></td>"
                            f"<td><span class='{lbl}'>{p_type}</span></td>"
                            f"<td>{macd_html}</td>"
                            f"</tr>"
                        )
                    html += "</table>"
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.info("Sinyal araniyor... Market taraniyor 🔍")

    while True:
        with side_placeholder.container():
            with radar.lock:
                sorted_stats = sorted(
                    radar.stats_hourly.items(),
                    key=lambda x: x[1]['PUMP'] + x[1]['DUMP'],
                    reverse=True,
                )[:5]
                for sym, counts in sorted_stats:
                    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
                    st.markdown(f"""<div class="stat-card">
                        <a href="{tv_url}" target="_blank" class="sym-link">{sym}</a><br>
                        <small>
                            <span class="green-arrow">↑ {counts["PUMP"]}</span> |
                            <span class="red-arrow">↓ {counts["DUMP"]}</span>
                        </small>
                    </div>""", unsafe_allow_html=True)

        with radar.lock:
            signals = list(radar.signals)

        display_data = signals
        if search_query:
            display_data = [s for s in display_data if search_query in s['Symbol']]
        if mode_filter:
            display_data = [s for s in display_data if s['Mode'] in mode_filter]
        if pd_filter:
            display_data = [s for s in display_data if s['P/D'] in pd_filter]
        if macd_only:
            display_data = [s for s in display_data if s.get('MACD')]

        render_table(display_data, main_placeholder)

        time.sleep(1.5)

# ================================================================
# SAYFA 2: MACD RADAR
# ================================================================
elif page == "📊 MACD Radar":
    h1.caption("📊 15 dakikalık grafiklerde paralel MACD yükseliş tespiti")

    st.markdown("""
    <div style="background-color:#1a1030; border-left:4px solid #8e44ad; padding:12px 16px; border-radius:4px; margin-bottom:16px;">
        <b style="color:#c39bd3;">MACD Radar Nasıl Çalışır?</b><br>
        <span style="color:#d5dbdb; font-size:0.9rem;">
        15 dakikalık mum grafiğinde MACD çizgisi ve sinyal çizgisi <b>paralel biçimde yükseliyor</b> mu?<br>
        • Pozitif bölgede, bullish, yükselen ve histogram genişleyen mumlar sayılır.<br>
        • <b>Paralel(N)</b>: Son N mumda koşul sağlandı. 3-8 arası sinyal üretir.<br>
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns([1, 3])
    macd_search = col_m1.text_input("🔍 Symbol Ara", placeholder="BTC...", key="macd_search").upper()
    min_candles = col_m2.slider("Min Paralel Mum Sayısı", 1, 15, 1)

    st.divider()

    def _parse_macd_count(tag):
        try:
            return int(tag.split("(")[1].rstrip(")"))
        except:
            return 0

    macd_page_placeholder = st.empty()

    while True:
        with radar.lock:
            candidates = dict(radar.macd_candidates)

        filtered = {
            sym: info for sym, info in candidates.items()
            if (not macd_search or macd_search in sym)
            and _parse_macd_count(info.get("MACD Pattern", "")) >= min_candles
        }

        sorted_c = sorted(
            filtered.items(),
            key=lambda x: _parse_macd_count(x[1].get("MACD Pattern", "")),
            reverse=True,
        )

        with macd_page_placeholder.container():
            if sorted_c:
                html = (
                    "<table><tr>"
                    "<th>Symbol</th><th>Fiyat</th><th>MACD Pattern</th><th>Güncelleme</th>"
                    "</tr>"
                )
                for sym, info in sorted_c:
                    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
                    count = _parse_macd_count(info.get("MACD Pattern", ""))
                    strength_color = "#00ff88" if count >= 6 else "#f39c12" if count >= 3 else "#c39bd3"
                    html += (
                        f"<tr class='row-macd'>"
                        f"<td><a href='{tv_url}' target='_blank' class='sym-link'>{sym}</a></td>"
                        f"<td>{info['Fiyat']}</td>"
                        f"<td><span class='macd-tag' style='color:{strength_color}; font-size:0.95rem;'>"
                        f"{info['MACD Pattern']}</span></td>"
                        f"<td style='color:#666;'>{info.get('Güncelleme', 'N/A')}</td>"
                        f"</tr>"
                    )
                html += "</table>"
                st.markdown(html, unsafe_allow_html=True)
                st.caption(f"Toplam {len(sorted_c)} sembol | Tüm liste: {len(candidates)}")
            else:
                st.info("MACD taraniyor... Semboller analiz ediliyor 🔍")

        time.sleep(1.5)

# ================================================================        
# SAYFA 3: BIG MOVE HUNTER
# ================================================================


