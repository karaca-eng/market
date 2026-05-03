import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import threading
import requests
from datetime import datetime
from collections import deque

# ==================== CONFIGURATION ====================
MIN_VOL_3M = 40000
MIN_CHG_3M = 1.0
CONFIRM_CHG_15M = 1.0
FAST_STRIKE_CHG = 0.5
TRI_WINDOW = 180
MAX_DISPLAY_ROWS = 300
FETCH_INTERVAL = 3

PUMP_DUMP_THRESHOLD = 1.5

BINANCE_REST_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
]


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
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.rest_base_url = BINANCE_REST_URLS[0]
        self.price_cache_15m = {}
        self.debug_log = []  # debug mesajları buraya

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
                r = requests.get(f"{url}/fapi/v1/ping", timeout=3)
                if r.status_code == 200:
                    self.rest_base_url = url
                    self.log(f"✅ Binance bağlantısı OK: {url}")
                    return url
                else:
                    self.log(f"⚠️ {url} → status {r.status_code}")
            except Exception as e:
                self.log(f"❌ {url} → HATA: {e}")
        self.log("🔴 Hiçbir Binance URL'e bağlanılamadı!")
        return self.rest_base_url

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
            response = requests.get(url, headers=self.headers, timeout=2)
            if response.status_code == 200:
                price = float(response.json()[0][1])
                self.price_cache_15m[symbol] = (now, price)
                return price
        except:
            pass
        return None

    def add_signal(self, symbol, price, chg_main, chg_ref, vol, s_type, mode, score=50):
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
            })
            self.log(f"🚨 SİNYAL: {sym_clean} {s_type} {mode} {chg_main:+.2f}%")
            if len(self.signals) > MAX_DISPLAY_ROWS:
                self.signals.pop()


# ==================== WORKER ====================
@st.cache_resource
def get_radar_instance():
    return MarketRadar()


def binance_worker(radar_obj):
    radar_obj.log(">>> WORKER THREAD BAŞLADI")
    working_url = radar_obj.get_working_rest_url()
    radar_obj.log(f">>> Kullanılan URL: {working_url}")
    
    fetch_count = 0
    while True:
        try:
            url = f"{radar_obj.rest_base_url}/fapi/v1/ticker/24hr"
            r = requests.get(url, timeout=5)
            fetch_count += 1
            if r.status_code == 200:
                raw = r.json()
                formatted = [{'s': x['symbol'], 'c': x['lastPrice'], 'q': x['quoteVolume']} for x in raw]
                radar_obj.process_ticker(formatted)
                if fetch_count % 10 == 0:  # her 10 fetchte bir log bas
                    radar_obj.log(f"✅ Fetch #{fetch_count} | Pairs: {radar_obj.total_pairs} | Signals: {len(radar_obj.signals)} | History: {len(radar_obj.history)}")
            else:
                radar_obj.log(f"⚠️ HTTP {r.status_code}")
        except Exception as e:
            radar_obj.log(f"❌ WORKER HATA: {e}")
        time.sleep(FETCH_INTERVAL)


# ==================== STREAMLIT UI ====================
st.set_page_config(layout="wide", page_title="Market Radar")

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
    </style>
""", unsafe_allow_html=True)

radar = get_radar_instance()
if "thread_started" not in st.session_state:
    t = threading.Thread(target=binance_worker, args=(radar,), daemon=True)
    t.start()
    st.session_state.thread_started = True
    radar.log(">>> UI: Thread başlatıldı")

# Header
h1, h2, h3 = st.columns([2, 1, 1])
h1.title("📡 Market Radar")
h1.caption("⚡ Flash: Anlık hareket | 💎 Confirmed: 3dk + 15dk trend uyumu")

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
h3.metric("Signals", len(radar.signals))

st.divider()

# DEBUG PANEL
with st.expander("🔧 Debug Log (sorun giderme)", expanded=True):
    with radar.lock:
        logs = list(radar.debug_log)
    if logs:
        log_html = "<div class='debug-box'>" + "<br>".join(logs) + "</div>"
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.info("Henüz log yok...")

st.divider()

# Filtreler
col_filters = st.columns([1, 1, 1])
mode_filter = col_filters[0].multiselect(
    "Sinyal Modu",
    ["⚡ FLASH", "💎 CONFIRMED"],
    default=["⚡ FLASH", "💎 CONFIRMED"],
    key="mode_filter"
)
pd_filter = col_filters[1].multiselect(
    "Yön",
    ["PUMP", "BUY", "DUMP", "SELL"],
    default=["PUMP", "BUY", "DUMP", "SELL"],
    key="pd_filter"
)
search_query = col_filters[2].text_input("🔍 Symbol Filter", placeholder="BTC...", key="search").upper()

st.divider()

col_side, col_main = st.columns([1, 4])

with col_side:
    st.subheader("🔥 Top 5 Activity")
    side_placeholder = st.empty()

with col_main:
    st.subheader("📡 Intelligence Stream (Flash & Confirmed)")
    main_placeholder = st.empty()


def get_mode_css_class(mode):
    if "CONFIRMED" in mode:
        return "mode-confirmed"
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
                    "<th>Status</th><th>Type</th>"
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

                    html += (
                        f"<tr class='{r_cls}'>"
                        f"<td>{row['Time']}</td>"
                        f"<td><a href='{tv_url}' target='_blank' class='sym-link'>{sym}</a> "
                        f"<small class='green-arrow'>↑{row['SnapP']}</small> "
                        f"<small class='red-arrow'>↓{row['SnapD']}</small></td>"
                        f"<td>{row['Price']}</td>"
                        f"<td style='font-weight:bold;'>{row['Chg']:+.2f}%</td>"
                        f"<td>{row['Ref']:+.4f}</td>"
                        f"<td>{row['Vol'] / 1000:.0f}k</td>"
                        f"<td><span class='{mode_cls}'>{mode}</span></td>"
                        f"<td><span class='{lbl}'>{p_type}</span></td>"
                        f"</tr>"
                    )
                html += "</table>"
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("Sinyal aranıyor... Market taranıyor 🔍")


# UI Loop
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
                st.markdown(f'''<div class="stat-card">
                    <a href="{tv_url}" target="_blank" class="sym-link">{sym}</a><br>
                    <small>
                        <span class="green-arrow">↑ {counts["PUMP"]}</span> |
                        <span class="red-arrow">↓ {counts["DUMP"]}</span>
                    </small>
                </div>''', unsafe_allow_html=True)

    with radar.lock:
        signals = list(radar.signals)
        display_data = signals
        if search_query:
            display_data = [s for s in display_data if search_query in s['Symbol']]
        if mode_filter:
            display_data = [s for s in display_data if s['Mode'] in mode_filter]
        if pd_filter:
            display_data = [s for s in display_data if s['P/D'] in pd_filter]

    render_table(display_data, main_placeholder)
    time.sleep(1.5)
