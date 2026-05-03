import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import requests
from datetime import datetime
from collections import deque

# ==================== CONFIGURATION ====================
MIN_VOL_3M       = 40000
MIN_CHG_3M       = 1.0
CONFIRM_CHG_15M  = 1.0
FAST_STRIKE_CHG  = 0.5
TRI_WINDOW       = 180
MAX_DISPLAY_ROWS = 100
FETCH_INTERVAL   = 10

PUMP_DUMP_THRESHOLD = 1.5

# Paralel MACD mum sayısı aralığı (dahil)
PARALLEL_MIN = 3
PARALLEL_MAX = 9

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


# ==================== MACD PATTERN ENGINE ====================

def _count_parallel_candles(macd: np.ndarray, signal: np.ndarray) -> int:
    """
    En son mumdan geriye doğru kaç ardışık 'sağlıklı paralel' mum var sayar.

    Sağlıklı mum koşulları (hepsi aynı anda):
      1. MACD  > 0  (sıfır üstünde)
      2. Signal > 0  (sıfır üstünde)
      3. MACD  > Signal  (MACD signal'ın üstünde)
      4. MACD[i]  > MACD[i-1]   (MACD yükseliyor)
      5. Signal[i] > Signal[i-1] (Signal yükseliyor)
      6. Gap daralsa bile en fazla %5 daralma toleransı  (0.95)
      7. PARALELLIK: MACD eğimi / Signal eğimi 0.5 ile 2.0 arasında.
         (İki çizgi benzer hızda, birbirine paralel yükseliyor)

    İlk sağlıksız mumda durur — kesintisiz seri uzunluğunu döner.
    Dönen değer 0 ise hiç paralel mum yok.
    """
    count = 0
    max_lookback = min(15, len(macd) - 1)

    for i in range(1, max_lookback + 1):
        m_curr = macd[-i]
        m_prev = macd[-(i + 1)]
        s_curr = signal[-i]
        s_prev = signal[-(i + 1)]

        gap_curr = m_curr - s_curr
        gap_prev = m_prev - s_prev

        # 1-5: Temel koşullar
        basic_ok = (
            m_curr > 0 and s_curr > 0
            and m_curr > s_curr
            and m_curr > m_prev
            and s_curr > s_prev
        )

        # 6: Histogram daralma kontrolü (max %5 tolerans)
        hist_ok = gap_curr >= gap_prev * 0.95

        # 7: Paralellik — eğim oranı kontrolü
        m_slope = m_curr - m_prev
        s_slope = s_curr - s_prev
        
        parallel_ok = False
        if m_slope > 0 and s_slope > 0:
            ratio = m_slope / s_slope
            parallel_ok = 0.5 <= ratio <= 2.0

        if basic_ok and hist_ok and parallel_ok:
            count += 1
        else:
            break

    return count


def analyze_macd_patterns(df: pd.DataFrame):
    """
    15m kline DataFrame'inden MACD pattern analizi yapar.

    Önce paralel mum sayısını kontrol eder (3-9 arası → etiket üretir).
    Paralel bulunamazsa 7 klasik pattern'i sırayla dener.
    Hiç eşleşme yoksa ("", 0) döner — sinyal yine de gönderilir.

    Return: (label: str, bonus_score: int)
    """
    if df is None or len(df) < 30:
        return "", 0

    try:
        closes = df["close"].values

        exp1      = pd.Series(closes).ewm(span=12, adjust=False).mean()
        exp2      = pd.Series(closes).ewm(span=26, adjust=False).mean()
        macd      = (exp1 - exp2).values
        signal    = pd.Series(macd).ewm(span=9, adjust=False).mean().values
        histogram = macd - signal

        m = macd
        s = signal
        h = histogram

        m_slope = m[-1] - m[-2]
        s_slope = s[-1] - s[-2]
        h_slope = h[-1] - h[-2]

        # ── PARALEL MUM SAYACI (öncelik 1) ────────────────────────
        parallel_count = _count_parallel_candles(m, s)
        if PARALLEL_MIN <= parallel_count <= PARALLEL_MAX:
            bonus = parallel_count * 6
            return f"📊 PARALEl({parallel_count})", bonus

        # ── PATTERN 1: WHALE TRAP ──────────────────────────────────
        if (m[-1] > 0 and m[-1] > s[-1] and m[-2] <= s[-2]
                and m_slope > 0 and h[-1] > h[-2] > 0):
            return "🐳 WHALE TRAP", 90

        # ── PATTERN 2: ZERO LINE REJECTION ────────────────────────
        zero_touch = abs(m[-2]) < abs(m[-3]) * 0.35
        bounced_up = m[-1] > m[-2] and m_slope > 0
        above_zero = m[-1] > 0
        if zero_touch and bounced_up and above_zero:
            return "🎯 ZERO REJECT", 85

        # ── PATTERN 3: BULLISH DIVERGENCE ─────────────────────────
        price_lower_low = closes[-1] < closes[-5] and closes[-5] < closes[-10]
        macd_higher_low = m[-1] > m[-5]
        hist_turning_up = h[-1] > h[-2]
        if price_lower_low and macd_higher_low and hist_turning_up:
            return "🔄 BULL DIVERGE", 80

        # ── PATTERN 4: HISTOGRAM REVERSAL ─────────────────────────
        if h[-3] < 0 and h[-2] < 0 and h[-1] > 0 and m_slope > 0:
            return "📊 HIST REVERSAL", 75

        # ── PATTERN 5: HIDDEN DIVERGENCE ──────────────────────────
        price_higher_low   = closes[-1] > closes[-5]
        macd_lower_reading = m[-1] < m[-5]
        still_positive     = m[-1] > 0 and m_slope > 0
        if price_higher_low and macd_lower_reading and still_positive:
            return "🌊 HIDDEN DIV", 70

        # ── PATTERN 6: DIAGONAL POWER ─────────────────────────────
        if (m_slope > 0 and s_slope > 0
                and h[-1] > 0 and h_slope > 0
                and abs(m[-1] - s[-1]) > abs(m[-2] - s[-2])):
            return "📐 DIAGONAL PWR", 65

        # ── PATTERN 7: FINAL BREAKOUT ─────────────────────────────
        crossover  = m[-1] > s[-1] and m[-2] <= s[-2]
        fast_slope = m_slope > s_slope * 1.5 if s_slope != 0 else m_slope > 0
        hist_pos   = h[-1] > 0
        if crossover and fast_slope and hist_pos:
            return "🚀 FINAL BRKOUT", 60

    except Exception:
        pass

    return "", 0


# ==================== CORE CLASS ====================
class MarketRadar:
    def __init__(self):
        self.history       = {}
        self.signals       = []
        self.stats_hourly  = {}
        self.stats_4h      = {}
        self.lock          = threading.RLock()
        self.last_heartbeat = 0
        self.total_pairs   = 0
        self.last_reset_hour     = datetime.now().hour
        self.last_reset_4h_block = datetime.now().hour // 4

        self.headers = {
            'User-Agent':      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept':          'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Cache-Control':   'no-cache',
            'Pragma':          'no-cache',
            'Connection':      'keep-alive',
        }
        self.rest_base_url = BINANCE_REST_URLS[0]
        self.url_index     = 0

        self.price_cache_15m = {}
        self.kline_cache     = {}

        self.debug_log = []

    def log(self, msg):
        t    = datetime.now().strftime("%H:%M:%S")
        full = f"[{t}] {msg}"
        print(full, flush=True)
        with self.lock:
            self.debug_log.insert(0, full)
            if len(self.debug_log) > 50:
                self.debug_log.pop()

    def get_working_rest_url(self):
        for url in BINANCE_REST_URLS:
            try:
                r = requests.get(f"{url}/fapi/v1/ping", headers=self.headers, timeout=5)
                if r.status_code == 200:
                    self.rest_base_url = url
                    self.log(f"✅ Binance bağlantısı OK: {url}")
                    return url
                self.log(f"⚠️ {url} → status {r.status_code}")
            except Exception as e:
                self.log(f"❌ {url} → HATA: {e}")
        self.log("🔴 Hiçbir Binance URL'e bağlanılamadı!")
        return self.rest_base_url

    def rotate_url(self):
        self.url_index     = (self.url_index + 1) % len(BINANCE_REST_URLS)
        self.rest_base_url = BINANCE_REST_URLS[self.url_index]
        self.log(f"🔄 URL değiştirildi → {self.rest_base_url}")

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
            self.total_pairs    = len(data)
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
        past_1m = next((x for x in reversed(hist) if now - x[0] >= 60),         hist[0])
        past_3m = next((x for x in reversed(hist) if now - x[0] >= TRI_WINDOW),  hist[0])

        c1     = ((current[1] - past_1m[1]) / past_1m[1]) * 100
        c3     = ((current[1] - past_3m[1]) / past_3m[1]) * 100
        vol_3m = current[2] - past_3m[2]
        vol_1m = current[2] - past_1m[2]

        if abs(c1) >= FAST_STRIKE_CHG and vol_1m >= 50000:
            direction = "up" if c1 > 0 else "down"
            label     = get_signal_label(direction, c1)
            self.add_signal(symbol, current[1], c1, 0, vol_1m, label, "⚡ FLASH", score=40)
            return

        if vol_3m >= MIN_VOL_3M and abs(c3) >= MIN_CHG_3M:
            price_15m_ago = self.get_15m_price(symbol)
            if price_15m_ago:
                c15 = ((current[1] - price_15m_ago) / price_15m_ago) * 100
                is_consistent = (c3 > 0 and c15 > 0) or (c3 < 0 and c15 < 0)
                if is_consistent and abs(c15) >= CONFIRM_CHG_15M:
                    direction = "up" if c3 > 0 else "down"
                    label     = get_signal_label(direction, c3)
                    self.add_signal(symbol, current[1], c3, c15, vol_3m, label, "💎 CONFIRMED", score=55)

    def get_15m_price(self, symbol):
        now = time.time()
        if symbol in self.price_cache_15m:
            cache_time, price = self.price_cache_15m[symbol]
            if now - cache_time < 300:
                return price
        try:
            url      = f"{self.rest_base_url}/fapi/v1/klines?symbol={symbol}&interval=15m&limit=2"
            response = requests.get(url, headers=self.headers, timeout=3)
            if response.status_code == 200:
                price = float(response.json()[0][1])
                self.price_cache_15m[symbol] = (now, price)
                return price
        except Exception:
            pass
        return None

    def get_kline_df(self, symbol):
        now = time.time()
        if symbol in self.kline_cache:
            cache_ts, df = self.kline_cache[symbol]
            if now - cache_ts < 300:
                return df
        try:
            url  = f"{self.rest_base_url}/fapi/v1/klines?symbol={symbol}&interval=15m&limit=100"
            resp = requests.get(url, headers=self.headers, timeout=3)
            if resp.status_code == 200:
                raw = resp.json()
                df  = pd.DataFrame(raw, columns=[
                    "t", "open", "high", "low", "close", "v",
                    "c1", "q_volume", "c2", "c3", "c4", "c5"
                ])
                df[["open", "high", "low", "close", "q_volume"]] = \
                    df[["open", "high", "low", "close", "q_volume"]].astype(float)
                self.kline_cache[symbol] = (now, df)
                return df
        except Exception:
            pass
        return None

    def add_signal(self, symbol, price, chg_main, chg_ref, vol, s_type, mode, score=50):
        t_str     = datetime.now().strftime("%H:%M:%S")
        sym_clean = symbol.replace("USDT", "")
        is_up     = s_type in ("PUMP", "BUY")
        stat_key  = "PUMP" if is_up else "DUMP"

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
            snap_p = self.stats_4h[sym_clean]["PUMP"]
            snap_d = self.stats_4h[sym_clean]["DUMP"]

        df                     = self.get_kline_df(symbol)
        macd_label, macd_bonus = analyze_macd_patterns(df)
        final_score            = score + (macd_bonus // 5)

        with self.lock:
            self.signals.insert(0, {
                "Time":   t_str,
                "Symbol": sym_clean,
                "Price":  f"{price:.4f}" if price < 1 else f"{price:.2f}",
                "Chg":    chg_main,
                "Ref":    chg_ref,
                "Vol":    vol,
                "P/D":    s_type,
                "Mode":   mode,
                "MACD":   macd_label,
                "Score":  final_score,
                "SnapP":  snap_p,
                "SnapD":  snap_d,
            })
            macd_info = f" | MACD: {macd_label}" if macd_label else ""
            self.log(f"🚨 SİNYAL: {sym_clean} {s_type} {mode} {chg_main:+.2f}%{macd_info}")
            if len(self.signals) > MAX_DISPLAY_ROWS:
                self.signals.pop()


# ==================== WORKER ====================
@st.cache_resource
def get_radar_instance():
    return MarketRadar()


def binance_worker(radar_obj):
    radar_obj.log(">>> WORKER THREAD BAŞLADI")
    radar_obj.get_working_rest_url()

    fetch_count = 0
    retry_delay = FETCH_INTERVAL

    while True:
        try:
            url = f"{radar_obj.rest_base_url}/fapi/v1/ticker/24hr"
            r   = requests.get(url, headers=radar_obj.headers, timeout=8)
            fetch_count += 1

            if r.status_code in (418, 429):
                retry_delay = min(retry_delay * 2, 120)
                radar_obj.log(
                    f"🚫 Rate limit ({r.status_code}) "
                    f"→ {retry_delay}s bekleniyor, URL değiştiriliyor..."
                )
                radar_obj.rotate_url()
                time.sleep(retry_delay)
                continue

            if r.status_code == 200:
                retry_delay = FETCH_INTERVAL
                raw       = r.json()
                formatted = [
                    {'s': x['symbol'], 'c': x['lastPrice'], 'q': x['quoteVolume']}
                    for x in raw
                ]
                radar_obj.process_ticker(formatted)
                if fetch_count % 10 == 0:
                    radar_obj.log(
                        f"✅ Fetch #{fetch_count} | "
                        f"Pairs: {radar_obj.total_pairs} | "
                        f"Signals: {len(radar_obj.signals)} | "
                        f"History: {len(radar_obj.history)}"
                    )
            else:
                radar_obj.log(f"⚠️ HTTP {r.status_code} → URL değiştiriliyor")
                radar_obj.rotate_url()
                retry_delay = min(retry_delay + 5, 60)

        except requests.exceptions.Timeout:
            radar_obj.log("⏱️ Timeout → URL değiştiriliyor")
            radar_obj.rotate_url()
            retry_delay = min(retry_delay + 5, 60)
        except Exception as e:
            radar_obj.log(f"❌ WORKER HATA: {e}")
            retry_delay = min(retry_delay + 5, 60)

        time.sleep(retry_delay)


# ==================== STREAMLIT UI ====================
st.set_page_config(layout="wide", page_title="Market Radar")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .status-live    { color: #00ff88; font-weight: bold; border: 1px solid #00ff88; padding: 2px 10px; border-radius: 15px; font-size: 0.8rem; }
    .status-offline { color: #ff4b4b; font-weight: bold; border: 1px solid #ff4b4b; padding: 2px 10px; border-radius: 15px; font-size: 0.8rem; }
    .pump-label  { background-color: #00ff88; color: black;  padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .dump-label  { background-color: #ff4b4b; color: white;  padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .buy-label   { background-color: #1a7f4b; color: #afffcf; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .sell-label  { background-color: #7f1a1a; color: #ffcfcf; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .mode-confirmed { background-color: #1abc9c; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .mode-flash     { background-color: #e67e22; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .macd-parallel  { background-color: #2a2000; color: #ffd700; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; border: 1px solid #ffd700; }
    .macd-label     { background-color: #1e1a2e; color: #c8a8ff; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; border: 1px solid #6a4fcf; }
    .macd-empty     { color: #444; font-size: 0.9rem; }
    .stat-card  { background-color: #1e2127; padding: 10px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #f1c40f; }
    .debug-box  { background-color: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 10px; font-family: monospace; font-size: 0.75rem; color: #aaa; max-height: 200px; overflow-y: auto; }
    table { width: 100%; border-collapse: collapse; }
    th { font-size: 0.78rem; color: #777; text-transform: uppercase; letter-spacing: 0.05em; padding: 8px 14px; border-bottom: 2px solid #333; }
    td { white-space: nowrap; padding: 10px 14px; border-bottom: 1px solid #1a1a1a; }
    .sym-link       { color: #f1c40f; text-decoration: none; font-weight: bold; font-size: 1.05rem; }
    .sym-link:hover { color: #fff; }
    .green-arrow { color: #00ff88; font-weight: bold; }
    .red-arrow   { color: #ff4b4b; font-weight: bold; }
    .row-flash-pump { background-color: rgba(0, 255, 136, 0.22) !important; border-left: 4px solid #00ff88; }
    .row-flash-dump { background-color: rgba(255, 75,  75,  0.22) !important; border-left: 4px solid #ff4b4b; }
    .row-conf-pump  { background-color: rgba(0, 255, 136, 0.07) !important; }
    .row-conf-dump  { background-color: rgba(255, 75,  75,  0.07) !important; }
    </style>
""", unsafe_allow_html=True)

radar = get_radar_instance()
if "thread_started" not in st.session_state:
    t = threading.Thread(target=binance_worker, args=(radar,), daemon=True)
    t.start()
    st.session_state.thread_started = True
    radar.log(">>> UI: Thread başlatıldı")

h1, h2, h3 = st.columns([2, 1, 1])
h1.title("📡 Market Radar")
h1.caption(
    "⚡ Flash: Anlık | 💎 Confirmed: 3dk+15dk | "
    "🟡 PARALEl(n): n ardışık sağlıklı MACD mumu | 🟣 Diğer MACD pattern"
)

elapsed     = time.time() - radar.last_heartbeat
status_html = (
    '<span class="status-live">● SYSTEM LIVE</span>'
    if elapsed < 15
    else '<span class="status-offline">● RECONNECTING</span>'
)
h2.markdown(f"<div style='margin-top:10px;'>{status_html}</div>", unsafe_allow_html=True)
h2.markdown(
    '<a href="https://x.com/SinyalEngineer" target="_blank" '
    'style="color:white;text-decoration:none;">𝕏 @SinyalEngineer</a>',
    unsafe_allow_html=True,
)
h3.metric("Pairs Tracked", radar.total_pairs)
h3.metric("Signals", len(radar.signals))

st.divider()

with st.expander("🔧 Debug Log", expanded=False):
    with radar.lock:
        logs = list(radar.debug_log)
    if logs:
        st.markdown(
            "<div class='debug-box'>" + "<br>".join(logs) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Henüz log yok...")

st.divider()

col_f = st.columns([1, 1, 1, 1])
mode_filter  = col_f[0].multiselect(
    "Sinyal Modu", ["⚡ FLASH", "💎 CONFIRMED"],
    default=["⚡ FLASH", "💎 CONFIRMED"], key="mode_filter",
)
pd_filter    = col_f[1].multiselect(
    "Yön", ["PUMP", "BUY", "DUMP", "SELL"],
    default=["PUMP", "BUY", "DUMP", "SELL"], key="pd_filter",
)
macd_only    = col_f[2].checkbox("🔬 Sadece MACD eşleşenler", value=False, key="macd_only")
search_query = col_f[3].text_input("🔍 Symbol", placeholder="BTC...", key="search").upper()

st.divider()

col_side, col_main = st.columns([1, 4])

with col_side:
    st.subheader("🔥 Top 5 Activity")
    side_placeholder = st.empty()

with col_main:
    st.subheader("📡 Intelligence Stream")
    main_placeholder = st.empty()


def get_mode_css(mode):
    return "mode-confirmed" if "CONFIRMED" in mode else "mode-flash"

def label_css(s_type):
    return {
        "PUMP": "pump-label", "DUMP": "dump-label",
        "BUY":  "buy-label",  "SELL": "sell-label",
    }.get(s_type, "buy-label")

def row_css(s_type, mode):
    is_up = s_type in ("PUMP", "BUY")
    if "FLASH" in mode:
        return "row-flash-pump" if is_up else "row-flash-dump"
    return "row-conf-pump" if is_up else "row-conf-dump"

def macd_cell_html(macd_lbl: str) -> str:
    if not macd_lbl:
        return "<span class='macd-empty'>—</span>"
    if "PARALEl" in macd_lbl:
        return f"<span class='macd-parallel'>{macd_lbl}</span>"
    return f"<span class='macd-label'>{macd_lbl}</span>"


def render_table(display_data, placeholder):
    with placeholder.container():
        if display_data:
            html = (
                "<table><tr>"
                "<th>Saat</th>"
                "<th>Symbol (4H ↑/↓)</th>"
                "<th>Fiyat</th>"
                "<th>Momentum</th>"
                "<th>15m Ref</th>"
                "<th>Vol</th>"
                "<th>Mod</th>"
                "<th>Yön</th>"
                "<th>MACD Pattern</th>"
                "</tr>"
            )
            for row in display_data:
                sym    = row['Symbol']
                tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
                p_type = row['P/D']
                mode   = row['Mode']

                r_cls    = row_css(p_type, mode)
                lbl      = label_css(p_type)
                mode_cls = get_mode_css(mode)
                m_cell   = macd_cell_html(row.get('MACD', ''))

                html += (
                    f"<tr class='{r_cls}'>"
                    f"<td>{row['Time']}</td>"
                    f"<td>"
                    f"  <a href='{tv_url}' target='_blank' class='sym-link'>{sym}</a> "
                    f"  <small class='green-arrow'>↑{row['SnapP']}</small> "
                    f"  <small class='red-arrow'>↓{row['SnapD']}</small>"
                    f"</td>"
                    f"<td>{row['Price']}</td>"
                    f"<td style='font-weight:bold;'>{row['Chg']:+.2f}%</td>"
                    f"<td>{row['Ref']:+.4f}</td>"
                    f"<td>{row['Vol'] / 1000:.0f}k</td>"
                    f"<td><span class='{mode_cls}'>{mode}</span></td>"
                    f"<td><span class='{lbl}'>{p_type}</span></td>"
                    f"<td>{m_cell}</td>"
                    f"</tr>"
                )
            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Sinyal aranıyor... Market taranıyor 🔍")


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
            st.markdown(
                f'<div class="stat-card">'
                f'  <a href="{tv_url}" target="_blank" class="sym-link">{sym}</a><br>'
                f'  <small>'
                f'    <span class="green-arrow">↑ {counts["PUMP"]}</span> | '
                f'    <span class="red-arrow">↓ {counts["DUMP"]}</span>'
                f'  </small>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with radar.lock:
        display_data = list(radar.signals)

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
