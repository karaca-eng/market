import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import threading
import requests
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# ==================== CONFIGURATION ====================
MIN_VOL_3M = 40000
MIN_CHG_3M = 1.0
CONFIRM_CHG_15M = 1.0
FAST_STRIKE_CHG = 0.5
TRI_WINDOW = 180
MAX_DISPLAY_ROWS = 100
FETCH_INTERVAL = 3

PUMP_DUMP_THRESHOLD = 1.5

# MACD Paralel Yükselme Ayarları
MACD_MIN_CANDLES = 3   # Sinyal için minimum ardışık mum
MACD_MAX_CANDLES = 9   # Sinyal için maksimum ardışık mum (üstü zaten güçlü trend, giriş geç)
MACD_COOLDOWN = 180    # Aynı sembol için minimum sinyal aralığı (sn)
MACD_EXECUTOR = ThreadPoolExecutor(max_workers=15)

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
        self.debug_log = []

        # --- MACD Paralel: ek state ---
        self.macd_sent = {}        # symbol -> son sinyal zamanı (cooldown)
        self.macd_sent_keys = {}   # "SYMBOL_count" -> timestamp (aynı pattern tekrar gönderilmesin)
        self.macd_candidates = {}  # Radar tablosu için (symbol -> dict)
        # 1 dk'dan kısa hareketlerde analiz tetiklemeyi engellemek için
        self.macd_last_trigger = {}  # symbol -> son trigger zamanı

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
                # MACD trigger: her ticker'da değil, yeterli hareket varsa
                self._maybe_trigger_macd(symbol, price, now)

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
                "MACD": macd_tag or "",   # YENİ: MACD pattern etiketi
            })
            self.log(f"🚨 SİNYAL: {sym_clean} {s_type} {mode} {chg_main:+.2f}%" +
                     (f" | {macd_tag}" if macd_tag else ""))
            if len(self.signals) > MAX_DISPLAY_ROWS:
                self.signals.pop()

    # ================================================================
    # MACD PARALEL YÜKSELİŞ — V14'ten uyarlandı
    # ================================================================

    def _maybe_trigger_macd(self, symbol, price, now):
        """
        process_ticker içinden çağrılır.
        Fiyat 1 dakika içinde yeterince hareket ettiyse MACD analizini
        thread pool'a gönderir. Aşırı tetiklenmeyi önlemek için
        sembol başına 20 sn cooldown uygulanır.
        """
        hist = list(self.history.get(symbol, []))
        if len(hist) < 6:
            return

        # 20 sn cooldown — her ticker'da analiz açma
        last_t = self.macd_last_trigger.get(symbol, 0)
        if now - last_t < 20:
            return

        past_1m = next((x for x in reversed(hist) if now - x[0] >= 60), hist[0])
        p_chg_1m = abs(((price - past_1m[1]) / past_1m[1]) * 100)

        if p_chg_1m >= 0.25:
            self.macd_last_trigger[symbol] = now
            MACD_EXECUTOR.submit(self._run_macd_analysis, symbol, price)

    def _fetch_klines_macd(self, symbol):
        """15m mumlarını çeker, sadece close sütunu gerekli."""
        url = f"{self.rest_base_url}/fapi/v1/klines?symbol={symbol}&interval=15m&limit=60"
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code != 200:
                return None
            raw = resp.json()
            closes = [float(c[4]) for c in raw]
            return pd.Series(closes, name='close')
        except:
            return None

    def _analyze_macd_window(self, closes: pd.Series) -> int:
        """
        Ardışık kaç mumda MACD 'sağlıklı paralel yükseliş' içinde?
        Koşullar (V14 ile aynı):
          1. macd > 0 ve sinyal > 0
          2. macd > sinyal
          3. Her ikisi de bir önceki muma göre yükseliyor
          4. Histogram genişlemesi: fark >= önceki fark * 0.97
        """
        exp1 = closes.ewm(span=12, adjust=False).mean()
        exp2 = closes.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        count = 0
        for i in range(1, 16):
            if len(macd_line) < i + 1:
                break
            m_curr = macd_line.iloc[-i]
            m_prev = macd_line.iloc[-(i + 1)]
            s_curr = signal_line.iloc[-i]
            s_prev = signal_line.iloc[-(i + 1)]

            healthy = (
                m_curr > 0 and s_curr > 0 and
                m_curr > s_curr and
                m_curr > m_prev and s_curr > s_prev and
                (m_curr - s_curr) >= (m_prev - s_prev) * 0.97
            )
            if healthy:
                count += 1
            else:
                break
        return count

    def _run_macd_analysis(self, symbol, price):
        """
        Thread pool'da çalışır.
        MACD sayısına göre:
          - Her sayı >= 1: radar_candidates güncellenir
          - 3 <= sayı <= 9: mevcut sinyallere MACD etiketi eklenir
            veya standalone MACD sinyali eklenir
        """
        closes = self._fetch_klines_macd(symbol)
        if closes is None:
            return

        macd_count = self._analyze_macd_window(closes)
        sym_clean = symbol.replace("USDT", "")

        # Radar tablosunu güncelle (1+ mum)
        if macd_count >= 1:
            with self.lock:
                self.macd_candidates[sym_clean] = {
                    "Sembol": sym_clean,
                    "Fiyat": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                    "MACD Pattern": f"Paralel({macd_count})",
                    "Güncelleme": datetime.now().strftime("%H:%M:%S"),
                }
        else:
            with self.lock:
                self.macd_candidates.pop(sym_clean, None)
            return

        # Sinyal eşiği: 3-9 mum arası
        if not (MACD_MIN_CANDLES <= macd_count <= MACD_MAX_CANDLES):
            return

        now = time.time()
        alert_key = f"{sym_clean}_{macd_count}"

        with self.lock:
            # Aynı pattern tekrar gitmesin
            if alert_key in self.macd_sent_keys:
                return
            # Genel cooldown: aynı sembol için 180 sn
            if now - self.macd_sent.get(sym_clean, 0) < MACD_COOLDOWN:
                return
            self.macd_sent_keys[alert_key] = now
            self.macd_sent[sym_clean] = now

        macd_tag = f"Paralel({macd_count})"

        # Mevcut son sinyalde aynı sembol varsa sadece MACD etiketini güncelle
        updated = False
        with self.lock:
            for sig in self.signals[:15]:
                if sig.get('Symbol') == sym_clean and sig.get('MACD') == "":
                    sig['MACD'] = macd_tag
                    updated = True
                    self.log(f"🔷 MACD TAG eklendi: {sym_clean} → {macd_tag}")
                    break

        # Mevcut sinyalde yoksa standalone MACD sinyali ekle
        if not updated:
            self.add_signal(
                symbol=symbol,
                price=price,
                chg_main=0.0,
                chg_ref=0.0,
                vol=0,
                s_type="BUY",
                mode="📊 MACD",
                score=60,
                macd_tag=macd_tag,
            )
            self.log(f"🔷 MACD SİNYAL: {sym_clean} {macd_tag}")


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
                if fetch_count % 10 == 0:
                    radar_obj.log(
                        f"✅ Fetch #{fetch_count} | Pairs: {radar_obj.total_pairs} | "
                        f"Signals: {len(radar_obj.signals)} | "
                        f"MACD Radar: {len(radar_obj.macd_candidates)}"
                    )
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
    .mode-macd  { background-color: #8e44ad; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .macd-tag   { background-color: #2c1654; color: #c39bd3; padding: 2px 7px; border-radius: 4px; font-size: 0.78rem; font-weight: bold; border: 1px solid #8e44ad; }
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
    .macd-radar-card { background: #1a1030; border: 1px solid #8e44ad; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; }
    .macd-radar-sym { color: #c39bd3; font-weight: bold; font-size: 1rem; }
    .macd-radar-tag { color: #f0c3ff; font-size: 0.82rem; }
    .macd-radar-time { color: #666; font-size: 0.72rem; }
    </style>
""", unsafe_allow_html=True)

radar = get_radar_instance()
if "thread_started" not in st.session_state:
    t = threading.Thread(target=binance_worker, args=(radar,), daemon=True)
    t.start()
    st.session_state.thread_started = True
    radar.log(">>> UI: Thread başlatıldı")

# Header
h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
h1.title("📡 Market Radar")
h1.caption("⚡ Flash: Anlık hareket | 💎 Confirmed: 3dk+15dk | 📊 MACD: Paralel yükseliş (3-9 mum)")

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
h4.metric("MACD Radar", len(radar.macd_candidates))

st.divider()

# DEBUG PANEL
with st.expander("🔧 Debug Log (sorun giderme)", expanded=False):
    with radar.lock:
        logs = list(radar.debug_log)
    if logs:
        log_html = "<div class='debug-box'>" + "<br>".join(logs) + "</div>"
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.info("Henüz log yok...")

st.divider()

# Filtreler
col_filters = st.columns([1, 1, 1, 1])
mode_filter = col_filters[0].multiselect(
    "Sinyal Modu",
    ["⚡ FLASH", "💎 CONFIRMED", "📊 MACD"],
    default=["⚡ FLASH", "💎 CONFIRMED", "📊 MACD"],
    key="mode_filter"
)
pd_filter = col_filters[1].multiselect(
    "Yön",
    ["PUMP", "BUY", "DUMP", "SELL"],
    default=["PUMP", "BUY", "DUMP", "SELL"],
    key="pd_filter"
)
search_query = col_filters[2].text_input("🔍 Symbol Filter", placeholder="BTC...", key="search").upper()
macd_only = col_filters[3].checkbox("Sadece MACD etiketli sinyaller", value=False)

st.divider()

# Layout: aktivite | sinyal tablosu | macd radar
col_side, col_main, col_macd = st.columns([1, 4, 1])

with col_side:
    st.subheader("🔥 Top 5 Activity")
    side_placeholder = st.empty()

with col_main:
    st.subheader("📡 Intelligence Stream")
    main_placeholder = st.empty()

with col_macd:
    st.subheader("📊 MACD Radar")
    macd_placeholder = st.empty()


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

                    # Vol: MACD standalone sinyallerde vol=0, gösterme
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
                st.info("Sinyal aranıyor... Market taranıyor 🔍")


# ==================== UI LOOP ====================
while True:
    # --- Sol: Top 5 Aktivite ---
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

    # --- Orta: Sinyal tablosu ---
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

    # --- Sağ: MACD Radar ---
    with macd_placeholder.container():
        with radar.lock:
            candidates = dict(radar.macd_candidates)

        if candidates:
            # MACD mum sayısına göre büyükten küçüğe sırala
            def _macd_sort_key(item):
                tag = item[1].get("MACD Pattern", "Paralel(0)")
                try:
                    return int(tag.split("(")[1].rstrip(")"))
                except:
                    return 0

            sorted_c = sorted(candidates.items(), key=_macd_sort_key, reverse=True)
            for sym, info in sorted_c[:20]:
                tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
                st.markdown(f'''<div class="macd-radar-card">
                    <a href="{tv_url}" target="_blank" class="macd-radar-sym">{sym}</a><br>
                    <span class="macd-radar-tag">{info["MACD Pattern"]}</span>
                    &nbsp;
                    <span style="color:#aaa;font-size:0.8rem">{info["Fiyat"]}</span><br>
                    <span class="macd-radar-time">{info["Güncelleme"]}</span>
                </div>''', unsafe_allow_html=True)
        else:
            st.caption("MACD taranıyor...")

    time.sleep(1.5)
