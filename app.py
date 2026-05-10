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
import random
import socket

# ==================== CONFIGURATION ====================
MIN_VOL_3M = 15000
MIN_CHG_3M = 0.8
CONFIRM_CHG_15M = 1.0
FAST_STRIKE_CHG = 0.8
TRI_WINDOW = 181
MAX_DISPLAY_ROWS = 100
FETCH_INTERVAL = 10
PUMP_DUMP_THRESHOLD = 2.2

SIGNAL_DEDUP_SECONDS = 60

MACD_PATTERN_COOLDOWN = 180
_KLINE_SEMAPHORE = threading.Semaphore(2)

# ==================== YENİ: ORDER BOOK / OI / FUNDING AYARLARI ====================
OB_FETCH_INTERVAL = 15          # saniye — order book polling aralığı
OB_IMBALANCE_THRESHOLD = 3.0    # bid/ask oranı: 3x büyükse baskı var
OB_DEPTH_LIMIT = 20             # kaç seviye depth çekilsin
OB_MIN_BID_SIZE = 100_000       # USDT — küçük coinlerde gürültüyü filtrele

OI_FETCH_INTERVAL = 30          # saniye — OI polling aralığı
OI_RISE_THRESHOLD = 0.8         # % — OI artış eşiği (düşük tutuldu, erken yakalama)
OI_FLAT_PRICE_MAX_CHG = 0.3     # % — "fiyat yatay iken" filtresi
OI_DIVERGE_CHG = 0.5            # % — fiyat yukarı ama OI düşüyorsa sahte pump eşiği

FUNDING_FETCH_INTERVAL = 60     # saniye — funding rate polling aralığı
FUNDING_SQUEEZE_THRESHOLD = -0.05  # % — bu kadar negatif altı = short ağır, squeeze ortamı
FUNDING_EXTREME_THRESHOLD = -0.10  # % — aşırı negatif = yüksek öncelikli sinyal
FUNDING_DEDUP_SECONDS = 300     # funding sinyali ne sıklıkla tekrar basılsın
# ==================== / ====================

BINANCE_REST_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

# ==================== SESSION FACTORY ====================

def create_session():
    session = requests.Session()
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36',
    ]
    session.headers.update({
        'User-Agent': random.choice(user_agents),
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
    })
    retry_strategy = Retry(
        total=2,
        backoff_factor=1.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=5,
        pool_maxsize=10,
        pool_block=False,
    )
    session.mount("https://", adapter)
    return session


# ==================== MACD HESAPLAMALARI ====================

def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    close = np.array(close_prices, dtype=float)
    def ema(data, span):
        k = 2 / (span + 1)
        result = [data[0]]
        for price in data[1:]:
            result.append(price * k + result[-1] * (1 - k))
        return np.array(result)
    exp1 = ema(close, fast)
    exp2 = ema(close, slow)
    macd = exp1 - exp2
    sig = ema(macd, signal)
    return macd, sig


def detect_whale_trap(m, s, volume_ratio=1.0):
    if len(m) < 6:
        return None
    macd_range = max(m[-6:-1]) - min(m[-6:-1])
    avg_move = np.mean([abs(m[i] - m[i-1]) for i in range(-5, -1)])
    compression = macd_range < avg_move * 2.5
    conditions = {
        "zero_above": m[-1] > 0,
        "crossover": m[-1] > s[-1] and m[-2] <= s[-2],
        "compression": compression,
        "signal_turning": s[-1] > s[-2] > s[-3],
        "volume_confirm": volume_ratio > 1.8,
        "momentum_accel": (m[-1] - m[-2]) > (m[-2] - m[-3]) > 0,
    }
    score = sum(conditions.values())
    if score == 6: return "WHALE TRAP — EFSANE (6/6)"
    if score >= 4: return f"WHALE TRAP — Guclu ({score}/6)"
    if score >= 3: return f"WHALE TRAP — Zayif ({score}/6)"
    return None


def detect_final_breakout(m, s, close_prices):
    if len(m) < 6 or len(close_prices) < 7:
        return None
    h = m - s
    hist_expanding = abs(h[-1]) > abs(h[-2]) > abs(h[-3]) and h[-1] > 0
    slopes = [m[-i] - m[-i-1] for i in range(1, 4)]
    accel = slopes[0] > slopes[1] > slopes[2] > 0
    signal_flip = s[-1] > s[-2] > s[-3]
    prices = np.array(close_prices)
    higher_high = prices[-1] > max(prices[-6:-1])
    conditions = {
        "crossover": m[-1] > s[-1] and m[-2] <= s[-2],
        "hist_expanding": hist_expanding,
        "accel": accel,
        "signal_flip": signal_flip,
        "higher_high": higher_high,
        "strong_momentum": (m[-1] - m[-2]) > (s[-1] - s[-2]) * 2.0,
    }
    score = sum(conditions.values())
    if score == 6: return "FINAL BREAKOUT — EFSANE (6/6)"
    if score >= 4: return f"FINAL BREAKOUT — Guclu ({score}/6)"
    if score >= 3: return f"FINAL BREAKOUT — Erken ({score}/6)"
    return None


def detect_triple_cross(m, s):
    if len(m) < 10:
        return None
    crosses = []
    for i in range(len(m) - 1, 1, -1):
        if m[i] > s[i] and m[i-1] <= s[i-1]:
            crosses.append({"bar_index": i, "level": m[i], "below_zero": m[i] < 0})
    if len(crosses) < 3:
        return None
    c1, c2, c3 = crosses[0], crosses[1], crosses[2]
    if not (m[-1] > s[-1] and m[-2] <= s[-2]): return None
    if not c3["below_zero"]: return None
    if not (c1["level"] > c2["level"] > c3["level"]): return None
    if (c1["bar_index"] - c2["bar_index"]) < 3: return None
    if (c2["bar_index"] - c3["bar_index"]) < 3: return None
    diff1 = c2["level"] - c3["level"]
    diff2 = c1["level"] - c2["level"]
    accelerating = diff2 > diff1 > 0
    levels = f"[{c3['level']:.6f}->{c2['level']:.6f}->{c1['level']:.6f}]"
    if accelerating: return f"TRIPLE CROSS — EFSANE {levels}"
    return f"TRIPLE CROSS — Guclu {levels}"


def detect_dalga(m, s):
    if len(m) < 50:
        return None, {}
    h = m - s
    threshold = np.mean([abs(m[i] - s[i]) for i in range(-50, 0)]) * 0.20
    state = 0
    meta = {}
    step1_macds = []
    i = 1
    while i < len(m):
        idx = i - len(m)
        if state == 0:
            cond_below = m[idx] < 0 and s[idx] < 0
            cond_close = abs(m[idx] - s[idx]) < threshold
            cond_slope_m = (m[idx] - m[idx-1]) <= 0
            cond_slope_s = (s[idx] - s[idx-1]) <= 0
            if cond_below and cond_close and cond_slope_m and cond_slope_s:
                step1_macds.append(m[idx])
                if len(step1_macds) >= 3:
                    state = 1
                    meta["step1_index"] = i
                    meta["step1_low"] = min(step1_macds)
            else:
                step1_macds = []
        elif state == 1:
            if m[idx-1] <= 0 and m[idx] > 0:
                state = 2
                meta["step2_index"] = i
                meta["step2_peak"] = m[idx]
        elif state == 2:
            if m[idx-1] >= 0 and m[idx] < 0:
                if m[idx] > meta["step1_low"]:
                    state = 3
                    meta["step3_index"] = i
                    meta["step3_low"] = m[idx]
                else:
                    state = 0
                    step1_macds = []
                    meta = {}
        elif state == 3:
            if m[idx] < 0:
                if m[idx] < meta["step1_low"]:
                    state = 0
                    step1_macds = []
                    meta = {}
                else:
                    meta["step3_low"] = min(meta["step3_low"], m[idx])
            elif m[idx-1] <= 0 and m[idx] > 0:
                state = 4
                meta["step4_index"] = i
        elif state == 4:
            cond_above = m[idx] > 0
            cond_rising = m[idx] > m[idx-1]
            cond_hist = h[idx] > h[idx-1]
            cond_new_high = m[idx] > meta["step2_peak"]
            if cond_above and cond_rising and cond_hist and cond_new_high:
                state = 5
                meta["step5_index"] = i
                break
        i += 1
    labels = {
        5: "DALGA — AL SINYALI (5/5)",
        4: "DALGA — TAKIPTE (4/5)",
        3: "DALGA — Geri Cekildi (3/5)",
        2: "DALGA — Sifir Ustu (2/5)",
        1: "DALGA — Sikisma (1/5)",
    }
    if state >= 1:
        return labels[state], meta
    return None, {}


def run_signals(close_prices, volume_ratio=1.0):
    if len(close_prices) < 50:
        return {"WHALE_TRAP": None, "FINAL_BREAKOUT": None, "TRIPLE_CROSS": None, "DALGA": None}
    m, s = calculate_macd(close_prices)
    dalga, _ = detect_dalga(m, s)
    return {
        "WHALE_TRAP": detect_whale_trap(m, s, volume_ratio),
        "FINAL_BREAKOUT": detect_final_breakout(m, s, close_prices),
        "TRIPLE_CROSS": detect_triple_cross(m, s),
        "DALGA": dalga,
    }


def get_signal_label(direction: str, chg: float) -> str:
    if abs(chg) >= PUMP_DUMP_THRESHOLD:
        return "PUMP" if direction == "up" else "DUMP"
    return "BUY" if direction == "up" else "SELL"


# ==================== YENİ: ORDER BOOK BASINCI ANALİZİ ====================

def calc_order_book_imbalance(bids, asks):
    """
    bids / asks: [[fiyat_str, miktar_str], ...]
    Döner: (bid_total_usdt, ask_total_usdt, imbalance_ratio, direction)
    imbalance_ratio > 1 → bid baskısı (alım), < 1 → ask baskısı (satım)
    """
    bid_total = sum(float(p) * float(q) for p, q in bids)
    ask_total = sum(float(p) * float(q) for p, q in asks)
    if ask_total == 0:
        return bid_total, ask_total, 99.0, "up"
    if bid_total == 0:
        return bid_total, ask_total, 0.0, "down"
    ratio = bid_total / ask_total
    direction = "up" if ratio >= 1.0 else "down"
    return bid_total, ask_total, ratio, direction


# ==================== ANA SINIF ====================

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
        self.price_cache_15m = {}
        self.debug_log = deque(maxlen=100)
        self.session = create_session()
        self.session_created_at = time.time()

        self._url_index = 0
        self._url_failures = 0
        self._url_lock = threading.Lock()
        self._consecutive_errors = 0
        self._total_requests = 0
        self._rate_limit_hits = 0

        self.macd_pattern_sent = {}
        self.macd_pattern_candidates = {}
        self.macd_pattern_last_trigger = {}

        self._signal_last_time = {}

        self.dbg_flash_checked = 0
        self.dbg_flash_ok = 0
        self.dbg_flash_blocked_dedup = 0
        self.dbg_confirmed_checked = 0
        self.dbg_confirmed_ok = 0
        self.dbg_confirmed_blocked_vol = 0
        self.dbg_confirmed_blocked_chg = 0
        self.dbg_confirmed_blocked_15m = 0
        self.dbg_confirmed_blocked_dedup = 0

        self._circuit_open = False
        self._circuit_open_until = 0
        self._circuit_event = threading.Event()

        self._current_fetch_interval = FETCH_INTERVAL
        self._healthy_streak = 0

        self._worker_thread = None
        self._stop_event = threading.Event()

        self._macd_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="macd")

        # ==================== YENİ: OB / OI / FUNDING STATE ====================

        # Order Book son baskı tablosu: sembol → {ratio, direction, bid_usdt, ask_usdt, ts}
        self.ob_state = {}
        self._ob_last_fetch = {}          # sembol → son fetch zamanı
        self._ob_semaphore = threading.Semaphore(4)  # eş zamanlı OB isteği limiti

        # Open Interest önbelleği: sembol → {oi, price, ts}
        self.oi_state = {}
        self._oi_last_fetch = {}          # sembol → son fetch zamanı
        self._oi_semaphore = threading.Semaphore(4)

        # Funding Rate önbelleği: sembol → {rate, ts}
        self.funding_state = {}
        self._funding_last_fetch = 0      # tüm semboller toplu çekilir
        self._funding_signal_sent = {}    # sembol → son sinyal zamanı

        # UI'da göstermek için özet tablolar
        self.ob_pressure_log = deque(maxlen=50)    # son OB sinyal kayıtları
        self.oi_divergence_log = deque(maxlen=50)  # son OI uyarıları
        self.funding_log = deque(maxlen=50)        # son funding anomalileri

        # OB/OI/Funding worker thread
        self._aux_worker_thread = None
        # ==================== / ====================

    def log(self, msg):
        t = datetime.now().strftime("%H:%M:%S")
        full = f"[{t}] {msg}"
        print(full, flush=True)
        with self.lock:
            self.debug_log.appendleft(full)

    # ==================== THREAD WATCHDOG ====================

    def ensure_worker_running(self):
        restarted = False
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self.log(">>> WATCHDOG: Worker thread yok veya ölü — yeniden başlatılıyor")
            self._stop_event.clear()
            self._circuit_event.clear()
            self._refresh_session_if_needed(force=True)
            t = threading.Thread(
                target=self._worker_loop,
                name="MarketRadarWorker",
                daemon=True,
            )
            t.start()
            self._worker_thread = t
            self.log(f">>> WATCHDOG: Thread başlatıldı (id={t.ident})")
            restarted = True

        # YENİ: Yardımcı (OB/OI/Funding) thread watchdog
        if self._aux_worker_thread is None or not self._aux_worker_thread.is_alive():
            self.log(">>> WATCHDOG: Aux worker (OB/OI/Funding) başlatılıyor")
            t2 = threading.Thread(
                target=self._aux_worker_loop,
                name="AuxWorker",
                daemon=True,
            )
            t2.start()
            self._aux_worker_thread = t2
            self.log(f">>> WATCHDOG: Aux thread başlatıldı (id={t2.ident})")

        return restarted

    def stop_worker(self):
        self._stop_event.set()
        self._circuit_event.set()

    # ==================== URL YÖNETİMİ ====================

    def _get_current_url(self):
        with self._url_lock:
            return BINANCE_REST_URLS[self._url_index % len(BINANCE_REST_URLS)]

    def _rotate_url(self, reason="failure"):
        with self._url_lock:
            old_idx = self._url_index
            self._url_index = (self._url_index + 1) % len(BINANCE_REST_URLS)
            self._url_failures = 0
            new_url = BINANCE_REST_URLS[self._url_index]
            self.log(f"URL rotasyon [{reason}]: [{old_idx}] -> [{self._url_index}] {new_url}")

    def _mark_url_success(self):
        with self._url_lock:
            self._url_failures = 0
            self._consecutive_errors = 0
            self._healthy_streak += 1
            if self._healthy_streak > 10:
                self._current_fetch_interval = max(FETCH_INTERVAL, self._current_fetch_interval - 1)
                self._healthy_streak = 0

    def _mark_url_failure(self):
        with self._url_lock:
            self._url_failures += 1
            self._consecutive_errors += 1
            self._healthy_streak = 0
            self._current_fetch_interval = min(60, self._current_fetch_interval + 2)
            if self._url_failures >= 2:
                self._rotate_url("failure_threshold")

    def get_working_rest_url(self):
        for i, url in enumerate(BINANCE_REST_URLS):
            try:
                r = self.session.get(f"{url}/fapi/v1/ping", timeout=6)
                if r.status_code == 200:
                    with self._url_lock:
                        self._url_index = i
                    self.log(f"Başlangıç URL: {url}")
                    return url
                else:
                    self.log(f"{url} -> status {r.status_code}")
            except Exception as e:
                self.log(f"{url} -> HATA: {str(e)[:60]}")
            time.sleep(0.5)
        self.log("UYARI: Tüm pingler başarısız, ilk URL ile devam ediliyor")
        return BINANCE_REST_URLS[0]

    # ==================== CIRCUIT BREAKER ====================

    def _check_circuit(self):
        if self._circuit_open:
            remaining = self._circuit_open_until - time.time()
            if remaining > 0:
                return False
            self._circuit_open = False
            self._circuit_event.clear()
            self.log("Circuit breaker KAPANDI")
        return True

    def _open_circuit(self, duration=60):
        self._circuit_open = True
        self._circuit_open_until = time.time() + duration
        self._circuit_event.set()
        self.log(f"Circuit breaker AÇILDI — {duration}s bekleniyor")

    def _interruptible_sleep(self, seconds):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._stop_event.is_set():
                return
            remaining = deadline - time.time()
            self._circuit_event.wait(timeout=min(1.0, remaining))
            self._circuit_event.clear()

    # ==================== SESSION YÖNETİMİ ====================

    def _refresh_session_if_needed(self, force=False):
        now = time.time()
        age = now - self.session_created_at
        if force or age > 600:
            try:
                old_session = self.session
                self.session = create_session()
                self.session_created_at = now
                try:
                    old_session.close()
                except Exception:
                    pass
                self.log(f"Session yenilendi (age={age:.0f}s, force={force})")
                return True
            except Exception as e:
                self.log(f"Session yenileme hatası: {e}")
        return False

    # ==================== HTTP İSTEK ====================

    def _safe_request(self, url, timeout=8):
        if self._stop_event.is_set():
            return None
        if not self._check_circuit():
            return None

        self._total_requests += 1

        try:
            time.sleep(random.uniform(0.01, 0.06))
            response = self.session.get(url, timeout=timeout)

            if response.status_code == 200:
                self._mark_url_success()
                for header_name, val in response.headers.items():
                    if 'X-MBX-USED-WEIGHT' in header_name.upper():
                        try:
                            weight = int(val)
                            if weight > 1000:
                                self.log(f"Weight yüksek: {weight}/1200 — yavaşlatılıyor")
                                self._current_fetch_interval = min(60, self._current_fetch_interval + 3)
                        except ValueError:
                            pass
                        break
                return response

            if response.status_code == 418:
                retry_after = int(response.headers.get('Retry-After', 120))
                self.log(f"418 IP BAN! {retry_after}s — session yenileniyor")
                self._rate_limit_hits += 1
                self._refresh_session_if_needed(force=True)
                self._open_circuit(retry_after)
                return None

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 30))
                self.log(f"429 Rate limit — {retry_after}s bekleniyor")
                self._rate_limit_hits += 1
                self._interruptible_sleep(retry_after)
                return None

            if response.status_code in (500, 502, 503, 504, 520, 521, 522, 523, 524):
                self.log(f"HTTP {response.status_code} — URL rotasyonu")
                self._mark_url_failure()
                return None

            self.log(f"Beklenmedik HTTP {response.status_code}")
            return None

        except requests.exceptions.ConnectionError as e:
            err_str = str(e).lower()
            if "remote end closed" in err_str or "connection aborted" in err_str:
                self.log("Keep-alive hatası — session yenileniyor")
                self._refresh_session_if_needed(force=True)
            else:
                self.log(f"Bağlantı hatası: {str(e)[:80]}")
                self._mark_url_failure()
            return None

        except requests.exceptions.Timeout:
            self.log("Timeout — URL rotasyonu")
            self._mark_url_failure()
            return None

        except requests.exceptions.RequestException as e:
            self.log(f"İstek hatası: {str(e)[:80]}")
            self._mark_url_failure()
            return None

        except Exception as e:
            self.log(f"Beklenmedik hata: {str(e)[:80]}")
            self._mark_url_failure()
            return None

    # ==================== RESET LOJİĞİ ====================

    def check_resets(self):
        now = datetime.now()
        if now.hour != self.last_reset_hour:
            self.stats_hourly.clear()
            self.last_reset_hour = now.hour
        if (now.hour // 4) != self.last_reset_4h_block:
            self.stats_4h.clear()
            self.last_reset_4h_block = now.hour // 4

    # ==================== 15M CACHE TEMİZLİĞİ ====================

    def _clean_15m_cache(self):
        now = time.time()
        expired = [k for k, (t, _) in self.price_cache_15m.items() if now - t > 600]
        for k in expired:
            del self.price_cache_15m[k]
        if expired:
            self.log(f"15m cache temizlendi: {len(expired)} kayıt silindi")

    # ==================== TICKER İŞLEME ====================

    def process_ticker(self, data):
        now = time.time()
        with self.lock:
            self.check_resets()
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

        self.dbg_flash_checked += 1
        if abs(c1) >= FAST_STRIKE_CHG and vol_1m >= 50000:
            direction = "up" if c1 > 0 else "down"
            label = get_signal_label(direction, c1)
            self.add_signal(symbol, current[1], c1, 0, vol_1m, label, "FLASH", score=40)
            self.dbg_flash_ok += 1

        self.dbg_confirmed_checked += 1
        vol_check = max(abs(vol_3m), abs(vol_1m))
        if vol_check < MIN_VOL_3M:
            self.dbg_confirmed_blocked_vol += 1
        elif abs(c3) < MIN_CHG_3M:
            self.dbg_confirmed_blocked_chg += 1
        else:
            price_15m_ago = self.get_15m_price(symbol)
            if not price_15m_ago:
                self.dbg_confirmed_blocked_15m += 1
            else:
                c15 = ((current[1] - price_15m_ago) / price_15m_ago) * 100
                is_consistent = (c3 > 0 and c15 > 0) or (c3 < 0 and c15 < 0)
                if is_consistent and abs(c15) >= CONFIRM_CHG_15M:
                    direction = "up" if c3 > 0 else "down"
                    label = get_signal_label(direction, c3)
                    self.add_signal(symbol, current[1], c3, c15, vol_3m, label, "CONFIRMED", score=55)
                    self.dbg_confirmed_ok += 1
                else:
                    self.dbg_confirmed_blocked_chg += 1

    def get_15m_price(self, symbol):
        now = time.time()
        if symbol in self.price_cache_15m:
            cache_time, price = self.price_cache_15m[symbol]
            if now - cache_time < 300:
                return price
        try:
            url = f"{self._get_current_url()}/fapi/v1/klines?symbol={symbol}&interval=15m&limit=2"
            response = self._safe_request(url, timeout=5)
            if response and response.status_code == 200:
                price = float(response.json()[0][1])
                self.price_cache_15m[symbol] = (now, price)
                return price
        except Exception as e:
            self.log(f"15m price hata ({symbol}): {e}")
        return None

    def add_signal(self, symbol, price, chg_main, chg_ref, vol, s_type, mode, score=50, macd_pattern=None, extra_tag=""):
        t_str = datetime.now().strftime("%H:%M:%S")
        sym_clean = symbol.replace("USDT", "")
        is_up = s_type in ("PUMP", "BUY")
        stat_key = "PUMP" if is_up else "DUMP"
        with self.lock:
            dedup_key = f"{sym_clean}|{mode}"
            now_t = time.time()
            last_sig_t = self._signal_last_time.get(dedup_key, 0)
            if now_t - last_sig_t < SIGNAL_DEDUP_SECONDS:
                if mode == "FLASH":
                    self.dbg_flash_blocked_dedup += 1
                else:
                    self.dbg_confirmed_blocked_dedup += 1
                return
            self._signal_last_time[dedup_key] = now_t

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
                "MACD_Pattern": macd_pattern or "",
                "ExtraTag": extra_tag,          # YENİ: OB/OI/Funding etiketi
            })
            self.log(f"SİNYAL: {sym_clean} {s_type} {mode} {chg_main:+.2f}%" +
                     (f" | {macd_pattern}" if macd_pattern else "") +
                     (f" | {extra_tag}" if extra_tag else ""))
            if len(self.signals) > MAX_DISPLAY_ROWS:
                self.signals.pop()

            if mode in ("FLASH", "CONFIRMED"):
                last_t = self.macd_pattern_last_trigger.get(symbol, 0)
                if now_t - last_t >= MACD_PATTERN_COOLDOWN:
                    self.macd_pattern_last_trigger[symbol] = now_t
                    try:
                        self._macd_executor.submit(self._run_macd_pattern_analysis, symbol, price)
                    except RuntimeError:
                        self.log("MACD executor kapalı — görev atlandı")

    # ==================== MACD PATTERN ANALİZİ ====================

    def _fetch_klines_for_pattern(self, symbol, interval="15m", limit=200):
        with _KLINE_SEMAPHORE:
            url = f"{self._get_current_url()}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
            try:
                resp = self._safe_request(url, timeout=8)
                if not resp or resp.status_code != 200:
                    return None
                raw = resp.json()
                closes = [float(c[4]) for c in raw]
                volumes = [float(c[5]) for c in raw]
                return closes, volumes
            except Exception as e:
                self.log(f"Kline hatası ({symbol} {interval}): {e}")
                return None

    def _run_macd_pattern_analysis(self, symbol, price):
        if self._stop_event.is_set():
            return
        result = self._fetch_klines_for_pattern(symbol, "15m", 200)
        if result is None:
            return
        closes, volumes = result
        if len(closes) < 50:
            return
        volume_ratio = 1.0
        if len(volumes) >= 21:
            avg_vol = np.mean(volumes[-21:-1])
            if avg_vol > 0:
                volume_ratio = volumes[-1] / avg_vol
        signals = run_signals(closes, volume_ratio)
        sym_clean = symbol.replace("USDT", "")
        now = time.time()
        best_pattern = None
        best_score = 0
        for key, value in signals.items():
            if value is None:
                continue
            score = 0
            if "EFSANE" in value: score = 100
            elif "Guclu" in value: score = 70
            elif any(x in value for x in ["Erken", "Zayif", "TAKIPTE", "AL SINYALI"]): score = 50
            elif any(x in value for x in ["Sikisma", "Sifir Ustu", "Geri Cekildi"]): score = 30
            if score > best_score:
                best_score = score
                best_pattern = value
        with self.lock:
            if best_pattern:
                self.macd_pattern_candidates[sym_clean] = {
                    "Sembol": sym_clean,
                    "Fiyat": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                    "Pattern": best_pattern,
                    "Guncelleme": datetime.now().strftime("%H:%M:%S"),
                }
                if now - self.macd_pattern_sent.get(sym_clean, 0) < MACD_PATTERN_COOLDOWN:
                    for sig in self.signals[:20]:
                        if sig.get('Symbol') == sym_clean:
                            sig['MACD_Pattern'] = best_pattern
                            break
                else:
                    self.macd_pattern_sent[sym_clean] = now
                    updated = False
                    for sig in self.signals[:20]:
                        if sig.get('Symbol') == sym_clean:
                            sig['MACD_Pattern'] = best_pattern
                            updated = True
                            self.log(f"MACD PATTERN eklendi: {sym_clean} -> {best_pattern}")
                            break
                    if not updated:
                        self.add_signal(
                            symbol=symbol, price=price,
                            chg_main=0.0, chg_ref=0.0, vol=0,
                            s_type="BUY", mode="MACD PATTERN",
                            score=60, macd_pattern=best_pattern,
                        )
            else:
                self.macd_pattern_candidates.pop(sym_clean, None)

    # ================================================================
    # YENİ: ORDER BOOK BASINCI
    # ================================================================

    def _fetch_order_book(self, symbol):
        """
        /fapi/v1/depth endpoint'inden bid/ask verisini çeker.
        Döner: (bid_usdt, ask_usdt, ratio, direction) | None
        """
        with self._ob_semaphore:
            url = (
                f"{self._get_current_url()}/fapi/v1/depth"
                f"?symbol={symbol}&limit={OB_DEPTH_LIMIT}"
            )
            try:
                resp = self._safe_request(url, timeout=6)
                if not resp or resp.status_code != 200:
                    return None
                data = resp.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                if not bids or not asks:
                    return None
                bid_usdt, ask_usdt, ratio, direction = calc_order_book_imbalance(bids, asks)
                return bid_usdt, ask_usdt, ratio, direction
            except Exception as e:
                self.log(f"OB fetch hatası ({symbol}): {str(e)[:60]}")
                return None

    def _process_order_book(self, symbol, price):
        """
        Tek sembol için OB çek, analiz et, gerekirse sinyal üret.
        Aux worker'dan çağrılır.
        """
        now = time.time()
        last = self._ob_last_fetch.get(symbol, 0)
        if now - last < OB_FETCH_INTERVAL:
            return
        self._ob_last_fetch[symbol] = now

        result = self._fetch_order_book(symbol)
        if result is None:
            return
        bid_usdt, ask_usdt, ratio, direction = result

        sym_clean = symbol.replace("USDT", "")
        t_str = datetime.now().strftime("%H:%M:%S")

        with self.lock:
            self.ob_state[sym_clean] = {
                "ratio": ratio,
                "direction": direction,
                "bid_usdt": bid_usdt,
                "ask_usdt": ask_usdt,
                "ts": t_str,
            }

        # Sinyal koşulları
        # 1) Aşırı imbalance: bid/ask ≥ OB_IMBALANCE_THRESHOLD ve henüz fiyat hareket etmemiş
        #    (henüz hareket etmemiş = mevcut 1m chg küçük → bu sinyal ÖNCÜ)
        # 2) Ask tarafı baskısı da aynı mantıkla ters yönde

        if bid_usdt < OB_MIN_BID_SIZE and ask_usdt < OB_MIN_BID_SIZE:
            return  # Düşük likidite, gürültü

        dedup_key = f"{sym_clean}|OB_PRESSURE"
        now_t = time.time()
        with self.lock:
            last_sig = self._signal_last_time.get(dedup_key, 0)
            if now_t - last_sig < SIGNAL_DEDUP_SECONDS:
                return

        if ratio >= OB_IMBALANCE_THRESHOLD:
            # Güçlü bid baskısı
            tag = f"📗 OB BİD BASKISI {ratio:.1f}x ({bid_usdt/1000:.0f}k vs {ask_usdt/1000:.0f}k)"
            self._signal_last_time[dedup_key] = now_t
            with self.lock:
                self.ob_pressure_log.appendleft({
                    "Time": t_str, "Symbol": sym_clean, "Price": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                    "Tag": tag, "Ratio": f"{ratio:.2f}x", "Direction": "BID",
                })
            self.add_signal(
                symbol=symbol, price=price,
                chg_main=0.0, chg_ref=0.0, vol=int(bid_usdt),
                s_type="BUY", mode="OB PRESSURE",
                score=65, extra_tag=tag,
            )
            self.log(f"OB PRESSURE (BID): {sym_clean} ratio={ratio:.2f}x")

        elif ratio <= (1.0 / OB_IMBALANCE_THRESHOLD):
            # Güçlü ask baskısı
            tag = f"📕 OB ASK BASKISI {1/ratio:.1f}x ({ask_usdt/1000:.0f}k vs {bid_usdt/1000:.0f}k)"
            self._signal_last_time[dedup_key] = now_t
            with self.lock:
                self.ob_pressure_log.appendleft({
                    "Time": t_str, "Symbol": sym_clean, "Price": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                    "Tag": tag, "Ratio": f"{1/ratio:.2f}x", "Direction": "ASK",
                })
            self.add_signal(
                symbol=symbol, price=price,
                chg_main=0.0, chg_ref=0.0, vol=int(ask_usdt),
                s_type="SELL", mode="OB PRESSURE",
                score=65, extra_tag=tag,
            )
            self.log(f"OB PRESSURE (ASK): {sym_clean} ratio={1/ratio:.2f}x")

    # ================================================================
    # YENİ: OPEN INTEREST ANALİZİ
    # ================================================================

    def _fetch_open_interest(self, symbol):
        """
        /fapi/v1/openInterest endpoint'inden anlık OI çeker.
        Döner: float (USDT cinsinden OI) | None
        """
        with self._oi_semaphore:
            url = f"{self._get_current_url()}/fapi/v1/openInterest?symbol={symbol}"
            try:
                resp = self._safe_request(url, timeout=6)
                if not resp or resp.status_code != 200:
                    return None
                data = resp.json()
                return float(data.get("openInterest", 0))
            except Exception as e:
                self.log(f"OI fetch hatası ({symbol}): {str(e)[:60]}")
                return None

    def _process_open_interest(self, symbol, price):
        """
        OI değişimi analizi:
        - Fiyat yatay + OI artıyor → sessiz birikim (ÖNCÜ)
        - Fiyat yukarı + OI düşüyor → short kapanışı, sahte pump uyarısı
        - Fiyat + OI birlikte artıyor → güçlü trend teyidi
        """
        now = time.time()
        last = self._oi_last_fetch.get(symbol, 0)
        if now - last < OI_FETCH_INTERVAL:
            return
        self._oi_last_fetch[symbol] = now

        oi_current = self._fetch_open_interest(symbol)
        if oi_current is None or oi_current == 0:
            return

        sym_clean = symbol.replace("USDT", "")
        t_str = datetime.now().strftime("%H:%M:%S")

        with self.lock:
            prev = self.oi_state.get(sym_clean)

        if prev is None:
            with self.lock:
                self.oi_state[sym_clean] = {"oi": oi_current, "price": price, "ts": now}
            return

        prev_oi = prev["oi"]
        prev_price = prev["price"]
        if prev_oi == 0:
            return

        oi_chg_pct = ((oi_current - prev_oi) / prev_oi) * 100
        price_chg_pct = ((price - prev_price) / prev_price) * 100 if prev_price > 0 else 0

        with self.lock:
            self.oi_state[sym_clean] = {"oi": oi_current, "price": price, "ts": now}

        dedup_key_acc = f"{sym_clean}|OI_ACCUM"
        dedup_key_fake = f"{sym_clean}|OI_FAKE"
        dedup_key_trend = f"{sym_clean}|OI_TREND"
        now_t = time.time()

        # Senaryo 1: Sessiz birikim — fiyat yatay, OI artıyor
        if (abs(price_chg_pct) <= OI_FLAT_PRICE_MAX_CHG
                and oi_chg_pct >= OI_RISE_THRESHOLD):
            last_sig = self._signal_last_time.get(dedup_key_acc, 0)
            if now_t - last_sig >= SIGNAL_DEDUP_SECONDS:
                tag = f"🔵 OI BİRİKİM: OI +{oi_chg_pct:.2f}% | Fiyat düz ({price_chg_pct:+.2f}%)"
                self._signal_last_time[dedup_key_acc] = now_t
                with self.lock:
                    self.oi_divergence_log.appendleft({
                        "Time": t_str, "Symbol": sym_clean,
                        "Price": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                        "OI_Chg": f"+{oi_chg_pct:.2f}%",
                        "Price_Chg": f"{price_chg_pct:+.2f}%",
                        "Senaryo": "Sessiz Birikim",
                    })
                self.add_signal(
                    symbol=symbol, price=price,
                    chg_main=oi_chg_pct, chg_ref=price_chg_pct, vol=0,
                    s_type="BUY", mode="OI SIGNAL",
                    score=70, extra_tag=tag,
                )
                self.log(f"OI BİRİKİM: {sym_clean} OI={oi_chg_pct:+.2f}% fiyat={price_chg_pct:+.2f}%")

        # Senaryo 2: Sahte pump — fiyat yukarı, OI düşüyor (short kapanışı)
        elif (price_chg_pct >= OI_DIVERGE_CHG
              and oi_chg_pct <= -OI_RISE_THRESHOLD):
            last_sig = self._signal_last_time.get(dedup_key_fake, 0)
            if now_t - last_sig >= SIGNAL_DEDUP_SECONDS:
                tag = f"⚠️ OI SAHTE PUMP: Fiyat {price_chg_pct:+.2f}% | OI {oi_chg_pct:.2f}% (short kpn)"
                self._signal_last_time[dedup_key_fake] = now_t
                with self.lock:
                    self.oi_divergence_log.appendleft({
                        "Time": t_str, "Symbol": sym_clean,
                        "Price": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                        "OI_Chg": f"{oi_chg_pct:.2f}%",
                        "Price_Chg": f"{price_chg_pct:+.2f}%",
                        "Senaryo": "Sahte Pump (short kapanışı)",
                    })
                self.add_signal(
                    symbol=symbol, price=price,
                    chg_main=price_chg_pct, chg_ref=oi_chg_pct, vol=0,
                    s_type="SELL", mode="OI SIGNAL",
                    score=60, extra_tag=tag,
                )
                self.log(f"OI SAHTE PUMP: {sym_clean} fiyat={price_chg_pct:+.2f}% OI={oi_chg_pct:.2f}%")

        # Senaryo 3: Güçlü trend teyidi — fiyat + OI birlikte artıyor
        elif (price_chg_pct >= OI_DIVERGE_CHG
              and oi_chg_pct >= OI_RISE_THRESHOLD):
            last_sig = self._signal_last_time.get(dedup_key_trend, 0)
            if now_t - last_sig >= SIGNAL_DEDUP_SECONDS:
                tag = f"✅ OI TREND TEYİDİ: Fiyat {price_chg_pct:+.2f}% + OI +{oi_chg_pct:.2f}%"
                self._signal_last_time[dedup_key_trend] = now_t
                with self.lock:
                    self.oi_divergence_log.appendleft({
                        "Time": t_str, "Symbol": sym_clean,
                        "Price": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                        "OI_Chg": f"+{oi_chg_pct:.2f}%",
                        "Price_Chg": f"{price_chg_pct:+.2f}%",
                        "Senaryo": "Güçlü Trend Teyidi",
                    })
                self.add_signal(
                    symbol=symbol, price=price,
                    chg_main=price_chg_pct, chg_ref=oi_chg_pct, vol=0,
                    s_type="BUY", mode="OI SIGNAL",
                    score=75, extra_tag=tag,
                )
                self.log(f"OI TREND: {sym_clean} fiyat={price_chg_pct:+.2f}% OI={oi_chg_pct:+.2f}%")

    # ================================================================
    # YENİ: FUNDING RATE ANOMALİSİ
    # ================================================================

    def _fetch_all_funding_rates(self):
        """
        /fapi/v1/premiumIndex — tüm semboller için funding rate döner.
        Her sembol için lastFundingRate alanı kullanılır.
        Ağır endpoint değil (1 istek, tüm market).
        """
        url = f"{self._get_current_url()}/fapi/v1/premiumIndex"
        try:
            resp = self._safe_request(url, timeout=10)
            if not resp or resp.status_code != 200:
                return None
            data = resp.json()
            result = {}
            for item in data:
                sym = item.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                try:
                    rate = float(item.get("lastFundingRate", 0)) * 100  # % cinsinden
                    mark_price = float(item.get("markPrice", 0))
                    result[sym] = {"rate": rate, "price": mark_price}
                except (ValueError, TypeError):
                    continue
            return result
        except Exception as e:
            self.log(f"Funding fetch hatası: {str(e)[:80]}")
            return None

    def _process_funding_rates(self):
        """
        Tüm semboller için funding rate anomalisi tara.
        - Aşırı negatif funding: short'lar çok ağır → squeeze ortamı → BUY sinyali
        - Aşırı pozitif funding: long'lar çok ağır → long squeeze riski → SELL uyarısı (isteğe bağlı, şimdilik sadece loglanır)
        """
        now = time.time()
        if now - self._funding_last_fetch < FUNDING_FETCH_INTERVAL:
            return
        self._funding_last_fetch = now

        rates = self._fetch_all_funding_rates()
        if not rates:
            return

        t_str = datetime.now().strftime("%H:%M:%S")

        with self.lock:
            # Tüm state'i güncelle
            for sym, info in rates.items():
                sym_clean = sym.replace("USDT", "")
                self.funding_state[sym_clean] = {
                    "rate": info["rate"],
                    "price": info["price"],
                    "ts": t_str,
                }

        # Anormalileri tara
        for sym, info in rates.items():
            if self._stop_event.is_set():
                break
            rate = info["rate"]
            price = info["price"]
            sym_clean = sym.replace("USDT", "")

            # Sadece aşırı negatif funding → short squeeze potansiyeli
            if rate > FUNDING_SQUEEZE_THRESHOLD:  # negatif değil, pas
                continue

            now_t = time.time()
            last_sig = self._funding_signal_sent.get(sym_clean, 0)
            if now_t - last_sig < FUNDING_DEDUP_SECONDS:
                continue

            # Öncelik belirle
            is_extreme = rate <= FUNDING_EXTREME_THRESHOLD
            score = 80 if is_extreme else 65
            strength_label = "⚡ AŞIRI NEGATİF" if is_extreme else "🔻 Negatif"
            tag = f"💰 FUNDING {strength_label}: {rate:.4f}% → Short Squeeze Ortamı"

            self._funding_signal_sent[sym_clean] = now_t

            with self.lock:
                self.funding_log.appendleft({
                    "Time": t_str,
                    "Symbol": sym_clean,
                    "Price": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                    "Rate": f"{rate:.4f}%",
                    "Durum": "Squeeze Ortamı" if not is_extreme else "AŞIRI — Yüksek Öncelik",
                })

            self.add_signal(
                symbol=sym, price=price,
                chg_main=rate, chg_ref=0.0, vol=0,
                s_type="BUY", mode="FUNDING",
                score=score, extra_tag=tag,
            )
            self.log(f"FUNDING ANOMALİ: {sym_clean} rate={rate:.4f}% {'EXTREME' if is_extreme else ''}")

    # ================================================================
    # YENİ: YARDIMCI WORKER LOOP (OB / OI / FUNDING)
    # ================================================================

    def _aux_worker_loop(self):
        """
        Ana ticker loop'undan bağımsız, düşük frekanslı:
        - Order Book: aktif semboller üzerinde polling
        - Open Interest: aktif semboller üzerinde polling
        - Funding Rate: tüm market, toplu
        """
        self.log(">>> AUX WORKER LOOP BAŞLADI (OB/OI/Funding)")

        ob_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ob_oi")
        cycle = 0

        while not self._stop_event.is_set():
            try:
                cycle += 1

                # ---- Aktif sembol listesi (history'den al, son 60s veri olanlar) ----
                now = time.time()
                with self.lock:
                    active_symbols = [
                        sym for sym, hist in self.history.items()
                        if hist and (now - hist[-1][0]) < 60
                    ]

                if not active_symbols:
                    self._interruptible_sleep(5)
                    continue

                # ---- Funding Rate (toplu, FUNDING_FETCH_INTERVAL'da bir) ----
                self._process_funding_rates()

                # ---- OB + OI (her sembol için, kendi interval'larına göre) ----
                # Sembol başına çok istek atmamak için rastgele shuffle + limit
                shuffled = list(active_symbols)
                random.shuffle(shuffled)
                selected = shuffled[:30]  # Her cycle en fazla 30 sembol işle

                futures = []
                for sym in selected:
                    if self._stop_event.is_set():
                        break
                    with self.lock:
                        hist = self.history.get(sym)
                        price = float(hist[-1][1]) if hist else 0.0

                    if price <= 0:
                        continue

                    # OB ve OI görevlerini paralel gönder
                    f1 = ob_executor.submit(self._process_order_book, sym, price)
                    f2 = ob_executor.submit(self._process_open_interest, sym, price)
                    futures.extend([f1, f2])

                # Tüm görevlerin bitmesini bekle (timeout ile)
                for f in futures:
                    try:
                        f.result(timeout=10)
                    except Exception as e:
                        self.log(f"Aux task hatası: {str(e)[:60]}")

                if cycle % 10 == 0:
                    with self.lock:
                        ob_count = len(self.ob_state)
                        oi_count = len(self.oi_state)
                        fund_count = len(self.funding_state)
                    self.log(
                        f"AUX cycle #{cycle} | OB:{ob_count} OI:{oi_count} "
                        f"Funding:{fund_count} | Active:{len(active_symbols)}"
                    )

                self._interruptible_sleep(5)  # Aux loop 5s'de bir döner

            except Exception as e:
                wait = 10
                self.log(f"AUX WORKER İSTİSNA: {str(e)[:120]} — {wait}s bekle")
                self._interruptible_sleep(wait)

        self.log(">>> AUX WORKER LOOP DURDU")

    # ==================== WORKER LOOP ====================

    def _worker_loop(self):
        self.log(">>> WORKER LOOP BAŞLADI (v4.1)")
        self.get_working_rest_url()

        fetch_count = 0
        cache_clean_counter = 0

        while not self._stop_event.is_set():
            try:
                if self._circuit_open:
                    remaining = max(1, int(self._circuit_open_until - time.time()))
                    self.log(f"Circuit açık — {remaining}s bekleniyor (kesilebilir)")
                    self._interruptible_sleep(remaining)
                    continue

                current_url = self._get_current_url()
                url = f"{current_url}/fapi/v1/ticker/24hr"
                response = self._safe_request(url, timeout=10)
                fetch_count += 1
                cache_clean_counter += 1

                if response and response.status_code == 200:
                    self.last_heartbeat = time.time()
                    raw = response.json()
                    formatted = [
                        {'s': x['symbol'], 'c': x['lastPrice'], 'q': x['quoteVolume']}
                        for x in raw
                    ]
                    self.process_ticker(formatted)

                    if fetch_count % 200 == 0:
                        self._refresh_session_if_needed(force=True)
                    if cache_clean_counter >= 30:
                        self._clean_15m_cache()
                        cache_clean_counter = 0
                    if fetch_count % 10 == 0:
                        self.log(
                            f"Fetch #{fetch_count} | {current_url.replace('https://', '')} | "
                            f"Pairs:{self.total_pairs} Signals:{len(self.signals)} "
                            f"MACD:{len(self.macd_pattern_candidates)} "
                            f"OB:{len(self.ob_state)} OI:{len(self.oi_state)} "
                            f"Funding:{len(self.funding_state)} "
                            f"Interval:{self._current_fetch_interval}s"
                        )

                    self._interruptible_sleep(self._current_fetch_interval)

                else:
                    wait = min(3 * (self._consecutive_errors + 1), 45)
                    self.log(f"Fetch başarısız — {wait}s bekleniyor")
                    self._interruptible_sleep(wait)

            except Exception as e:
                wait = min(5 * (self._consecutive_errors + 1), 60)
                self.log(f"WORKER İSTİSNA (döngü devam): {str(e)[:120]} — {wait}s bekle")
                self._interruptible_sleep(wait)

        self.log(">>> WORKER LOOP DURDU (stop_event)")


# ==================== STREAMLIT CACHE ====================

@st.cache_resource
def get_radar_instance():
    return MarketRadar()


# ==================== STREAMLIT UI ====================

st.set_page_config(layout="wide", page_title="Market Radar Pro v4.1")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .status-live { color: #00ff88; font-weight: bold; border: 1px solid #00ff88; padding: 2px 10px; border-radius: 15px; font-size: 0.8rem; }
    .status-offline { color: #ff4b4b; font-weight: bold; border: 1px solid #ff4b4b; padding: 2px 10px; border-radius: 15px; font-size: 0.8rem; }
    .status-warn { color: #f1c40f; font-weight: bold; border: 1px solid #f1c40f; padding: 2px 10px; border-radius: 15px; font-size: 0.8rem; }
    .pump-label  { background-color: #00ff88; color: black;  padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .dump-label  { background-color: #ff4b4b; color: white;  padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .buy-label   { background-color: #1a7f4b; color: #afffcf; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .sell-label  { background-color: #7f1a1a; color: #ffcfcf; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .mode-confirmed { background-color: #1abc9c; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .mode-flash { background-color: #e67e22; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .mode-macd  { background-color: #8e44ad; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .mode-ob    { background-color: #1a5276; color: #aed6f1; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .mode-oi    { background-color: #1e8449; color: #abebc6; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .mode-funding { background-color: #7d6608; color: #f9e79f; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .macd-pattern-tag { padding: 2px 8px; border-radius: 4px; font-size: 0.78rem; font-weight: bold; display: inline-block; white-space: nowrap; }
    .macd-whale-efsane { background-color: #1a3a4a; color: #00d4ff; border: 1px solid #00d4ff; }
    .macd-whale-guclu { background-color: #1a2a3a; color: #4db8ff; border: 1px solid #4db8ff; }
    .macd-whale-zayif { background-color: #1a2020; color: #88aabb; border: 1px solid #557788; }
    .macd-breakout-efsane { background-color: #1a3a1a; color: #00ff88; border: 1px solid #00ff88; }
    .macd-breakout-guclu { background-color: #1a2a1a; color: #4dff88; border: 1px solid #4dff88; }
    .macd-breakout-erken { background-color: #1a2515; color: #88cc66; border: 1px solid #88cc66; }
    .macd-triple-efsane { background-color: #2a1a3a; color: #ff6bff; border: 1px solid #ff6bff; }
    .macd-triple-guclu { background-color: #221530; color: #cc88dd; border: 1px solid #cc88dd; }
    .macd-dalga-al { background-color: #3a2a0a; color: #ffcc00; border: 1px solid #ffcc00; }
    .macd-dalga-takip { background-color: #2a2010; color: #ccaa44; border: 1px solid #ccaa44; }
    .macd-dalga-diger { background-color: #1a1a10; color: #999966; border: 1px solid #777755; }
    .stat-card { background-color: #1e2127; padding: 10px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #f1c40f; }
    .debug-box { background-color: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 10px; font-family: monospace; font-size: 0.75rem; color: #aaa; max-height: 200px; overflow-y: auto; }
    .watchdog-ok { color: #00ff88; font-size: 0.75rem; }
    .watchdog-warn { color: #f1c40f; font-size: 0.75rem; }
    .extra-tag-cell { font-size: 0.78rem; color: #bbb; max-width: 280px; white-space: normal; word-break: break-word; }
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
    .row-ob         { background-color: rgba(26, 82, 118, 0.18) !important; border-left: 3px solid #2980b9 !important; }
    .row-oi         { background-color: rgba(30, 132, 73, 0.15) !important; border-left: 3px solid #27ae60 !important; }
    .row-funding    { background-color: rgba(125, 102, 8, 0.20) !important; border-left: 3px solid #f1c40f !important; }
    </style>
""", unsafe_allow_html=True)

radar = get_radar_instance()

watchdog_restarted = radar.ensure_worker_running()

# ==================== SIDEBAR ====================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Sayfa Seç",
    ["Normal Sinyaller", "MACD Pattern Radar", "OB / OI / Funding", "Sistem Durumu"],
    index=0,
)
st.sidebar.markdown("---")

with st.sidebar:
    elapsed = time.time() - radar.last_heartbeat
    current_url = radar._get_current_url()
    url_short = current_url.replace("https://", "")

    if elapsed < 15:
        st.success(f"CANLI | {url_short}")
    elif elapsed < 30:
        st.warning(f"YAVAS | {url_short} | {elapsed:.0f}s")
    else:
        st.error(f"KESİNTİ | {url_short} | {elapsed:.0f}s")

    thread_alive = radar._worker_thread is not None and radar._worker_thread.is_alive()
    aux_alive = radar._aux_worker_thread is not None and radar._aux_worker_thread.is_alive()
    if thread_alive:
        st.markdown('<span class="watchdog-ok">● Main thread canlı</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="watchdog-warn">⟳ Main thread yeniden başlatılıyor...</span>', unsafe_allow_html=True)
    if aux_alive:
        st.markdown('<span class="watchdog-ok">● Aux thread (OB/OI/F) canlı</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="watchdog-warn">⟳ Aux thread yeniden başlatılıyor...</span>', unsafe_allow_html=True)

    if watchdog_restarted:
        st.warning("⚡ Watchdog thread'i yeniden başlattı")

    st.caption(f"URL hata: {radar._url_failures}/2 | Circuit: {'AÇIK' if radar._circuit_open else 'KAPALI'}")
    st.caption(f"Fetch interval: {radar._current_fetch_interval}s | Rate hits: {radar._rate_limit_hits}")
    st.caption("v4.1 OB/OI/FUNDING | Market Radar Pro")

# ==================== HEADER ====================
h1, h2, h3, h4, h5 = st.columns([2, 1, 1, 1, 1])
h1.title("Market Radar Pro")
h1.caption("v4.1 — Order Book Basıncı, Open Interest Divergence, Funding Rate Anomalisi")

elapsed = time.time() - radar.last_heartbeat
if elapsed < 15:
    status_html = '<span class="status-live">● SYSTEM LIVE</span>'
elif elapsed < 30:
    status_html = f'<span class="status-warn">● YAVAS ({elapsed:.0f}s)</span>'
else:
    status_html = f'<span class="status-offline">● RECONNECTING ({elapsed:.0f}s)</span>'

h2.markdown(f"<div style='margin-top:10px;'>{status_html}</div>", unsafe_allow_html=True)
h2.markdown(
    '<a href="https://x.com/SinyalEngineer" target="_blank" style="color:white; text-decoration:none;">X @SinyalEngineer</a>',
    unsafe_allow_html=True,
)
h3.metric("Pairs", radar.total_pairs)
h3.metric("Signals", len(radar.signals))
h4.metric("MACD Aday", len(radar.macd_pattern_candidates))
h4.metric("OB İzlenen", len(radar.ob_state))
h5.metric("OI İzlenen", len(radar.oi_state))
h5.metric("Funding", len(radar.funding_state))

st.divider()

with st.expander("Debug Log", expanded=False):
    with radar.lock:
        logs = list(radar.debug_log)
    if logs:
        log_html = "<div class='debug-box'>" + "<br>".join(logs) + "</div>"
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.info("Henüz log yok...")

st.divider()

# ==================== YARDIMCI FONKSİYONLAR ====================

def get_macd_pattern_css_class(pattern):
    if not pattern:
        return ""
    p = pattern.upper()
    if "WHALE TRAP" in p:
        if "EFSANE" in p: return "macd-whale-efsane"
        if "GUCLU" in p: return "macd-whale-guclu"
        return "macd-whale-zayif"
    if "FINAL BREAKOUT" in p:
        if "EFSANE" in p: return "macd-breakout-efsane"
        if "GUCLU" in p: return "macd-breakout-guclu"
        return "macd-breakout-erken"
    if "TRIPLE CROSS" in p:
        if "EFSANE" in p: return "macd-triple-efsane"
        return "macd-triple-guclu"
    if "DALGA" in p:
        if "AL SINYALI" in p: return "macd-dalga-al"
        if "TAKIPTE" in p: return "macd-dalga-takip"
        return "macd-dalga-diger"
    return ""


def get_pattern_score_sort(pattern):
    if not pattern:
        return 0
    p = pattern.upper()
    score = 0
    if "EFSANE" in p: score += 100
    elif "GUCLU" in p: score += 70
    elif "AL SINYALI" in p: score += 60
    elif "TAKIPTE" in p: score += 50
    elif "ERKEN" in p: score += 40
    elif "ZAYIF" in p: score += 30
    else: score += 20
    if "WHALE TRAP" in p: score += 4
    elif "FINAL BREAKOUT" in p: score += 3
    elif "TRIPLE CROSS" in p: score += 2
    elif "DALGA" in p: score += 1
    return score


def get_mode_css_class(mode):
    if "CONFIRMED" in mode: return "mode-confirmed"
    if "MACD" in mode: return "mode-macd"
    if "OB PRESSURE" in mode: return "mode-ob"
    if "OI SIGNAL" in mode: return "mode-oi"
    if "FUNDING" in mode: return "mode-funding"
    return "mode-flash"


def label_css(s_type):
    return {
        "PUMP": "pump-label", "DUMP": "dump-label",
        "BUY": "buy-label", "SELL": "sell-label"
    }.get(s_type, "buy-label")


def row_css(s_type, mode):
    if "MACD" in mode: return "row-macd"
    if "OB PRESSURE" in mode: return "row-ob"
    if "OI SIGNAL" in mode: return "row-oi"
    if "FUNDING" in mode: return "row-funding"
    is_up = s_type in ("PUMP", "BUY")
    if "FLASH" in mode: return "row-flash-pump" if is_up else "row-flash-dump"
    return "row-conf-pump" if is_up else "row-conf-dump"


# ================================================================
# SAYFA 1: NORMAL SİNYALLER
# ================================================================

if page == "Normal Sinyaller":
    col_filters = st.columns([1, 1, 1, 1, 1])
    mode_filter = col_filters[0].multiselect(
        "Sinyal Modu",
        ["FLASH", "CONFIRMED", "MACD PATTERN", "OB PRESSURE", "OI SIGNAL", "FUNDING"],
        default=["FLASH", "CONFIRMED", "MACD PATTERN", "OB PRESSURE", "OI SIGNAL", "FUNDING"],
        key="mode_filter"
    )
    pd_filter = col_filters[1].multiselect(
        "Yön",
        ["PUMP", "BUY", "DUMP", "SELL"],
        default=["PUMP", "BUY", "DUMP", "SELL"],
        key="pd_filter"
    )
    search_query = col_filters[2].text_input("Symbol Filtre", placeholder="BTC...", key="search").upper()
    macd_only = col_filters[3].checkbox("Sadece MACD Pattern'li", value=False)
    ob_oi_only = col_filters[4].checkbox("Sadece OB/OI/Funding", value=False)

    st.divider()
    col_side, col_main = st.columns([1, 5])

    with col_side:
        st.subheader("Top 5 Activity")
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
                    <span class="green-arrow">{counts["PUMP"]}</span> |
                    <span class="red-arrow">{counts["DUMP"]}</span>
                </small>
            </div>""", unsafe_allow_html=True)

        # YENİ: Funding Rate extremes mini widget
        st.markdown("---")
        st.subheader("En Negatif Funding")
        with radar.lock:
            funding_sorted = sorted(
                radar.funding_state.items(),
                key=lambda x: x[1]["rate"],
            )[:5]
        for sym_c, finfo in funding_sorted:
            rate = finfo["rate"]
            color = "#ff4b4b" if rate <= FUNDING_EXTREME_THRESHOLD else "#f1c40f"
            st.markdown(
                f"<span style='color:{color}; font-weight:bold;'>{sym_c}</span> "
                f"<span style='color:#888; font-size:0.85rem;'>{rate:.4f}%</span>",
                unsafe_allow_html=True,
            )

    with col_main:
        st.subheader("Intelligence Stream")
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
            display_data = [s for s in display_data if s.get('MACD_Pattern')]
        if ob_oi_only:
            display_data = [s for s in display_data if s['Mode'] in ("OB PRESSURE", "OI SIGNAL", "FUNDING")]

        if display_data:
            html = (
                "<table><tr>"
                "<th>Time</th><th>Symbol (4H ^/v)</th><th>Price</th>"
                "<th>Momentum</th><th>Ref</th><th>Vol</th>"
                "<th>Status</th><th>Type</th><th>MACD Pattern</th><th>Ek Sinyal</th>"
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
                macd_val = row.get('MACD_Pattern', '')
                extra_tag = row.get('ExtraTag', '')
                if macd_val:
                    pat_cls = get_macd_pattern_css_class(macd_val)
                    macd_html_str = f"<span class='macd-pattern-tag {pat_cls}'>{macd_val}</span>"
                else:
                    macd_html_str = "-"
                vol_display = f"{row['Vol'] / 1000:.0f}k" if row['Vol'] > 0 else "-"
                ref_display = f"{row['Ref']:+.4f}" if row['Ref'] != 0 else "-"
                extra_html = f"<span class='extra-tag-cell'>{extra_tag}</span>" if extra_tag else "-"
                html += (
                    f"<tr class='{r_cls}'>"
                    f"<td>{row['Time']}</td>"
                    f"<td><a href='{tv_url}' target='_blank' class='sym-link'>{sym}</a> "
                    f"<small class='green-arrow'>{row['SnapP']}</small> "
                    f"<small class='red-arrow'>{row['SnapD']}</small></td>"
                    f"<td>{row['Price']}</td>"
                    f"<td style='font-weight:bold;'>{row['Chg']:+.2f}%</td>"
                    f"<td>{ref_display}</td>"
                    f"<td>{vol_display}</td>"
                    f"<td><span class='{mode_cls}'>{mode}</span></td>"
                    f"<td><span class='{lbl}'>{p_type}</span></td>"
                    f"<td>{macd_html_str}</td>"
                    f"<td>{extra_html}</td>"
                    f"</tr>"
                )
            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Sinyal aranıyor... Market taranıyor")

    time.sleep(2)
    st.rerun()


# ================================================================
# SAYFA 2: MACD PATTERN RADAR
# ================================================================

elif page == "MACD Pattern Radar":
    st.markdown("""
    <div style="background-color:#1a1030; border-left:4px solid #8e44ad; padding:12px 16px; border-radius:4px; margin-bottom:16px;">
        <b style="color:#c39bd3;">MACD Pattern Radar Nasıl Çalışır?</b><br>
        <span style="color:#d5dbdb; font-size:0.9rem;">
        15 dakikalık mumda 4 farklı MACD pattern'i arar:<br><br>
        <b style="color:#00d4ff;">WHALE TRAP</b> — Sıfır üzerinde, sıkışma sonrası kesişim<br>
        <b style="color:#00ff88;">FINAL BREAKOUT</b> — Histogram genişliyor, momentum artıyor<br>
        <b style="color:#ff6bff;">TRIPLE CROSS</b> — Üç kez yukarı kesişim, her biri öncekinden yüksek<br>
        <b style="color:#ffcc00;">DALGA</b> — 5 adımlı sistem: Sıkışma → Sıfır → Geri → Tekrar → AL<br>
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
    pattern_search = col_m1.text_input("Symbol Ara", placeholder="BTC...", key="pattern_search").upper()
    pattern_filter = col_m2.multiselect(
        "Pattern Filtre",
        ["WHALE TRAP", "FINAL BREAKOUT", "TRIPLE CROSS", "DALGA"],
        default=["WHALE TRAP", "FINAL BREAKOUT", "TRIPLE CROSS", "DALGA"],
        key="pattern_filter"
    )
    min_strength = col_m3.selectbox(
        "Min Güçlendirme",
        ["Tümü", "Zayif/Erken+", "Guclu+", "EFSANE"],
        index=0,
        key="min_strength"
    )

    st.divider()

    with radar.lock:
        candidates = dict(radar.macd_pattern_candidates)

    filtered = {}
    for sym, info in candidates.items():
        pattern = info.get("Pattern", "")
        if pattern_search and pattern_search not in sym:
            continue
        if pattern_filter and not any(pf in pattern for pf in pattern_filter):
            continue
        if min_strength == "EFSANE" and "EFSANE" not in pattern:
            continue
        if min_strength == "Guclu+" and not any(x in pattern for x in ["EFSANE", "Guclu"]):
            continue
        if min_strength == "Zayif/Erken+" and not any(x in pattern for x in ["EFSANE", "Guclu", "Erken", "Zayif", "AL SINYALI", "TAKIPTE"]):
            continue
        filtered[sym] = info

    sorted_c = sorted(
        filtered.items(),
        key=lambda x: get_pattern_score_sort(x[1].get("Pattern", "")),
        reverse=True,
    )

    if sorted_c:
        html = "<table><tr><th>Symbol</th><th>Fiyat</th><th>MACD Pattern</th><th>Güncelleme</th></tr>"
        for sym, info in sorted_c:
            tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
            pattern = info.get("Pattern", "")
            pat_cls = get_macd_pattern_css_class(pattern)
            html += (
                f"<tr class='row-macd'>"
                f"<td><a href='{tv_url}' target='_blank' class='sym-link'>{sym}</a></td>"
                f"<td>{info['Fiyat']}</td>"
                f"<td><span class='macd-pattern-tag {pat_cls}' style='font-size:0.95rem;'>{pattern}</span></td>"
                f"<td style='color:#666;'>{info.get('Guncelleme', 'N/A')}</td>"
                f"</tr>"
            )
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)
        st.caption(f"Gösterilen: {len(sorted_c)} | Toplam aday: {len(candidates)}")
    else:
        st.info("MACD pattern taraması yapılıyor...")

    time.sleep(2)
    st.rerun()


# ================================================================
# YENİ SAYFA 3: OB / OI / FUNDING
# ================================================================

elif page == "OB / OI / Funding":
    st.subheader("Order Book Basıncı · Open Interest · Funding Rate")

    tab1, tab2, tab3 = st.tabs(["📗 Order Book Basıncı", "📊 OI Analizi", "💰 Funding Rate"])

    # ---- TAB 1: ORDER BOOK ----
    with tab1:
        st.markdown("""
        <div style="background-color:#0a1a2a; border-left:4px solid #2980b9; padding:10px 14px; border-radius:4px; margin-bottom:12px;">
        <b style="color:#aed6f1;">Order Book Basıncı Nedir?</b><br>
        <span style="color:#d5dbdb; font-size:0.85rem;">
        Balina futures'ta pozisyon açmadan önce order book'ta büyük bid/ask duvarları oluşturur.
        Bid tarafı ask'tan 3x+ büyükse ve fiyat henüz hareket etmemişse → öncü alım sinyali.<br>
        Eşik: <b>OB_IMBALANCE_THRESHOLD = {}</b> | Depth: <b>{} seviye</b>
        </span>
        </div>
        """.format(OB_IMBALANCE_THRESHOLD, OB_DEPTH_LIMIT), unsafe_allow_html=True)

        col_ob1, col_ob2 = st.columns([3, 2])

        with col_ob1:
            st.markdown("**Son OB Sinyalleri**")
            with radar.lock:
                ob_log = list(radar.ob_pressure_log)
            if ob_log:
                html = "<table><tr><th>Zaman</th><th>Symbol</th><th>Fiyat</th><th>Yön</th><th>Oran</th><th>Açıklama</th></tr>"
                for row in ob_log:
                    direction = row.get("Direction", "")
                    dir_color = "#00ff88" if direction == "BID" else "#ff4b4b"
                    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{row['Symbol']}USDT.P"
                    html += (
                        f"<tr class='row-ob'>"
                        f"<td>{row['Time']}</td>"
                        f"<td><a href='{tv_url}' target='_blank' class='sym-link'>{row['Symbol']}</a></td>"
                        f"<td>{row['Price']}</td>"
                        f"<td style='color:{dir_color}; font-weight:bold;'>{direction}</td>"
                        f"<td style='font-weight:bold;'>{row['Ratio']}</td>"
                        f"<td class='extra-tag-cell'>{row['Tag']}</td>"
                        f"</tr>"
                    )
                html += "</table>"
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("OB sinyali bekleniyor — semboller taranıyor...")

        with col_ob2:
            st.markdown("**Anlık OB Durumu (Top 20 — Bid Baskısı)**")
            with radar.lock:
                ob_sorted = sorted(
                    [(sym, info) for sym, info in radar.ob_state.items() if info["ratio"] >= 1.5],
                    key=lambda x: x[1]["ratio"],
                    reverse=True,
                )[:20]
            if ob_sorted:
                for sym_c, info in ob_sorted:
                    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym_c}USDT.P"
                    ratio = info["ratio"]
                    color = "#00ff88" if ratio >= OB_IMBALANCE_THRESHOLD else "#aaa"
                    bar_width = min(int(ratio * 15), 100)
                    st.markdown(
                        f"<a href='{tv_url}' target='_blank' class='sym-link'>{sym_c}</a> "
                        f"<span style='color:{color}; font-weight:bold;'>{ratio:.2f}x</span> "
                        f"<span style='display:inline-block; width:{bar_width}px; height:8px; "
                        f"background:{color}; border-radius:4px; vertical-align:middle;'></span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Henüz yeterli OB verisi yok...")

    # ---- TAB 2: OPEN INTEREST ----
    with tab2:
        st.markdown("""
        <div style="background-color:#0a2a1a; border-left:4px solid #27ae60; padding:10px 14px; border-radius:4px; margin-bottom:12px;">
        <b style="color:#abebc6;">Open Interest Analizi Nedir?</b><br>
        <span style="color:#d5dbdb; font-size:0.85rem;">
        <b>🔵 Sessiz Birikim:</b> Fiyat yatay ama OI artıyor → balina pozisyon açıyor<br>
        <b>⚠️ Sahte Pump:</b> Fiyat yukarı ama OI düşüyor → short kapanışı, gerçek alım yok<br>
        <b>✅ Trend Teyidi:</b> Fiyat + OI birlikte artıyor → güçlü trend<br>
        Eşikler: OI artış <b>≥{:.1f}%</b> | Fiyat düzlüğü <b>≤{:.1f}%</b>
        </span>
        </div>
        """.format(OI_RISE_THRESHOLD, OI_FLAT_PRICE_MAX_CHG), unsafe_allow_html=True)

        col_oi1, col_oi2 = st.columns([3, 2])

        with col_oi1:
            st.markdown("**Son OI Uyarıları**")
            with radar.lock:
                oi_log = list(radar.oi_divergence_log)
            if oi_log:
                html = "<table><tr><th>Zaman</th><th>Symbol</th><th>Fiyat</th><th>OI Δ</th><th>Fiyat Δ</th><th>Senaryo</th></tr>"
                for row in oi_log:
                    senaryo = row.get("Senaryo", "")
                    if "Birikim" in senaryo: s_color = "#3498db"
                    elif "Sahte" in senaryo: s_color = "#e74c3c"
                    else: s_color = "#2ecc71"
                    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{row['Symbol']}USDT.P"
                    html += (
                        f"<tr class='row-oi'>"
                        f"<td>{row['Time']}</td>"
                        f"<td><a href='{tv_url}' target='_blank' class='sym-link'>{row['Symbol']}</a></td>"
                        f"<td>{row['Price']}</td>"
                        f"<td style='font-weight:bold;'>{row['OI_Chg']}</td>"
                        f"<td>{row['Price_Chg']}</td>"
                        f"<td style='color:{s_color}; font-weight:bold;'>{senaryo}</td>"
                        f"</tr>"
                    )
                html += "</table>"
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("OI analizi bekleniyor — semboller taranıyor...")

        with col_oi2:
            st.markdown("**OI İzlenen Semboller**")
            with radar.lock:
                oi_count = len(radar.oi_state)
            st.metric("Toplam OI Takip", oi_count)
            st.caption(f"Fetch aralığı: {OI_FETCH_INTERVAL}s | Her cycle max 30 sembol")

    # ---- TAB 3: FUNDING RATE ----
    with tab3:
        st.markdown("""
        <div style="background-color:#2a2a0a; border-left:4px solid #f1c40f; padding:10px 14px; border-radius:4px; margin-bottom:12px;">
        <b style="color:#f9e79f;">Funding Rate Anomalisi Nedir?</b><br>
        <span style="color:#d5dbdb; font-size:0.85rem;">
        Funding rate aşırı negatife düştüğünde short'lar çok ağır basıyor demektir.
        Balina bu ortamda uzun pozisyon açar ve squeeze başlatır.<br>
        <b>Squeeze eşiği: ≤{:.2f}%</b> | <b>Aşırı eşik: ≤{:.2f}%</b> | Tekrar sinyali: {}s
        </span>
        </div>
        """.format(FUNDING_SQUEEZE_THRESHOLD, FUNDING_EXTREME_THRESHOLD, FUNDING_DEDUP_SECONDS), unsafe_allow_html=True)

        col_f1, col_f2 = st.columns([2, 2])

        with col_f1:
            st.markdown("**Funding Anomali Sinyalleri**")
            with radar.lock:
                fund_log = list(radar.funding_log)
            if fund_log:
                html = "<table><tr><th>Zaman</th><th>Symbol</th><th>Fiyat</th><th>Funding Rate</th><th>Durum</th></tr>"
                for row in fund_log:
                    durum = row.get("Durum", "")
                    d_color = "#ff4b4b" if "AŞIRI" in durum else "#f1c40f"
                    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{row['Symbol']}USDT.P"
                    html += (
                        f"<tr class='row-funding'>"
                        f"<td>{row['Time']}</td>"
                        f"<td><a href='{tv_url}' target='_blank' class='sym-link'>{row['Symbol']}</a></td>"
                        f"<td>{row['Price']}</td>"
                        f"<td style='color:#ff4b4b; font-weight:bold;'>{row['Rate']}</td>"
                        f"<td style='color:{d_color}; font-weight:bold;'>{durum}</td>"
                        f"</tr>"
                    )
                html += "</table>"
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("Funding anomalisi aranıyor...")

        with col_f2:
            st.markdown("**Tüm Market — En Negatif 20 Funding**")
            with radar.lock:
                funding_all = sorted(
                    radar.funding_state.items(),
                    key=lambda x: x[1]["rate"],
                )[:20]
            if funding_all:
                html = "<table><tr><th>Symbol</th><th>Funding Rate</th><th>Fiyat</th></tr>"
                for sym_c, finfo in funding_all:
                    rate = finfo["rate"]
                    price_f = finfo["price"]
                    if rate <= FUNDING_EXTREME_THRESHOLD:
                        r_color = "#ff4b4b"
                        row_style = "row-funding"
                    elif rate <= FUNDING_SQUEEZE_THRESHOLD:
                        r_color = "#f1c40f"
                        row_style = "row-funding"
                    else:
                        r_color = "#888"
                        row_style = ""
                    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym_c}USDT.P"
                    html += (
                        f"<tr class='{row_style}'>"
                        f"<td><a href='{tv_url}' target='_blank' class='sym-link'>{sym_c}</a></td>"
                        f"<td style='color:{r_color}; font-weight:bold;'>{rate:.4f}%</td>"
                        f"<td>{price_f:.4f if price_f < 1 else price_f:.2f}</td>"
                        f"</tr>"
                    )
                html += "</table>"
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("Funding verileri yükleniyor...")

    time.sleep(3)
    st.rerun()


# ================================================================
# SAYFA 4: SİSTEM DURUMU
# ================================================================

elif page == "Sistem Durumu":
    st.subheader("Sistem Sağlık Paneli")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    thread_alive = radar._worker_thread is not None and radar._worker_thread.is_alive()
    aux_alive = radar._aux_worker_thread is not None and radar._aux_worker_thread.is_alive()

    with col_s1:
        st.metric("Toplam İstek", radar._total_requests)
        st.metric("Rate Limit Çarpma", radar._rate_limit_hits)
        st.metric("Ardışık Hata", radar._consecutive_errors)

    with col_s2:
        st.metric("Aktif URL", radar._get_current_url().replace("https://", ""))
        st.metric("URL Hata Sayısı", f"{radar._url_failures}/2")
        st.metric("Circuit Breaker", "AÇIK" if radar._circuit_open else "KAPALI")

    with col_s3:
        st.metric("Fetch Interval", f"{radar._current_fetch_interval}s")
        st.metric("Session Yaşı", f"{time.time() - radar.session_created_at:.0f}s")
        st.metric("Main Worker", "✅ CANLI" if thread_alive else "⚠️ BAŞLATIYOR")

    with col_s4:
        st.metric("Aux Worker (OB/OI/F)", "✅ CANLI" if aux_alive else "⚠️ BAŞLATIYOR")
        st.metric("OB İzlenen", len(radar.ob_state))
        st.metric("Funding İzlenen", len(radar.funding_state))

    st.divider()
    st.subheader("Bağlantı İstikrarı")

    elapsed = time.time() - radar.last_heartbeat
    if elapsed < 15:
        st.success(f"✅ Bağlantı sağlıklı — son heartbeat {elapsed:.1f}s önce")
    elif elapsed < 60:
        st.warning(f"⚠️ Bağlantı yavaş — son heartbeat {elapsed:.1f}s önce")
    else:
        st.error(f"❌ Bağlantı kesik — {elapsed:.1f}s önce. Watchdog müdahale ediyor.")

    st.divider()
    st.subheader("OB / OI / Funding Konfigürasyonu")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    with cfg_col1:
        st.markdown("**Order Book**")
        st.caption(f"Fetch aralığı: {OB_FETCH_INTERVAL}s")
        st.caption(f"Imbalance eşiği: {OB_IMBALANCE_THRESHOLD}x")
        st.caption(f"Depth: {OB_DEPTH_LIMIT} seviye")
        st.caption(f"Min likidite: {OB_MIN_BID_SIZE:,} USDT")
    with cfg_col2:
        st.markdown("**Open Interest**")
        st.caption(f"Fetch aralığı: {OI_FETCH_INTERVAL}s")
        st.caption(f"OI artış eşiği: {OI_RISE_THRESHOLD}%")
        st.caption(f"Fiyat düzlük max: {OI_FLAT_PRICE_MAX_CHG}%")
        st.caption(f"Sahte pump fiyat eşiği: {OI_DIVERGE_CHG}%")
    with cfg_col3:
        st.markdown("**Funding Rate**")
        st.caption(f"Fetch aralığı: {FUNDING_FETCH_INTERVAL}s")
        st.caption(f"Squeeze eşiği: {FUNDING_SQUEEZE_THRESHOLD}%")
        st.caption(f"Aşırı eşik: {FUNDING_EXTREME_THRESHOLD}%")
        st.caption(f"Dedup: {FUNDING_DEDUP_SECONDS}s")

    st.info("""
    **v4.1 Yeni Özellikler (v4.0'a ek):**

    **🔵 1. Order Book Basıncı (`OB PRESSURE` modu)**
    `/fapi/v1/depth` endpoint'i — gerçek zamanlı bid/ask imbalance.
    Bid tarafı ask'tan 3x+ büyükse ve henüz fiyat hareket etmemişse öncü alım sinyali üretir.
    Ayrı `_aux_worker_loop` içinde, semboller arasında dağıtılmış polling.

    **📊 2. Open Interest Analizi (`OI SIGNAL` modu)**
    `/fapi/v1/openInterest` endpoint'i — 3 senaryo:
    • Sessiz Birikim: Fiyat yatay + OI artıyor
    • Sahte Pump: Fiyat yukarı + OI düşüyor (short kapanışı)
    • Trend Teyidi: Fiyat + OI birlikte artıyor

    **💰 3. Funding Rate Anomalisi (`FUNDING` modu)**
    `/fapi/v1/premiumIndex` — tek istekle tüm market.
    Aşırı negatif funding → short'lar çok ağır → squeeze ortamı → BUY sinyali.
    Extreme (-0.10%) ve normal (-0.05%) iki seviye ayrı score ile basılır.

    **Mimari değişiklik:** Yeni `_aux_worker_loop` ikinci bir thread olarak çalışır.
    Ana loop (ticker) ile tamamen bağımsız. Watchdog her ikisini de izler.
    """)

    time.sleep(3)
    st.rerun()
