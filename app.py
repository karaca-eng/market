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
MIN_VOL_3M = 40000
MIN_CHG_3M = 1.0
CONFIRM_CHG_15M = 1.3
FAST_STRIKE_CHG = 0.8
TRI_WINDOW = 180
MAX_DISPLAY_ROWS = 100
FETCH_INTERVAL = 10
PUMP_DUMP_THRESHOLD = 2.2

MACD_PATTERN_COOLDOWN = 180
_KLINE_SEMAPHORE = threading.Semaphore(2)

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
        respect_retry_after_header=False,  # Manuel yönetiyoruz
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

        # Circuit breaker
        self._circuit_open = False
        self._circuit_open_until = 0
        self._circuit_event = threading.Event()  # YENİ: sleep interrupt için

        # Adaptive fetch interval
        self._current_fetch_interval = FETCH_INTERVAL
        self._healthy_streak = 0

        # YENİ: Thread watchdog
        self._worker_thread = None
        self._stop_event = threading.Event()

        # YENİ: MACD executor (instance'a taşındı, yönetilebilir)
        self._macd_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="macd")

    def log(self, msg):
        t = datetime.now().strftime("%H:%M:%S")
        full = f"[{t}] {msg}"
        print(full, flush=True)
        with self.lock:
            self.debug_log.appendleft(full)

    # ==================== THREAD WATCHDOG ====================

    def ensure_worker_running(self):
        """
        Her UI refresh'te çağrılır.
        Thread ölmüşse temiz şekilde yeniden başlatır.
        """
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self.log(">>> WATCHDOG: Worker thread yok veya ölü — yeniden başlatılıyor")
            self._stop_event.clear()
            self._circuit_event.clear()
            # Yeni session aç (eski bağlantılar kirli olabilir)
            self._refresh_session_if_needed(force=True)
            t = threading.Thread(
                target=self._worker_loop,
                name="MarketRadarWorker",
                daemon=True,
            )
            t.start()
            self._worker_thread = t
            self.log(f">>> WATCHDOG: Thread başlatıldı (id={t.ident})")
            return True
        return False

    def stop_worker(self):
        """Graceful shutdown"""
        self._stop_event.set()
        self._circuit_event.set()  # Uyuyan thread'i uyandır

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
        self._circuit_event.set()  # _interruptible_sleep'i uyandır
        self.log(f"Circuit breaker AÇILDI — {duration}s bekleniyor")

    def _interruptible_sleep(self, seconds):
        """
        time.sleep() yerine kullan — stop_event veya circuit_event ile
        kesilebilir. Thread donmaz.
        """
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._stop_event.is_set():
                return
            remaining = deadline - time.time()
            # max 1 saniyelik dilimler halinde uyu
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
                # Weight header kontrolü
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
        """Eski cache kayıtlarını sil — bellek sızıntısını önler"""
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

        if abs(c1) >= FAST_STRIKE_CHG and vol_1m >= 50000:
            direction = "up" if c1 > 0 else "down"
            label = get_signal_label(direction, c1)
            self.add_signal(symbol, current[1], c1, 0, vol_1m, label, "FLASH", score=40)
            return

        if vol_3m >= MIN_VOL_3M and abs(c3) >= MIN_CHG_3M:
            price_15m_ago = self.get_15m_price(symbol)
            if price_15m_ago:
                c15 = ((current[1] - price_15m_ago) / price_15m_ago) * 100
                is_consistent = (c3 > 0 and c15 > 0) or (c3 < 0 and c15 < 0)
                if is_consistent and abs(c15) >= CONFIRM_CHG_15M:
                    direction = "up" if c3 > 0 else "down"
                    label = get_signal_label(direction, c3)
                    self.add_signal(symbol, current[1], c3, c15, vol_3m, label, "CONFIRMED", score=55)

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

    def add_signal(self, symbol, price, chg_main, chg_ref, vol, s_type, mode, score=50, macd_pattern=None):
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
                "MACD_Pattern": macd_pattern or "",
            })
            self.log(f"SİNYAL: {sym_clean} {s_type} {mode} {chg_main:+.2f}%" +
                     (f" | {macd_pattern}" if macd_pattern else ""))
            if len(self.signals) > MAX_DISPLAY_ROWS:
                self.signals.pop()

            if mode in ("FLASH", "CONFIRMED"):
                now_t = time.time()
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

    # ==================== WORKER LOOP ====================

    def _worker_loop(self):
        """
        Tek sorumluluk: Binance'ten veri çek, işle, uy.
        Exception ne olursa olsun döngü kırılmaz.
        Thread watchdog bu methodu çağırır.
        """
        self.log(">>> WORKER LOOP BAŞLADI (v4.0)")
        self.get_working_rest_url()

        fetch_count = 0
        cache_clean_counter = 0

        while not self._stop_event.is_set():
            try:
                # Circuit breaker aktifse interrupt edilebilir şekilde bekle
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

                    # Periyodik görevler
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
                            f"Interval:{self._current_fetch_interval}s"
                        )

                    self._interruptible_sleep(self._current_fetch_interval)

                else:
                    # Başarısız fetch — kısa exponential backoff
                    wait = min(3 * (self._consecutive_errors + 1), 45)
                    self.log(f"Fetch başarısız — {wait}s bekleniyor")
                    self._interruptible_sleep(wait)

            except Exception as e:
                # Hiçbir şey thread'i öldüremez
                wait = min(5 * (self._consecutive_errors + 1), 60)
                self.log(f"WORKER İSTİSNA (döngü devam): {str(e)[:120]} — {wait}s bekle")
                self._interruptible_sleep(wait)

        self.log(">>> WORKER LOOP DURDU (stop_event)")


# ==================== STREAMLIT CACHE ====================

@st.cache_resource
def get_radar_instance():
    return MarketRadar()


# ==================== STREAMLIT UI ====================

st.set_page_config(layout="wide", page_title="Market Radar Pro v4.0")

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
    </style>
""", unsafe_allow_html=True)

radar = get_radar_instance()

# ==================== WATCHDOG: Her refresh'te çalışır ====================
# Eski kod: if "thread_started" not in st.session_state: ...
# Yeni kod: thread canlı mı diye her seferinde kontrol et
watchdog_restarted = radar.ensure_worker_running()

# ==================== SIDEBAR ====================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Sayfa Seç",
    ["Normal Sinyaller", "MACD Pattern Radar", "Sistem Durumu"],
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
    if thread_alive:
        st.markdown('<span class="watchdog-ok">● Thread canlı</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="watchdog-warn">⟳ Thread yeniden başlatılıyor...</span>', unsafe_allow_html=True)

    if watchdog_restarted:
        st.warning("⚡ Watchdog thread'i yeniden başlattı")

    st.caption(f"URL hata: {radar._url_failures}/2 | Circuit: {'AÇIK' if radar._circuit_open else 'KAPALI'}")
    st.caption(f"Fetch interval: {radar._current_fetch_interval}s | Rate hits: {radar._rate_limit_hits}")
    st.caption("v4.0 SELF-HEALING | Market Radar Pro")

# ==================== HEADER ====================
h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
h1.title("Market Radar Pro")
h1.caption("v4.0 SELF-HEALING — Watchdog, kesilebilir sleep, bellek yönetimi, executor koruması")

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
h4.metric("Hata Sayısı", radar._consecutive_errors)

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


# ================================================================
# SAYFA 1: NORMAL SİNYALLER
# ================================================================

if page == "Normal Sinyaller":
    col_filters = st.columns([1, 1, 1, 1])
    mode_filter = col_filters[0].multiselect(
        "Sinyal Modu",
        ["FLASH", "CONFIRMED", "MACD PATTERN"],
        default=["FLASH", "CONFIRMED", "MACD PATTERN"],
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

    st.divider()
    col_side, col_main = st.columns([1, 5])

    def get_mode_css_class(mode):
        if "CONFIRMED" in mode: return "mode-confirmed"
        if "MACD" in mode: return "mode-macd"
        return "mode-flash"

    def label_css(s_type):
        return {"PUMP": "pump-label", "DUMP": "dump-label", "BUY": "buy-label", "SELL": "sell-label"}.get(s_type, "buy-label")

    def row_css(s_type, mode):
        if "MACD" in mode: return "row-macd"
        is_up = s_type in ("PUMP", "BUY")
        if "FLASH" in mode: return "row-flash-pump" if is_up else "row-flash-dump"
        return "row-conf-pump" if is_up else "row-conf-dump"

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

        if display_data:
            html = (
                "<table><tr>"
                "<th>Time</th><th>Symbol (4H ^/v)</th><th>Price</th>"
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
                macd_val = row.get('MACD_Pattern', '')
                if macd_val:
                    pat_cls = get_macd_pattern_css_class(macd_val)
                    macd_html_str = f"<span class='macd-pattern-tag {pat_cls}'>{macd_val}</span>"
                else:
                    macd_html_str = "-"
                vol_display = f"{row['Vol'] / 1000:.0f}k" if row['Vol'] > 0 else "-"
                ref_display = f"{row['Ref']:+.4f}" if row['Ref'] != 0 else "-"
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
# SAYFA 3: SİSTEM DURUMU
# ================================================================

elif page == "Sistem Durumu":
    st.subheader("Sistem Sağlık Paneli")

    col_s1, col_s2, col_s3 = st.columns(3)

    thread_alive = radar._worker_thread is not None and radar._worker_thread.is_alive()

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
        st.metric("Worker Thread", "✅ CANLI" if thread_alive else "⚠️ BAŞLATIYOR")

    st.divider()
    st.subheader("Bağlantı İstikrarı")

    elapsed = time.time() - radar.last_heartbeat
    if elapsed < 15:
        st.success(f"✅ Bağlantı sağlıklı — son heartbeat {elapsed:.1f}s önce")
    elif elapsed < 60:
        st.warning(f"⚠️ Bağlantı yavaş — son heartbeat {elapsed:.1f}s önce")
    else:
        st.error(f"❌ Bağlantı kesik — {elapsed:.1f}s önce. Watchdog müdahale ediyor.")

    st.info("""
    **v4.0 SELF-HEALING Değişiklikleri (v3.0'a göre):**

    **🔴 Kök Sorun 1 — Thread ölünce dirilemediyordu:**
    Eski: `if "thread_started" not in session_state` → sadece bir kez başlatıyordu
    Yeni: `ensure_worker_running()` her UI refresh'te `.is_alive()` kontrol eder, ölmüşse yeniden başlatır

    **🔴 Kök Sorun 2 — `time.sleep()` thread'i donduruyordu:**
    Eski: `time.sleep(retry_after)` → circuit breaker içinde 120s donabiliyordu
    Yeni: `_interruptible_sleep()` → 1s dilimlerinde, `stop_event` / `circuit_event` ile kesilebilir

    **🔴 Kök Sorun 3 — 15m cache büyüyordu (bellek sızıntısı):**
    Yeni: `_clean_15m_cache()` her 30 fetch'te bir 10 dakikadan eski kayıtları siler

    **🔴 Kök Sorun 4 — MACD executor zombie task biriktiriyordu:**
    Yeni: Executor instance'a taşındı, `stop_event` kontrolü ile görevler erken çıkıyor

    **🔴 Kök Sorun 5 — Worker exception'da tamamen ölüyordu:**
    Yeni: `while not stop_event` döngüsünde her exception yakalanıyor, loop kırılmıyor
    """)

    time.sleep(3)
    st.rerun()
