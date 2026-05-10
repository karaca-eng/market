import streamlit as st
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

# ==================== KONFİGÜRASYON ====================
MIN_VOL_3M            = 40000
MIN_CHG_3M            = 1.0
CONFIRM_CHG_15M       = 1.3
FAST_STRIKE_CHG       = 1.0
TRI_WINDOW            = 181
MAX_DISPLAY_ROWS      = 100
FETCH_INTERVAL        = 10
PUMP_DUMP_THRESHOLD   = 2.2
MACD_PATTERN_COOLDOWN = 180
_KLINE_SEMAPHORE      = threading.Semaphore(2)

BINANCE_REST_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

# ==================== SESSION ====================

def create_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36',
        ]),
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
    })
    retry = Retry(total=2, backoff_factor=1.5,
                  status_forcelist=[500, 502, 503, 504],
                  allowed_methods=["GET"], raise_on_status=False,
                  respect_retry_after_header=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=10)
    session.mount("https://", adapter)
    return session

# ==================== MACD ====================

def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    close = np.array(close_prices, dtype=float)
    def ema(data, span):
        k = 2 / (span + 1)
        out = [data[0]]
        for p in data[1:]:
            out.append(p * k + out[-1] * (1 - k))
        return np.array(out)
    macd = ema(close, fast) - ema(close, slow)
    sig  = ema(macd, signal)
    return macd, sig

def detect_whale_trap(m, s, volume_ratio=1.0):
    if len(m) < 6: return None
    macd_range = max(m[-6:-1]) - min(m[-6:-1])
    avg_move   = np.mean([abs(m[i] - m[i-1]) for i in range(-5, -1)])
    conditions = {
        "zero_above":     m[-1] > 0,
        "crossover":      m[-1] > s[-1] and m[-2] <= s[-2],
        "compression":    macd_range < avg_move * 2.5,
        "signal_turning": s[-1] > s[-2] > s[-3],
        "volume_confirm": volume_ratio > 1.8,
        "momentum_accel": (m[-1]-m[-2]) > (m[-2]-m[-3]) > 0,
    }
    score = sum(conditions.values())
    if score == 6: return "WHALE TRAP — EFSANE (6/6)"
    if score >= 4: return f"WHALE TRAP — Guclu ({score}/6)"
    if score >= 3: return f"WHALE TRAP — Zayif ({score}/6)"
    return None

def detect_final_breakout(m, s, close_prices):
    if len(m) < 6 or len(close_prices) < 7: return None
    h = m - s
    prices = np.array(close_prices)
    conditions = {
        "crossover":       m[-1] > s[-1] and m[-2] <= s[-2],
        "hist_expanding":  abs(h[-1]) > abs(h[-2]) > abs(h[-3]) and h[-1] > 0,
        "accel":           (m[-1]-m[-2]) > (m[-2]-m[-3]) > (m[-3]-m[-4]) > 0,
        "signal_flip":     s[-1] > s[-2] > s[-3],
        "higher_high":     prices[-1] > max(prices[-6:-1]),
        "strong_momentum": (m[-1]-m[-2]) > (s[-1]-s[-2]) * 2.0,
    }
    score = sum(conditions.values())
    if score == 6: return "FINAL BREAKOUT — EFSANE (6/6)"
    if score >= 4: return f"FINAL BREAKOUT — Guclu ({score}/6)"
    if score >= 3: return f"FINAL BREAKOUT — Erken ({score}/6)"
    return None

def detect_triple_cross(m, s):
    if len(m) < 10: return None
    crosses = []
    for i in range(len(m)-1, 1, -1):
        if m[i] > s[i] and m[i-1] <= s[i-1]:
            crosses.append({"bar_index": i, "level": m[i], "below_zero": m[i] < 0})
    if len(crosses) < 3: return None
    c1, c2, c3 = crosses[0], crosses[1], crosses[2]
    if not (m[-1] > s[-1] and m[-2] <= s[-2]): return None
    if not c3["below_zero"]: return None
    if not (c1["level"] > c2["level"] > c3["level"]): return None
    if (c1["bar_index"] - c2["bar_index"]) < 3: return None
    if (c2["bar_index"] - c3["bar_index"]) < 3: return None
    levels = f"[{c3['level']:.6f}->{c2['level']:.6f}->{c1['level']:.6f}]"
    diff1, diff2 = c2["level"]-c3["level"], c1["level"]-c2["level"]
    return f"TRIPLE CROSS — {'EFSANE' if diff2>diff1>0 else 'Guclu'} {levels}"

def detect_dalga(m, s):
    if len(m) < 50: return None, {}
    h = m - s
    threshold   = np.mean([abs(m[i]-s[i]) for i in range(-50, 0)]) * 0.20
    state       = 0
    meta        = {}
    step1_macds = []
    i = 1
    while i < len(m):
        idx = i - len(m)
        if state == 0:
            if (m[idx]<0 and s[idx]<0 and abs(m[idx]-s[idx])<threshold
                    and (m[idx]-m[idx-1])<=0 and (s[idx]-s[idx-1])<=0):
                step1_macds.append(m[idx])
                if len(step1_macds) >= 3:
                    state = 1; meta["step1_index"] = i; meta["step1_low"] = min(step1_macds)
            else:
                step1_macds = []
        elif state == 1:
            if m[idx-1]<=0 and m[idx]>0:
                state = 2; meta["step2_index"] = i; meta["step2_peak"] = m[idx]
        elif state == 2:
            if m[idx-1]>=0 and m[idx]<0:
                if m[idx] > meta["step1_low"]:
                    state = 3; meta["step3_index"] = i; meta["step3_low"] = m[idx]
                else:
                    state = 0; step1_macds = []; meta = {}
        elif state == 3:
            if m[idx] < 0:
                if m[idx] < meta["step1_low"]: state = 0; step1_macds = []; meta = {}
                else: meta["step3_low"] = min(meta["step3_low"], m[idx])
            elif m[idx-1]<=0 and m[idx]>0:
                state = 4; meta["step4_index"] = i
        elif state == 4:
            if m[idx]>0 and m[idx]>m[idx-1] and h[idx]>h[idx-1] and m[idx]>meta["step2_peak"]:
                state = 5; meta["step5_index"] = i; break
        i += 1
    labels = {5:"DALGA — AL SINYALI (5/5)", 4:"DALGA — TAKIPTE (4/5)",
              3:"DALGA — Geri Cekildi (3/5)", 2:"DALGA — Sifir Ustu (2/5)",
              1:"DALGA — Sikisma (1/5)"}
    return (labels[state], meta) if state >= 1 else (None, {})

# ==================== ALTIN BREAKOUT ====================

def find_swing_highs(high_prices, lookback=30, left=2, right=2):
    """Son lookback mumun swing high seviyelerini bulur."""
    prices = list(high_prices)
    n      = len(prices)
    start  = max(0, n - lookback - right)
    window = prices[start:]
    highs  = []
    for i in range(left, len(window) - right):
        c = window[i]
        if (all(c >= window[i-j] for j in range(1, left+1)) and
                all(c >= window[i+j] for j in range(1, right+1))):
            highs.append(c)
    return sorted(set(highs))

def cluster_resistance_levels(levels, tolerance=0.003):
    """Birbirine yakın direnç seviyelerini kümeler."""
    if not levels: return []
    clustered, group = [], [levels[0]]
    for lvl in levels[1:]:
        gm = sum(group) / len(group)
        if abs(lvl - gm) / gm <= tolerance:
            group.append(lvl)
        else:
            clustered.append(sum(group) / len(group))
            group = [lvl]
    clustered.append(sum(group) / len(group))
    return clustered

def get_broken_and_active_resistances(high_prices, current_price, lookback=30):
    """
    Döndürür:
        broken : fiyatın altında kalan (kırılan) direnç seviyeleri
        active : fiyatın üzerindeki (henüz kırılmamış) seviyeler
    """
    raw      = find_swing_highs(high_prices, lookback=lookback)
    all_lvls = cluster_resistance_levels(raw, tolerance=0.003)
    broken   = [l for l in all_lvls if l <= current_price]
    active   = [l for l in all_lvls if l > current_price
                and (l - current_price) / current_price <= 0.05]
    return broken, sorted(active)

def calculate_rsi(close_prices, period=14):
    """Wilder RSI hesaplama."""
    closes = np.array(close_prices, dtype=float)
    if len(closes) < period + 1: return None
    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period-1) + gains[i]) / period
        avg_loss = (avg_loss * (period-1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))

def detect_altin_breakout(m, s, close_prices, high_prices, volumes):
    """
    Altın Breakout tespiti.

    Zorunlu (hepsi gerekli):
      1. MACD crossover  — m[-1] > s[-1]  VE  m[-2] <= s[-2]
      2. MACD >= 0       — sıfırda veya üstünde
      3. En az 1 direnç kırılmış

    Bonus (+skor):
      4. Hacim > 20-mum ortalamasının 1.5 katı
      5. Kapanış kırılan direncin üstünde
      6. 2+ direnç aynı anda kırılmış
      7. RSI > 50
    """
    if len(m) < 3 or len(close_prices) < 32 or len(high_prices) < 32:
        return None

    # Zorunlu 1: Crossover
    if not (m[-1] > s[-1] and m[-2] <= s[-2]):
        return None

    # Zorunlu 2: MACD >= 0
    if m[-1] < 0:
        return None

    # Direnç seviyeleri (son kapanmamış mumu hariç tut)
    hist_highs              = list(high_prices[-32:-1])
    broken, _active         = get_broken_and_active_resistances(
                                  hist_highs, close_prices[-1], lookback=30)

    # Zorunlu 3: En az 1 direnç kırılmış
    if not broken:
        return None

    # Skor — 3 zorunlu koşul zaten sağlandı
    score = 3

    # Bonus 4: Hacim
    if len(volumes) >= 21:
        avg_vol = np.mean(list(volumes)[-21:-1])
        if avg_vol > 0 and volumes[-1] > avg_vol * 1.5:
            score += 1

    # Bonus 5: Kapanış kırılan en yakın direncin üstünde
    nearest_broken = max(broken)
    if close_prices[-1] >= nearest_broken:
        score += 1

    # Bonus 6: Çoklu kırılma
    if len(broken) >= 2:
        score += 1

    # Bonus 7: RSI > 50
    rsi = calculate_rsi(close_prices)
    if rsi is not None and rsi > 50:
        score += 1

    info = f"[{len(broken)} direnç kırıldı]" if len(broken) > 1 else ""

    if score >= 7: return f"ALTIN BREAKOUT — EFSANE (7/7) {info}".strip()
    if score >= 5: return f"ALTIN BREAKOUT — Güçlü ({score}/7) {info}".strip()
    if score >= 3: return f"ALTIN BREAKOUT — Erken ({score}/7) {info}".strip()
    return None

# ==================== run_signals() ====================

def run_signals(close_prices, volume_ratio=1.0, high_prices=None, volumes=None):
    result = {
        "ALTIN_BREAKOUT": None,
        "WHALE_TRAP":      None,
        "FINAL_BREAKOUT":  None,
        "TRIPLE_CROSS":    None,
        "DALGA":           None,
    }
    if len(close_prices) < 50:
        return result

    m, s         = calculate_macd(close_prices)
    dalga, _     = detect_dalga(m, s)

    result["WHALE_TRAP"]     = detect_whale_trap(m, s, volume_ratio)
    result["FINAL_BREAKOUT"] = detect_final_breakout(m, s, close_prices)
    result["TRIPLE_CROSS"]   = detect_triple_cross(m, s)
    result["DALGA"]          = dalga

    if high_prices is not None and len(high_prices) >= 32:
        vols = volumes if volumes is not None else []
        result["ALTIN_BREAKOUT"] = detect_altin_breakout(
            m, s, close_prices, high_prices, vols)

    return result

def get_signal_label(direction, chg):
    if abs(chg) >= PUMP_DUMP_THRESHOLD:
        return "PUMP" if direction == "up" else "DUMP"
    return "BUY" if direction == "up" else "SELL"

# ==================== ANA SINIF ====================

class MarketRadar:
    def __init__(self):
        self.history               = {}
        self.signals               = []
        self.stats_hourly          = {}
        self.stats_4h              = {}
        self.lock                  = threading.RLock()
        self.last_heartbeat        = 0
        self.total_pairs           = 0
        self.last_reset_hour       = datetime.now().hour
        self.last_reset_4h_block   = datetime.now().hour // 4
        self.price_cache_15m       = {}
        self.debug_log             = deque(maxlen=100)
        self.session               = create_session()
        self.session_created_at    = time.time()
        self._url_index            = 0
        self._url_failures         = 0
        self._url_lock             = threading.Lock()
        self._consecutive_errors   = 0
        self._total_requests       = 0
        self._rate_limit_hits      = 0
        self.macd_pattern_sent     = {}
        self.macd_pattern_candidates = {}
        self.macd_pattern_last_trigger = {}
        self._circuit_open         = False
        self._circuit_open_until   = 0
        self._circuit_event        = threading.Event()
        self._current_fetch_interval = FETCH_INTERVAL
        self._healthy_streak       = 0
        self._worker_thread        = None
        self._stop_event           = threading.Event()
        self._macd_executor        = ThreadPoolExecutor(max_workers=3, thread_name_prefix="macd")

    def log(self, msg):
        t    = datetime.now().strftime("%H:%M:%S")
        full = f"[{t}] {msg}"
        print(full, flush=True)
        with self.lock:
            self.debug_log.appendleft(full)

    # ── Watchdog ──────────────────────────────────────────
    def ensure_worker_running(self):
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self.log(">>> WATCHDOG: Thread ölü — yeniden başlatılıyor")
            self._stop_event.clear()
            self._circuit_event.clear()
            self._refresh_session_if_needed(force=True)
            t = threading.Thread(target=self._worker_loop,
                                 name="MarketRadarWorker", daemon=True)
            t.start()
            self._worker_thread = t
            self.log(f">>> WATCHDOG: Thread başlatıldı (id={t.ident})")
            return True
        return False

    # ── URL Yönetimi ──────────────────────────────────────
    def _get_current_url(self):
        with self._url_lock:
            return BINANCE_REST_URLS[self._url_index % len(BINANCE_REST_URLS)]

    def _rotate_url(self, reason="failure"):
        with self._url_lock:
            old = self._url_index
            self._url_index = (self._url_index + 1) % len(BINANCE_REST_URLS)
            self._url_failures = 0
            self.log(f"URL rotasyon [{reason}]: [{old}]→[{self._url_index}]")

    def _mark_url_success(self):
        with self._url_lock:
            self._url_failures    = 0
            self._consecutive_errors = 0
            self._healthy_streak += 1
            if self._healthy_streak > 10:
                self._current_fetch_interval = max(FETCH_INTERVAL, self._current_fetch_interval - 1)
                self._healthy_streak = 0

    def _mark_url_failure(self):
        with self._url_lock:
            self._url_failures     += 1
            self._consecutive_errors += 1
            self._healthy_streak    = 0
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
            except Exception as e:
                self.log(f"{url} → HATA: {str(e)[:60]}")
            time.sleep(0.5)
        return BINANCE_REST_URLS[0]

    # ── Circuit Breaker ───────────────────────────────────
    def _check_circuit(self):
        if self._circuit_open:
            if time.time() < self._circuit_open_until:
                return False
            self._circuit_open = False
            self._circuit_event.clear()
            self.log("Circuit breaker KAPANDI")
        return True

    def _open_circuit(self, duration=60):
        self._circuit_open       = True
        self._circuit_open_until = time.time() + duration
        self._circuit_event.set()
        self.log(f"Circuit breaker AÇILDI — {duration}s")

    def _interruptible_sleep(self, seconds):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._stop_event.is_set(): return
            self._circuit_event.wait(timeout=min(1.0, deadline - time.time()))
            self._circuit_event.clear()

    # ── Session ───────────────────────────────────────────
    def _refresh_session_if_needed(self, force=False):
        age = time.time() - self.session_created_at
        if force or age > 600:
            try:
                old = self.session
                self.session = create_session()
                self.session_created_at = time.time()
                try: old.close()
                except: pass
                self.log(f"Session yenilendi (age={age:.0f}s)")
                return True
            except Exception as e:
                self.log(f"Session yenileme hatası: {e}")
        return False

    # ── HTTP ──────────────────────────────────────────────
    def _safe_request(self, url, timeout=8):
        if self._stop_event.is_set() or not self._check_circuit():
            return None
        self._total_requests += 1
        try:
            time.sleep(random.uniform(0.01, 0.06))
            r = self.session.get(url, timeout=timeout)
            if r.status_code == 200:
                self._mark_url_success()
                for k, v in r.headers.items():
                    if 'X-MBX-USED-WEIGHT' in k.upper():
                        try:
                            if int(v) > 1000:
                                self._current_fetch_interval = min(60, self._current_fetch_interval + 3)
                        except: pass
                        break
                return r
            if r.status_code == 418:
                ra = int(r.headers.get('Retry-After', 120))
                self.log(f"418 IP BAN — {ra}s")
                self._rate_limit_hits += 1
                self._refresh_session_if_needed(force=True)
                self._open_circuit(ra)
                return None
            if r.status_code == 429:
                ra = int(r.headers.get('Retry-After', 30))
                self.log(f"429 Rate limit — {ra}s")
                self._rate_limit_hits += 1
                self._interruptible_sleep(ra)
                return None
            if r.status_code in (500, 502, 503, 504, 520, 521, 522, 523, 524):
                self._mark_url_failure(); return None
            return None
        except requests.exceptions.ConnectionError as e:
            if "remote end closed" in str(e).lower():
                self._refresh_session_if_needed(force=True)
            else:
                self._mark_url_failure()
            return None
        except requests.exceptions.Timeout:
            self._mark_url_failure(); return None
        except Exception as e:
            self.log(f"İstek hatası: {str(e)[:80]}")
            self._mark_url_failure(); return None

    # ── Reset / Cache ─────────────────────────────────────
    def check_resets(self):
        now = datetime.now()
        if now.hour != self.last_reset_hour:
            self.stats_hourly.clear(); self.last_reset_hour = now.hour
        if (now.hour // 4) != self.last_reset_4h_block:
            self.stats_4h.clear(); self.last_reset_4h_block = now.hour // 4

    def _clean_15m_cache(self):
        now     = time.time()
        expired = [k for k, (t, _) in self.price_cache_15m.items() if now - t > 600]
        for k in expired: del self.price_cache_15m[k]

    # ── Ticker ───────────────────────────────────────────
    def process_ticker(self, data):
        now = time.time()
        with self.lock:
            self.check_resets()
            self.total_pairs = len(data)
            for item in data:
                sym = item['s']
                if not sym.endswith('USDT'): continue
                price, qvol = float(item['c']), float(item['q'])
                if sym not in self.history:
                    self.history[sym] = deque(maxlen=400)
                self.history[sym].append((now, price, qvol))
                self.check_logic(sym, now)

    def check_logic(self, symbol, now):
        hist = list(self.history[symbol])
        if len(hist) < 5: return
        cur    = hist[-1]
        p1m    = next((x for x in reversed(hist) if now - x[0] >= 60), hist[0])
        p3m    = next((x for x in reversed(hist) if now - x[0] >= TRI_WINDOW), hist[0])
        c1     = ((cur[1] - p1m[1]) / p1m[1]) * 100
        c3     = ((cur[1] - p3m[1]) / p3m[1]) * 100
        vol3m  = cur[2] - p3m[2]
        vol1m  = cur[2] - p1m[2]

        if abs(c1) >= FAST_STRIKE_CHG and vol1m >= 50000:
            self.add_signal(symbol, cur[1], c1, 0, vol1m,
                            get_signal_label("up" if c1>0 else "down", c1), "FLASH", 40)
            return

        if vol3m >= MIN_VOL_3M and abs(c3) >= MIN_CHG_3M:
            p15 = self.get_15m_price(symbol)
            if p15:
                c15 = ((cur[1] - p15) / p15) * 100
                if ((c3>0 and c15>0) or (c3<0 and c15<0)) and abs(c15) >= CONFIRM_CHG_15M:
                    self.add_signal(symbol, cur[1], c3, c15, vol3m,
                                    get_signal_label("up" if c3>0 else "down", c3),
                                    "CONFIRMED", 55)

    def get_15m_price(self, symbol):
        now = time.time()
        if symbol in self.price_cache_15m:
            t, p = self.price_cache_15m[symbol]
            if now - t < 300: return p
        try:
            url = f"{self._get_current_url()}/fapi/v1/klines?symbol={symbol}&interval=15m&limit=2"
            r   = self._safe_request(url, timeout=5)
            if r and r.status_code == 200:
                p = float(r.json()[0][1])
                self.price_cache_15m[symbol] = (now, p)
                return p
        except Exception as e:
            self.log(f"15m price hata ({symbol}): {e}")
        return None

    def add_signal(self, symbol, price, chg_main, chg_ref, vol,
                   s_type, mode, score=50, macd_pattern=None):
        t_str    = datetime.now().strftime("%H:%M:%S")
        sym_c    = symbol.replace("USDT", "")
        stat_key = "PUMP" if s_type in ("PUMP","BUY") else "DUMP"
        with self.lock:
            for s in self.signals[:10]:
                if s.get('Symbol') == sym_c and s.get('Mode') == mode: return
            self.stats_hourly.setdefault(sym_c, {"PUMP":0,"DUMP":0})[stat_key] += 1
            self.stats_4h.setdefault(sym_c, {"PUMP":0,"DUMP":0})[stat_key] += 1
            self.signals.insert(0, {
                "Time": t_str, "Symbol": sym_c,
                "Price": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                "Chg": chg_main, "Ref": chg_ref, "Vol": vol,
                "P/D": s_type, "Mode": mode, "Score": score,
                "SnapP": self.stats_4h[sym_c]["PUMP"],
                "SnapD": self.stats_4h[sym_c]["DUMP"],
                "MACD_Pattern": macd_pattern or "",
            })
            self.log(f"SİNYAL: {sym_c} {s_type} {mode} {chg_main:+.2f}%" +
                     (f" | {macd_pattern}" if macd_pattern else ""))
            if len(self.signals) > MAX_DISPLAY_ROWS:
                self.signals.pop()

            if mode in ("FLASH", "CONFIRMED"):
                now_t  = time.time()
                last_t = self.macd_pattern_last_trigger.get(symbol, 0)
                if now_t - last_t >= MACD_PATTERN_COOLDOWN:
                    self.macd_pattern_last_trigger[symbol] = now_t
                    try:
                        self._macd_executor.submit(
                            self._run_macd_pattern_analysis, symbol, price)
                    except RuntimeError:
                        self.log("MACD executor kapalı — görev atlandı")

    # ── MACD Pattern Kline Çekimi ─────────────────────────
    def _fetch_klines_for_pattern(self, symbol, interval="15m", limit=200):
        with _KLINE_SEMAPHORE:
            url = (f"{self._get_current_url()}/fapi/v1/klines"
                   f"?symbol={symbol}&interval={interval}&limit={limit}")
            try:
                resp = self._safe_request(url, timeout=8)
                if not resp or resp.status_code != 200: return None
                raw     = resp.json()
                closes  = [float(c[4]) for c in raw]
                highs   = [float(c[2]) for c in raw]   # HIGH — Altın Breakout için
                volumes = [float(c[5]) for c in raw]
                return closes, highs, volumes
            except Exception as e:
                self.log(f"Kline hatası ({symbol}): {e}")
                return None

    # ── MACD Pattern Analiz ───────────────────────────────
    def _run_macd_pattern_analysis(self, symbol, price):
        if self._stop_event.is_set(): return
        result = self._fetch_klines_for_pattern(symbol, "15m", 200)
        if result is None: return
        closes, highs, volumes = result
        if len(closes) < 50: return

        volume_ratio = 1.0
        if len(volumes) >= 21:
            avg = np.mean(volumes[-21:-1])
            if avg > 0: volume_ratio = volumes[-1] / avg

        signals = run_signals(closes, volume_ratio,
                              high_prices=highs, volumes=volumes)
        sym_c   = symbol.replace("USDT", "")
        now     = time.time()

        # Altın Breakout en yüksek önceliğe sahip
        PRIORITY = {
            "ALTIN_BREAKOUT": 10,
            "WHALE_TRAP":      4,
            "FINAL_BREAKOUT":  3,
            "TRIPLE_CROSS":    2,
            "DALGA":           1,
        }
        best_pattern, best_score, best_key = None, 0, None

        for key, value in signals.items():
            if value is None: continue
            strength = 0
            if "EFSANE" in value:   strength = 100
            elif any(x in value for x in ["Güçlü","Guclu"]): strength = 70
            elif any(x in value for x in ["Erken","Zayif","TAKIPTE","AL SINYALI"]): strength = 50
            elif any(x in value for x in ["Sikisma","Sifir Ustu","Geri Cekildi"]): strength = 30
            total = strength + PRIORITY.get(key, 0)
            if total > best_score:
                best_score = total; best_pattern = value; best_key = key

        with self.lock:
            if best_pattern:
                self.macd_pattern_candidates[sym_c] = {
                    "Sembol": sym_c,
                    "Fiyat":  f"{price:.4f}" if price < 1 else f"{price:.2f}",
                    "Pattern":    best_pattern,
                    "PatternTip": best_key or "",
                    "Guncelleme": datetime.now().strftime("%H:%M:%S"),
                }
                if now - self.macd_pattern_sent.get(sym_c, 0) < MACD_PATTERN_COOLDOWN:
                    for sig in self.signals[:20]:
                        if sig.get('Symbol') == sym_c:
                            sig['MACD_Pattern'] = best_pattern; break
                else:
                    self.macd_pattern_sent[sym_c] = now
                    updated = False
                    for sig in self.signals[:20]:
                        if sig.get('Symbol') == sym_c:
                            sig['MACD_Pattern'] = best_pattern
                            updated = True
                            self.log(f"MACD PATTERN: {sym_c} → {best_pattern}")
                            break
                    if not updated:
                        self.add_signal(symbol=symbol, price=price,
                                        chg_main=0.0, chg_ref=0.0, vol=0,
                                        s_type="BUY", mode="MACD PATTERN",
                                        score=60, macd_pattern=best_pattern)
            else:
                self.macd_pattern_candidates.pop(sym_c, None)

    # ── Worker Loop ───────────────────────────────────────
    def _worker_loop(self):
        self.log(">>> WORKER LOOP BAŞLADI (v4.1 Altın Breakout)")
        self.get_working_rest_url()
        fetch_count = cache_clean_ctr = 0

        while not self._stop_event.is_set():
            try:
                if self._circuit_open:
                    remaining = max(1, int(self._circuit_open_until - time.time()))
                    self.log(f"Circuit açık — {remaining}s bekleniyor")
                    self._interruptible_sleep(remaining)
                    continue

                url      = f"{self._get_current_url()}/fapi/v1/ticker/24hr"
                response = self._safe_request(url, timeout=10)
                fetch_count     += 1
                cache_clean_ctr += 1

                if response and response.status_code == 200:
                    self.last_heartbeat = time.time()
                    raw = response.json()
                    self.process_ticker([
                        {'s': x['symbol'], 'c': x['lastPrice'], 'q': x['quoteVolume']}
                        for x in raw
                    ])
                    if fetch_count % 200 == 0:
                        self._refresh_session_if_needed(force=True)
                    if cache_clean_ctr >= 30:
                        self._clean_15m_cache(); cache_clean_ctr = 0
                    if fetch_count % 10 == 0:
                        self.log(
                            f"Fetch #{fetch_count} | "
                            f"Pairs:{self.total_pairs} Signals:{len(self.signals)} "
                            f"MACD:{len(self.macd_pattern_candidates)} "
                            f"Interval:{self._current_fetch_interval}s"
                        )
                    self._interruptible_sleep(self._current_fetch_interval)
                else:
                    wait = min(3 * (self._consecutive_errors + 1), 45)
                    self._interruptible_sleep(wait)

            except Exception as e:
                wait = min(5 * (self._consecutive_errors + 1), 60)
                self.log(f"WORKER HATA: {str(e)[:120]} — {wait}s bekle")
                self._interruptible_sleep(wait)

        self.log(">>> WORKER LOOP DURDU")

# ==================== STREAMLIT ====================

@st.cache_resource
def get_radar_instance():
    return MarketRadar()

st.set_page_config(layout="wide", page_title="Market Radar Pro v4.1")

st.markdown("""<style>
.main{background-color:#0e1117}
.status-live{color:#00ff88;font-weight:bold;border:1px solid #00ff88;padding:2px 10px;border-radius:15px;font-size:.8rem}
.status-offline{color:#ff4b4b;font-weight:bold;border:1px solid #ff4b4b;padding:2px 10px;border-radius:15px;font-size:.8rem}
.status-warn{color:#f1c40f;font-weight:bold;border:1px solid #f1c40f;padding:2px 10px;border-radius:15px;font-size:.8rem}
.pump-label{background-color:#00ff88;color:black;padding:2px 8px;border-radius:4px;font-weight:bold}
.dump-label{background-color:#ff4b4b;color:white;padding:2px 8px;border-radius:4px;font-weight:bold}
.buy-label{background-color:#1a7f4b;color:#afffcf;padding:2px 8px;border-radius:4px;font-weight:bold}
.sell-label{background-color:#7f1a1a;color:#ffcfcf;padding:2px 8px;border-radius:4px;font-weight:bold}
.mode-confirmed{background-color:#1abc9c;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold}
.mode-flash{background-color:#e67e22;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold}
.mode-macd{background-color:#8e44ad;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold}
.macd-pattern-tag{padding:2px 8px;border-radius:4px;font-size:.78rem;font-weight:bold;display:inline-block;white-space:nowrap}
/* Altın Breakout renkleri */
.macd-altin-efsane{background-color:#2a1f00;color:#ffd700;border:1px solid #ffd700}
.macd-altin-guclu{background-color:#1f1800;color:#f0c040;border:1px solid #f0c040}
.macd-altin-erken{background-color:#181400;color:#b8a030;border:1px solid #b8a030}
/* Diğer pattern renkleri */
.macd-whale-efsane{background-color:#1a3a4a;color:#00d4ff;border:1px solid #00d4ff}
.macd-whale-guclu{background-color:#1a2a3a;color:#4db8ff;border:1px solid #4db8ff}
.macd-whale-zayif{background-color:#1a2020;color:#88aabb;border:1px solid #557788}
.macd-breakout-efsane{background-color:#1a3a1a;color:#00ff88;border:1px solid #00ff88}
.macd-breakout-guclu{background-color:#1a2a1a;color:#4dff88;border:1px solid #4dff88}
.macd-breakout-erken{background-color:#1a2515;color:#88cc66;border:1px solid #88cc66}
.macd-triple-efsane{background-color:#2a1a3a;color:#ff6bff;border:1px solid #ff6bff}
.macd-triple-guclu{background-color:#221530;color:#cc88dd;border:1px solid #cc88dd}
.macd-dalga-al{background-color:#3a2a0a;color:#ffcc00;border:1px solid #ffcc00}
.macd-dalga-takip{background-color:#2a2010;color:#ccaa44;border:1px solid #ccaa44}
.macd-dalga-diger{background-color:#1a1a10;color:#999966;border:1px solid #777755}
.stat-card{background-color:#1e2127;padding:10px;border-radius:10px;margin-bottom:10px;border-left:5px solid #f1c40f}
.debug-box{background-color:#1a1a2e;border:1px solid #333;border-radius:8px;padding:10px;font-family:monospace;font-size:.75rem;color:#aaa;max-height:200px;overflow-y:auto}
.watchdog-ok{color:#00ff88;font-size:.75rem}
.watchdog-warn{color:#f1c40f;font-size:.75rem}
table{width:100%;border-collapse:collapse}
th,td{white-space:nowrap;padding:12px 15px;text-align:left;border-bottom:1px solid #222}
.sym-link{color:#f1c40f;text-decoration:none;font-weight:bold;font-size:1.1rem}
.sym-link:hover{color:#fff}
.green-arrow{color:#00ff88;font-weight:bold}
.red-arrow{color:#ff4b4b;font-weight:bold}
.row-flash-pump{background-color:rgba(0,255,136,.22)!important;border-left:5px solid #00ff88!important}
.row-flash-dump{background-color:rgba(255,75,75,.22)!important;border-left:5px solid #ff4b4b!important}
.row-conf-pump{background-color:rgba(0,255,136,.08)!important}
.row-conf-dump{background-color:rgba(255,75,75,.08)!important}
.row-macd{background-color:rgba(142,68,173,.12)!important;border-left:3px solid #8e44ad!important}
.row-altin{background-color:rgba(255,215,0,.10)!important;border-left:4px solid #ffd700!important}
</style>""", unsafe_allow_html=True)

radar            = get_radar_instance()
watchdog_restart = radar.ensure_worker_running()

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sayfa Seç",
    ["Normal Sinyaller", "MACD Pattern Radar", "Sistem Durumu"], index=0)
st.sidebar.markdown("---")

with st.sidebar:
    elapsed   = time.time() - radar.last_heartbeat
    url_short = radar._get_current_url().replace("https://", "")
    if elapsed < 15:   st.success(f"CANLI | {url_short}")
    elif elapsed < 30: st.warning(f"YAVAS | {url_short} | {elapsed:.0f}s")
    else:              st.error(f"KESİNTİ | {url_short} | {elapsed:.0f}s")

    alive = radar._worker_thread is not None and radar._worker_thread.is_alive()
    if alive: st.markdown('<span class="watchdog-ok">● Thread canlı</span>', unsafe_allow_html=True)
    else:     st.markdown('<span class="watchdog-warn">⟳ Thread başlatılıyor...</span>', unsafe_allow_html=True)
    if watchdog_restart: st.warning("⚡ Watchdog devreye girdi")
    st.caption(f"Circuit: {'AÇIK' if radar._circuit_open else 'KAPALI'} | "
               f"Interval: {radar._current_fetch_interval}s | Rate: {radar._rate_limit_hits}")
    st.caption("v4.1 ALTIN BREAKOUT | Market Radar Pro")

# ── Header ────────────────────────────────────────────────
h1, h2, h3, h4 = st.columns([2,1,1,1])
h1.title("Market Radar Pro")
h1.caption("v4.1 — Altın Breakout: 15dk direnç kırılması + MACD≥0 + crossover")
elapsed = time.time() - radar.last_heartbeat
if elapsed < 15:   status_html = '<span class="status-live">● SYSTEM LIVE</span>'
elif elapsed < 30: status_html = f'<span class="status-warn">● YAVAS ({elapsed:.0f}s)</span>'
else:              status_html = f'<span class="status-offline">● RECONNECTING ({elapsed:.0f}s)</span>'
h2.markdown(f"<div style='margin-top:10px;'>{status_html}</div>", unsafe_allow_html=True)
h2.markdown('<a href="https://x.com/SinyalEngineer" target="_blank" style="color:white;text-decoration:none;">X @SinyalEngineer</a>', unsafe_allow_html=True)
h3.metric("Pairs", radar.total_pairs)
h3.metric("Signals", len(radar.signals))
h4.metric("MACD Aday", len(radar.macd_pattern_candidates))
h4.metric("Hata Sayısı", radar._consecutive_errors)

st.divider()

with st.expander("Debug Log", expanded=False):
    with radar.lock: logs = list(radar.debug_log)
    if logs:
        st.markdown("<div class='debug-box'>" + "<br>".join(logs) + "</div>", unsafe_allow_html=True)
    else:
        st.info("Henüz log yok...")

st.divider()

# ── Yardımcı Fonksiyonlar ─────────────────────────────────

def get_macd_pattern_css_class(pattern):
    if not pattern: return ""
    p = pattern.upper()
    # ALTIN önce kontrol et
    if "ALTIN BREAKOUT" in p:
        if "EFSANE" in p:                     return "macd-altin-efsane"
        if "GÜÇLÜ" in p or "GUCLU" in p:     return "macd-altin-guclu"
        return "macd-altin-erken"
    if "WHALE TRAP" in p:
        if "EFSANE" in p: return "macd-whale-efsane"
        if "GUCLU"  in p: return "macd-whale-guclu"
        return "macd-whale-zayif"
    if "FINAL BREAKOUT" in p:
        if "EFSANE" in p: return "macd-breakout-efsane"
        if "GUCLU"  in p: return "macd-breakout-guclu"
        return "macd-breakout-erken"
    if "TRIPLE CROSS" in p:
        return "macd-triple-efsane" if "EFSANE" in p else "macd-triple-guclu"
    if "DALGA" in p:
        if "AL SINYALI" in p: return "macd-dalga-al"
        if "TAKIPTE"    in p: return "macd-dalga-takip"
        return "macd-dalga-diger"
    return ""

def get_pattern_score_sort(pattern):
    if not pattern: return 0
    p     = pattern.upper()
    score = 0
    if   "EFSANE"    in p: score += 100
    elif "GÜÇLÜ"     in p or "GUCLU" in p: score += 70
    elif "AL SINYALI" in p: score += 60
    elif "TAKIPTE"   in p: score += 50
    elif "ERKEN"     in p: score += 40
    elif "ZAYIF"     in p: score += 30
    else:                   score += 20
    # Altın Breakout en yüksek tür bonusu
    if   "ALTIN BREAKOUT" in p: score += 5
    elif "WHALE TRAP"     in p: score += 4
    elif "FINAL BREAKOUT" in p: score += 3
    elif "TRIPLE CROSS"   in p: score += 2
    elif "DALGA"          in p: score += 1
    return score

def row_css(s_type, mode, macd_pattern=""):
    if macd_pattern and "ALTIN BREAKOUT" in macd_pattern.upper():
        return "row-altin"
    if "MACD" in mode: return "row-macd"
    is_up = s_type in ("PUMP","BUY")
    if "FLASH" in mode: return "row-flash-pump" if is_up else "row-flash-dump"
    return "row-conf-pump" if is_up else "row-conf-dump"

def get_mode_css(mode):
    if "CONFIRMED" in mode: return "mode-confirmed"
    if "MACD"      in mode: return "mode-macd"
    return "mode-flash"

def label_css(s_type):
    return {"PUMP":"pump-label","DUMP":"dump-label",
            "BUY":"buy-label","SELL":"sell-label"}.get(s_type,"buy-label")

# ================================================================
# SAYFA 1: NORMAL SİNYALLER
# ================================================================
if page == "Normal Sinyaller":
    col_f = st.columns([1,1,1,1])
    mode_filter  = col_f[0].multiselect("Sinyal Modu",
        ["FLASH","CONFIRMED","MACD PATTERN"],
        default=["FLASH","CONFIRMED","MACD PATTERN"], key="mode_filter")
    pd_filter    = col_f[1].multiselect("Yön",
        ["PUMP","BUY","DUMP","SELL"],
        default=["PUMP","BUY","DUMP","SELL"], key="pd_filter")
    search_query = col_f[2].text_input("Symbol Filtre", placeholder="BTC...", key="search").upper()
    macd_only    = col_f[3].checkbox("Sadece MACD Pattern'li", value=False)

    st.divider()
    col_side, col_main = st.columns([1,5])

    with col_side:
        st.subheader("Top 5 Activity")
        with radar.lock:
            sorted_stats = sorted(radar.stats_hourly.items(),
                                  key=lambda x: x[1]['PUMP']+x[1]['DUMP'],
                                  reverse=True)[:5]
        for sym, counts in sorted_stats:
            tv = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
            st.markdown(f"""<div class="stat-card">
                <a href="{tv}" target="_blank" class="sym-link">{sym}</a><br>
                <small><span class="green-arrow">{counts["PUMP"]}</span> |
                <span class="red-arrow">{counts["DUMP"]}</span></small>
            </div>""", unsafe_allow_html=True)

    with col_main:
        st.subheader("Intelligence Stream")
        with radar.lock: signals = list(radar.signals)

        disp = signals
        if search_query: disp = [s for s in disp if search_query in s['Symbol']]
        if mode_filter:  disp = [s for s in disp if s['Mode'] in mode_filter]
        if pd_filter:    disp = [s for s in disp if s['P/D'] in pd_filter]
        if macd_only:    disp = [s for s in disp if s.get('MACD_Pattern')]

        if disp:
            html = ("<table><tr><th>Time</th><th>Symbol (4H ^/v)</th><th>Price</th>"
                    "<th>Momentum</th><th>15m Ref</th><th>Vol</th>"
                    "<th>Status</th><th>Type</th><th>MACD Pattern</th></tr>")
            for row in disp:
                sym       = row['Symbol']
                tv        = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
                p_type    = row['P/D']
                mode      = row['Mode']
                macd_val  = row.get('MACD_Pattern','')
                r_cls     = row_css(p_type, mode, macd_val)
                if macd_val:
                    pat_cls      = get_macd_pattern_css_class(macd_val)
                    macd_html    = f"<span class='macd-pattern-tag {pat_cls}'>{macd_val}</span>"
                else:
                    macd_html = "-"
                vol_d = f"{row['Vol']/1000:.0f}k" if row['Vol'] > 0 else "-"
                ref_d = f"{row['Ref']:+.4f}"      if row['Ref'] != 0  else "-"
                html += (f"<tr class='{r_cls}'>"
                         f"<td>{row['Time']}</td>"
                         f"<td><a href='{tv}' target='_blank' class='sym-link'>{sym}</a> "
                         f"<small class='green-arrow'>{row['SnapP']}</small> "
                         f"<small class='red-arrow'>{row['SnapD']}</small></td>"
                         f"<td>{row['Price']}</td>"
                         f"<td style='font-weight:bold;'>{row['Chg']:+.2f}%</td>"
                         f"<td>{ref_d}</td><td>{vol_d}</td>"
                         f"<td><span class='{get_mode_css(mode)}'>{mode}</span></td>"
                         f"<td><span class='{label_css(p_type)}'>{p_type}</span></td>"
                         f"<td>{macd_html}</td></tr>")
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
    <div style="background-color:#1a1030;border-left:4px solid #8e44ad;
                padding:12px 16px;border-radius:4px;margin-bottom:16px;">
        <b style="color:#c39bd3;">MACD Pattern Radar — 5 Pattern</b><br>
        <span style="color:#d5dbdb;font-size:.9rem;">
        <b style="color:#ffd700;">ALTIN BREAKOUT</b> — 15dk direnç kırıldı + MACD≥0 + crossover + RSI/Hacim/Çoklu kırılma bonusu<br>
        <b style="color:#00d4ff;">WHALE TRAP</b> — Sıfır üzerinde sıkışma sonrası kesişim + ivme<br>
        <b style="color:#00ff88;">FINAL BREAKOUT</b> — Histogram genişliyor, momentum artıyor, HH<br>
        <b style="color:#ff6bff;">TRIPLE CROSS</b> — 3× yukarı kesişim, her biri öncekinden yüksek<br>
        <b style="color:#ffcc00;">DALGA</b> — 5 adımlı: Sıkışma→Sıfır→Geri→Tekrar→AL<br>
        </span>
    </div>""", unsafe_allow_html=True)

    col_m = st.columns([1,1,1])
    pat_search  = col_m[0].text_input("Symbol Ara", placeholder="BTC...", key="pat_search").upper()
    pat_filter  = col_m[1].multiselect("Pattern Filtre",
        ["ALTIN BREAKOUT","WHALE TRAP","FINAL BREAKOUT","TRIPLE CROSS","DALGA"],
        default=["ALTIN BREAKOUT","WHALE TRAP","FINAL BREAKOUT","TRIPLE CROSS","DALGA"],
        key="pat_filter")
    min_str     = col_m[2].selectbox("Min Güç",
        ["Tümü","Erken+","Güçlü+","EFSANE"], index=0, key="min_str")

    st.divider()

    with radar.lock: candidates = dict(radar.macd_pattern_candidates)

    filtered = {}
    for sym, info in candidates.items():
        pat = info.get("Pattern","")
        if pat_search and pat_search not in sym: continue
        if pat_filter and not any(pf in pat for pf in pat_filter): continue
        if min_str == "EFSANE"  and "EFSANE" not in pat: continue
        if min_str == "Güçlü+" and not any(x in pat for x in ["EFSANE","Güçlü","Guclu"]): continue
        if min_str == "Erken+"  and not any(x in pat for x in
            ["EFSANE","Güçlü","Guclu","Erken","Zayif","AL SINYALI","TAKIPTE"]): continue
        filtered[sym] = info

    sorted_c = sorted(filtered.items(),
                      key=lambda x: get_pattern_score_sort(x[1].get("Pattern","")),
                      reverse=True)

    if sorted_c:
        html = ("<table><tr><th>Symbol</th><th>Fiyat</th>"
                "<th>Pattern</th><th>Güncelleme</th></tr>")
        for sym, info in sorted_c:
            tv  = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
            pat = info.get("Pattern","")
            cls = get_macd_pattern_css_class(pat)
            row_bg = "row-altin" if "ALTIN BREAKOUT" in pat.upper() else "row-macd"
            html += (f"<tr class='{row_bg}'>"
                     f"<td><a href='{tv}' target='_blank' class='sym-link'>{sym}</a></td>"
                     f"<td>{info['Fiyat']}</td>"
                     f"<td><span class='macd-pattern-tag {cls}' style='font-size:.95rem;'>"
                     f"{pat}</span></td>"
                     f"<td style='color:#666;'>{info.get('Guncelleme','N/A')}</td></tr>")
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)
        st.caption(f"Gösterilen: {len(sorted_c)} | Toplam: {len(candidates)}")
    else:
        st.info("MACD pattern taraması yapılıyor...")

    time.sleep(2)
    st.rerun()

# ================================================================
# SAYFA 3: SİSTEM DURUMU
# ================================================================
elif page == "Sistem Durumu":
    st.subheader("Sistem Sağlık Paneli")
    c1, c2, c3 = st.columns(3)
    alive = radar._worker_thread is not None and radar._worker_thread.is_alive()

    with c1:
        st.metric("Toplam İstek",    radar._total_requests)
        st.metric("Rate Limit",      radar._rate_limit_hits)
        st.metric("Ardışık Hata",    radar._consecutive_errors)
    with c2:
        st.metric("Aktif URL",       radar._get_current_url().replace("https://",""))
        st.metric("URL Hata",        f"{radar._url_failures}/2")
        st.metric("Circuit Breaker", "AÇIK" if radar._circuit_open else "KAPALI")
    with c3:
        st.metric("Fetch Interval",  f"{radar._current_fetch_interval}s")
        st.metric("Session Yaşı",    f"{time.time()-radar.session_created_at:.0f}s")
        st.metric("Worker Thread",   "✅ CANLI" if alive else "⚠️ BAŞLATIYOR")

    st.divider()
    elapsed = time.time() - radar.last_heartbeat
    if elapsed < 15:  st.success(f"✅ Bağlantı sağlıklı — {elapsed:.1f}s önce")
    elif elapsed < 60: st.warning(f"⚠️ Yavaş — {elapsed:.1f}s önce")
    else:             st.error(f"❌ Kesik — {elapsed:.1f}s önce. Watchdog aktif.")

    st.info("""
    **v4.1 Altın Breakout — Nasıl Çalışır?**

    **3 Zorunlu Koşul (hepsi sağlanmalı):**
    - MACD crossover: mavi çizgi kırmızıyı yukarı kesmiş olmalı
    - MACD ≥ 0: kesişim sıfır çizgisinde veya üstünde gerçekleşmeli
    - En az 1 swing high direnci kırılmış (son 30 mumun pivot tepeleri)

    **4 Bonus Koşul (skoru artırır, 3+bonus = toplam 7 üzerinden):**
    - Hacim > 20 mumun ortalamasının 1.5 katı
    - Kapanış kırılan direncin üstünde
    - 2+ direnç seviyesi aynı anda kırılmış
    - RSI > 50 (momentum doğrulaması)

    **Etiket:** EFSANE (7/7) → Güçlü (5-6/7) → Erken (3-4/7)
    """)

    time.sleep(3)
    st.rerun()
