import streamlit as st
import pandas as pd
import numpy as np
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

# MACD Paralel Ayarları
MACD_MIN_CANDLES = 3
MACD_MAX_CANDLES = 8
MACD_COOLDOWN = 180
MACD_EXECUTOR = ThreadPoolExecutor(max_workers=15)

# BIG MOVE AYARLARI (LONG)
BIGMOVE_EXECUTOR = ThreadPoolExecutor(max_workers=20)
BIGMOVE_COOLDOWN = 600
BB_SQUEEZE_LOOKBACK = 100
BB_SQUEEZE_PERCENTILE = 5
MA200_MIN_BARS_BELOW = 20
MACD_RESISTANCE_LOOKBACK = 20

# SHORT BIG MOVE AYARLARI
SHORT_EXECUTOR = ThreadPoolExecutor(max_workers=20)
SHORT_COOLDOWN = 600
MA200_MIN_BARS_ABOVE = 20
RSI_OVERBOUGHT = 70
RSI_BEARISH_DIV_LOOKBACK = 14
DEATH_CROSS_MA_FAST = 50
DEATH_CROSS_MA_SLOW = 200
BB_UPPER_REJECTION_BARS = 3

# SPOT AYARLARI - DENGELİ BALINA FOKUSLU
SPOT_EXECUTOR = ThreadPoolExecutor(max_workers=15)
SPOT_FETCH_INTERVAL = 3
SPOT_MIN_VOL = 50000        # 500K → 100K (daha fazla sembol yakalasın)
SPOT_LARGE_TRADE_USD = 10000   # 100K → 50K (large daha erken yakalansın)
SPOT_WHALE_TRADE_USD = 5000  # 300K → 150K (whale daha sık görünsün)
SPOT_MEGA_WHALE_USD = 500000   # 1M → 500K (mega nadir zaten, yakalansın)
SPOT_COOLDOWN = 45             # 60 → 45 (daha sık güncelleme)
MAX_SPOT_SIGNALS = 150         # 100 → 150 (daha fazla sinyal göster)
SPOT_MAX_INTERESTING = 80      # 50 → 80 (daha fazla sembol analiz etsin)

BINANCE_REST_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
]

# Spot için sağlıklı URL listesi (futures gibi)
BINANCE_SPOT_URLS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
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
        self.spot_base_url = BINANCE_SPOT_URLS[0]
        self.price_cache_15m = {}
        self.debug_log = []

        # MACD state
        self.macd_sent = {}
        self.macd_sent_keys = {}
        self.macd_candidates = {}
        self.macd_last_trigger = {}

        # BIG MOVE LONG state
        self.bigmove_signals = []
        self.bigmove_sent = {}
        self.bigmove_candidates = {}
        self.bigmove_last_trigger = {}

        # BIG MOVE SHORT state
        self.shortmove_signals = []
        self.shortmove_sent = {}
        self.shortmove_candidates = {}
        self.shortmove_last_trigger = {}

        # SPOT state - Balina fokuslu
        self.spot_signals = []
        self.spot_last_trade_id = {}
        self.spot_last_fetch = {}
        self.spot_price_cache = {}
        self.spot_last_heartbeat = 0
        self.spot_total_pairs = 0
        self.spot_failed_urls = set()
        self.spot_url_fail_count = {}

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
                    self.log(f"✅ Binance Futures OK: {url}")
                    return url
            except Exception as e:
                self.log(f"❌ {url} → HATA: {e}")
        self.log("🔴 Hiçbir Futures URL'e bağlanılamadı!")
        return self.rest_base_url

    # ==================== SPOT BAĞLANTI YÖNETİMİ (FUTURES GİBİ) ====================
    def get_working_spot_url(self):
        """Futures gibi çalışan, sağlıklı spot URL bulucu"""
        for url in BINANCE_SPOT_URLS:
            try:
                r = requests.get(f"{url}/api/v3/ping", timeout=3)
                if r.status_code == 200:
                    self.spot_base_url = url
                    self.log(f"✅ Binance Spot OK: {url}")
                    return url
            except Exception as e:
                self.log(f"❌ Spot {url} → HATA: {e}")
        self.log("🔴 Hiçbir Spot URL'e bağlanılamadı!")
        return self.spot_base_url

    def _spot_request(self, endpoint, timeout=5):
        """Spot istekleri için otomatik failover (futures gibi)"""
        urls_to_try = [self.spot_base_url] + [u for u in BINANCE_SPOT_URLS if u != self.spot_base_url]
        last_error = None
        for url in urls_to_try:
            try:
                full_url = f"{url}{endpoint}"
                r = requests.get(full_url, timeout=timeout, headers=self.headers)
                if r.status_code == 200:
                    if url != self.spot_base_url:
                        self.spot_base_url = url
                        self.log(f"🔄 Spot URL değişti: {url}")
                    return r.json()
                elif r.status_code == 429:
                    self.log(f"⚠️ Spot rate limit: {url}")
                    time.sleep(1)
                else:
                    self.log(f"⚠️ Spot HTTP {r.status_code}: {url}")
            except Exception as e:
                last_error = e
                self.log(f"❌ Spot {url} → {e}")
                continue
        self.log(f"🔴 Tüm Spot URL'ler başarısız! Son hata: {last_error}")
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
                self._maybe_trigger_shortmove(symbol, price, now)

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
                "MACD": macd_tag or "",
            })
            self.log(f"🚨 SİNYAL: {sym_clean} {s_type} {mode} {chg_main:+.2f}%" +
                     (f" | {macd_tag}" if macd_tag else ""))
            if len(self.signals) > MAX_DISPLAY_ROWS:
                self.signals.pop()

    # ================================================================
    # SPOT İŞLEMLERİ - SADECE BALINA FOKUSLU
    # ================================================================

    def fetch_spot_ticker(self):
        """Spot piyasasındaki tüm USDT çiftlerini çek - failover destekli"""
        data = self._spot_request("/api/v3/ticker/24hr", timeout=8)
        if data is None:
            return []
        return data

    def fetch_spot_recent_trades(self, symbol, limit=100):
        """Belirli bir spot sembol için son işlemleri çek - failover destekli"""
        data = self._spot_request(f"/api/v3/trades?symbol={symbol}&limit={limit}", timeout=5)
        if data is None:
            return []
        return data

    def process_spot_ticker(self, data):
        """Spot ticker verilerini işle - sadece balina potansiyeli olan semboller"""
        now = time.time()
        interesting = []

        for item in data:
            symbol = item.get('symbol', '')
            if not symbol.endswith('USDT'):
                continue
            try:
                price = float(item['lastPrice'])
                chg_pct = float(item['priceChangePercent'])
                quote_vol = float(item['quoteVolume'])

                # Düşük hacimli atla
                if quote_vol < SPOT_MIN_VOL:
                    continue

                # Sadece ilginç durumlar: yüksek hacim VEYA büyük değişim
                is_interesting = (
                    abs(chg_pct) >= 3.0 or
                    quote_vol >= 10_000_000 or
                    (abs(chg_pct) >= 1.5 and quote_vol >= 2_000_000)
                )

                if is_interesting:
                    interesting.append({
                        'symbol': symbol,
                        'price': price,
                        'chg_pct': chg_pct,
                        'quote_vol': quote_vol,
                    })
            except:
                continue

        with self.lock:
            self.spot_last_heartbeat = now
            self.spot_total_pairs = len([x for x in data if x.get('symbol', '').endswith('USDT')])

        # En ilginç sembolleri sırala ve işle
        interesting.sort(key=lambda x: x['quote_vol'], reverse=True)
        for item in interesting[:SPOT_MAX_INTERESTING]:
            SPOT_EXECUTOR.submit(self._analyze_spot_whale, item, now)

    def _analyze_spot_whale(self, item, now):
        """Spot sembol için balina analizi - sadece büyük işlemler"""
        symbol = item['symbol']
        sym_clean = symbol.replace("USDT", "")
        price = item['price']
        chg_pct = item['chg_pct']
        quote_vol = item['quote_vol']

        # Cooldown kontrolü
        with self.lock:
            last = self.spot_last_fetch.get(sym_clean, 0)
            if now - last < SPOT_COOLDOWN:
                return
            self.spot_last_fetch[sym_clean] = now

        # Son işlemleri çek (daha fazla veri = daha iyi balina tespiti)
        trades = self.fetch_spot_recent_trades(symbol, limit=100)
        if not trades:
            return

        # Balina işlemlerini analiz et
        whale_trades = []
        large_trades = []
        buy_vol = 0.0
        sell_vol = 0.0
        total_usd = 0.0
        last_trade_id = None

        for t in trades:
            try:
                qty = float(t['qty'])
                trade_price = float(t['price'])
                usd_val = qty * trade_price
                is_buyer_maker = t.get('isBuyerMaker', False)
                # isBuyerMaker=True → satıcı agresif (SELL), False → alıcı agresif (BUY)
                side = "SELL" if is_buyer_maker else "BUY"
                trade_id = t.get('id', 0)

                if last_trade_id is None or trade_id > last_trade_id:
                    last_trade_id = trade_id

                if side == "BUY":
                    buy_vol += usd_val
                else:
                    sell_vol += usd_val
                total_usd += usd_val

                if usd_val >= SPOT_WHALE_TRADE_USD:
                    whale_trades.append({
                        'usd': usd_val,
                        'qty': qty,
                        'price': trade_price,
                        'side': side,
                        'time': t.get('time', 0),
                    })
                elif usd_val >= SPOT_LARGE_TRADE_USD:
                    large_trades.append({
                        'usd': usd_val,
                        'qty': qty,
                        'price': trade_price,
                        'side': side,
                        'time': t.get('time', 0),
                    })
            except:
                continue

        # Sadece balina varsa veya çok büyük hacim varsa sinyal üret
        has_whale = len(whale_trades) > 0
        has_large = len(large_trades) >= 3

        if not has_whale and not has_large and abs(chg_pct) < 5.0:
            return

        # Alım/satım baskısı
        total_vol = buy_vol + sell_vol
        buy_ratio = buy_vol / total_vol if total_vol > 0 else 0.5
        sell_ratio = 1 - buy_ratio

        # Dominant side
        if buy_ratio >= 0.65:
            dominant_side = "BUY"
        elif sell_ratio >= 0.65:
            dominant_side = "SELL"
        else:
            dominant_side = "NEUTRAL"

        # En büyük balina işlemi
        top_whale = max(whale_trades, key=lambda x: x['usd']) if whale_trades else None
        top_large = max(large_trades, key=lambda x: x['usd']) if large_trades else None
        max_trade_usd = top_whale['usd'] if top_whale else (top_large['usd'] if top_large else 0)

        # Sinyal tipi belirle - sadece balina odaklı
        is_mega = any(t['usd'] >= SPOT_MEGA_WHALE_USD for t in whale_trades)

        if is_mega:
            if dominant_side == "BUY":
                signal_type = "🐋 MEGA WHALE BUY"
                signal_color = "mega-whale-buy"
            elif dominant_side == "SELL":
                signal_type = "🐋 MEGA WHALE SELL"
                signal_color = "mega-whale-sell"
            else:
                signal_type = "🐋 MEGA WHALE"
                signal_color = "mega-whale-buy" if chg_pct > 0 else "mega-whale-sell"
        elif has_whale:
            if dominant_side == "BUY":
                signal_type = "🐋 WHALE BUY"
                signal_color = "whale-buy"
            elif dominant_side == "SELL":
                signal_type = "🐋 WHALE SELL"
                signal_color = "whale-sell"
            else:
                signal_type = "🐋 WHALE"
                signal_color = "whale-buy" if chg_pct > 0 else "whale-sell"
        elif has_large and abs(chg_pct) >= 3.0:
            if dominant_side == "BUY":
                signal_type = "💰 LARGE BUY"
                signal_color = "large-buy"
            elif dominant_side == "SELL":
                signal_type = "💰 LARGE SELL"
                signal_color = "large-sell"
            else:
                signal_type = "💰 LARGE"
                signal_color = "large-buy" if chg_pct > 0 else "large-sell"
        else:
            return

        t_str = datetime.now().strftime("%H:%M:%S")

        with self.lock:
            # Duplikat kontrol
            for s in self.spot_signals[:5]:
                if s.get('Symbol') == sym_clean and s.get('SignalType') == signal_type:
                    return

            self.spot_signals.insert(0, {
                "Time": t_str,
                "Symbol": sym_clean,
                "Price": f"{price:.6f}" if price < 0.01 else (f"{price:.4f}" if price < 1 else f"{price:.2f}"),
                "Chg": chg_pct,
                "QuoteVol": quote_vol,
                "BuyRatio": round(buy_ratio * 100, 1),
                "SellRatio": round(sell_ratio * 100, 1),
                "WhaleCount": len(whale_trades),
                "LargeCount": len(large_trades),
                "MaxTradeUSD": max_trade_usd,
                "SignalType": signal_type,
                "SignalColor": signal_color,
                "HasWhale": has_whale,
                "HasMega": is_mega,
                "DominantSide": dominant_side,
                "TotalTradesAnalyzed": len(trades),
            })
            self.log(f"💱 SPOT WHALE: {sym_clean} {signal_type} | "
                      f"Max: ${max_trade_usd:,.0f} | "
                      f"Whale: {len(whale_trades)} | "
                      f"Buy: {buy_ratio*100:.0f}%")
            if len(self.spot_signals) > MAX_SPOT_SIGNALS:
                self.spot_signals.pop()

    # ================================================================
    # MACD PARALEL YUKSELIS
    # ================================================================

    def _maybe_trigger_macd(self, symbol, price, now):
        hist = list(self.history.get(symbol, []))
        if len(hist) < 6:
            return
        last_t = self.macd_last_trigger.get(symbol, 0)
        if now - last_t < 20:
            return
        past_1m = next((x for x in reversed(hist) if now - x[0] >= 60), hist[0])
        p_chg_1m = abs(((price - past_1m[1]) / past_1m[1]) * 100)
        if p_chg_1m >= 0.25:
            self.macd_last_trigger[symbol] = now
            MACD_EXECUTOR.submit(self._run_macd_analysis, symbol, price)

    def _fetch_klines(self, symbol, interval, limit, is_spot=False):
        if is_spot:
            url = f"{self.spot_base_url}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        else:
            url = f"{self.rest_base_url}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code != 200:
                return None
            raw = resp.json()
            closes = [float(c[4]) for c in raw]
            return pd.Series(closes, name='close')
        except:
            return None

    def _fetch_klines_ohlc(self, symbol, interval, limit, is_spot=False):
        if is_spot:
            url = f"{self.spot_base_url}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        else:
            url = f"{self.rest_base_url}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code != 200:
                return None
            raw = resp.json()
            data = {
                'open': [float(c[1]) for c in raw],
                'high': [float(c[2]) for c in raw],
                'low': [float(c[3]) for c in raw],
                'close': [float(c[4]) for c in raw],
                'volume': [float(c[5]) for c in raw],
            }
            return pd.DataFrame(data)
        except:
            return None

    def _analyze_macd_window(self, closes: pd.Series) -> int:
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

            positive_zone = m_curr > 0 and s_curr > 0
            bullish = m_curr > s_curr
            rising = m_curr > m_prev and s_curr > s_prev
            histogram = (m_curr - s_curr) >= (m_prev - s_curr) * 0.95
            m_slope = m_curr - m_prev
            s_slope = s_curr - s_prev
            parallel = False
            if s_slope > 0:
                ratio = m_slope / s_slope
                parallel = 0.4 <= ratio <= 2.5

            if positive_zone and bullish and rising and histogram and parallel:
                count += 1
            else:
                break
        return count

    def _run_macd_analysis(self, symbol, price):
        closes = self._fetch_klines(symbol, "15m", 60)
        if closes is None:
            return

        macd_count = self._analyze_macd_window(closes)
        sym_clean = symbol.replace("USDT", "")

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

        if not (MACD_MIN_CANDLES <= macd_count <= MACD_MAX_CANDLES):
            return

        now = time.time()
        alert_key = f"{sym_clean}_{macd_count}"

        with self.lock:
            if alert_key in self.macd_sent_keys:
                return
            if now - self.macd_sent.get(sym_clean, 0) < MACD_COOLDOWN:
                return
            self.macd_sent_keys[alert_key] = now
            self.macd_sent[sym_clean] = now

        macd_tag = f"Paralel({macd_count})"
        updated = False
        with self.lock:
            for sig in self.signals[:15]:
                if sig.get('Symbol') == sym_clean and sig.get('MACD') == "":
                    sig['MACD'] = macd_tag
                    updated = True
                    break

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

    # ================================================================
    # BIG MOVE HUNTER (LONG)
    # ================================================================

    def _maybe_trigger_bigmove(self, symbol, price, now):
        hist = list(self.history.get(symbol, []))
        if len(hist) < 6:
            return
        last_t = self.bigmove_last_trigger.get(symbol, 0)
        if now - last_t < 45:
            return
        past_1m = next((x for x in reversed(hist) if now - x[0] >= 60), hist[0])
        p_chg_1m = abs(((price - past_1m[1]) / past_1m[1]) * 100)
        if p_chg_1m >= 0.40:
            self.bigmove_last_trigger[symbol] = now
            BIGMOVE_EXECUTOR.submit(self._run_bigmove_analysis, symbol, price)

    def _bollinger_squeeze_score(self, df: pd.DataFrame) -> tuple:
        if len(df) < 50:
            return False, 0, 0.0

        df['sma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['upper'] = df['sma20'] + (df['std20'] * 2)
        df['lower'] = df['sma20'] - (df['std20'] * 2)
        df['bandwidth'] = (df['upper'] - df['lower']) / df['sma20'] * 100

        recent_bw = df['bandwidth'].iloc[-BB_SQUEEZE_LOOKBACK:].dropna()
        if len(recent_bw) < 20:
            return False, 0, 0.0

        current_bw = df['bandwidth'].iloc[-1]
        percentile = (recent_bw < current_bw).mean() * 100
        is_squeeze = percentile <= BB_SQUEEZE_PERCENTILE

        squeeze_duration = 0
        for i in range(1, min(50, len(df))):
            bw = df['bandwidth'].iloc[-i]
            pctl = (recent_bw < bw).mean() * 100
            if pctl <= BB_SQUEEZE_PERCENTILE:
                squeeze_duration += 1
            else:
                break

        last = df.iloc[-1]
        bb_range = last['upper'] - last['lower']
        bb_position = (last['close'] - last['lower']) / bb_range if bb_range > 0 else 0.5

        return is_squeeze, squeeze_duration, bb_position

    def _ma200_breakout_analysis(self, df: pd.DataFrame) -> tuple:
        if len(df) < 250:
            return False, 0, 0.0, 0.0

        df['ma200'] = df['close'].rolling(window=200).mean()
        closes = df['close'].values
        ma200 = df['ma200'].values

        if closes[-1] <= ma200[-1]:
            return False, 0, ma200[-1], 0.0
        if closes[-2] >= ma200[-2]:
            return False, 0, ma200[-1], 0.0

        bars_below = 0
        for i in range(2, min(len(closes), 300)):
            if not np.isnan(ma200[-i]) and closes[-i] < ma200[-i]:
                bars_below += 1
            else:
                break

        if bars_below < MA200_MIN_BARS_BELOW:
            return False, bars_below, ma200[-1], 0.0

        distance_pct = ((closes[-1] - ma200[-1]) / ma200[-1]) * 100
        return True, bars_below, ma200[-1], distance_pct

    def _macd_resistance_break(self, closes: pd.Series) -> tuple:
        if len(closes) < 60:
            return False, 0.0, 0.0, 0.0, 0.0

        exp1 = closes.ewm(span=12, adjust=False).mean()
        exp2 = closes.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        m_now = macd_line.iloc[-1]
        s_now = signal_line.iloc[-1]
        h_now = histogram.iloc[-1]
        m_prev = macd_line.iloc[-2]
        s_prev = signal_line.iloc[-2]
        h_prev = histogram.iloc[-2]

        recent_hist = histogram.iloc[-MACD_RESISTANCE_LOOKBACK:-1]
        if len(recent_hist) < 5:
            return False, 0.0, m_now, s_now, h_now

        resistance_level = recent_hist.max()

        hist_turning_positive = h_prev <= 0 and h_now > 0
        hist_breaking_resistance = h_now > resistance_level and resistance_level > 0
        macd_above_signal = m_now > s_now
        macd_rising = m_now > m_prev

        is_break = (hist_turning_positive or hist_breaking_resistance) and macd_above_signal and macd_rising

        if not is_break and h_prev > 0 and h_now > h_prev and h_now > resistance_level and m_now > 0 and m_now > s_now:
            is_break = True

        return is_break, resistance_level, m_now, s_now, h_now

    def _run_bigmove_analysis(self, symbol, price):
        sym_clean = symbol.replace("USDT", "")
        now = time.time()

        with self.lock:
            if now - self.bigmove_sent.get(sym_clean, 0) < BIGMOVE_COOLDOWN:
                return

        df_4h = self._fetch_klines_ohlc(symbol, "4h", 300)
        closes_1h = self._fetch_klines(symbol, "1h", 100)

        if df_4h is None or closes_1h is None:
            return

        squeeze, squeeze_dur, bb_pos = self._bollinger_squeeze_score(df_4h)
        ma200_break, bars_below, ma200_val, dist_pct = self._ma200_breakout_analysis(df_4h)
        macd_break, res_level, m_now, s_now, h_now = self._macd_resistance_break(closes_1h)

        conditions_met = []
        total_score = 0

        if squeeze:
            conditions_met.append(f"BB-Squeeze({squeeze_dur})")
            total_score += min(20 + squeeze_dur * 2, 40)

        if ma200_break:
            conditions_met.append(f"MA200-Break({bars_below})")
            total_score += 35
            if bars_below > 50:
                total_score += 10

        if macd_break:
            conditions_met.append("MACD-1H-Break")
            total_score += 25

        if not conditions_met:
            return
        if len(conditions_met) < 2 and not ma200_break:
            return
        if total_score < 50:
            return

        with self.lock:
            self.bigmove_sent[sym_clean] = now

        t_str = datetime.now().strftime("%H:%M:%S")

        if squeeze and not ma200_break and not macd_break:
            with self.lock:
                self.bigmove_candidates[sym_clean] = {
                    "Time": t_str,
                    "Symbol": sym_clean,
                    "Price": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                    "Score": total_score,
                    "Conditions": ", ".join(conditions_met),
                    "Status": "Squeeze Tespiti",
                    "BB_Pos": f"{bb_pos:.2f}",
                }
            return

        with self.lock:
            for s in self.bigmove_signals[:5]:
                if s.get('Symbol') == sym_clean:
                    return

            self.bigmove_signals.insert(0, {
                "Time": t_str,
                "Symbol": sym_clean,
                "Price": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                "Score": total_score,
                "Conditions": ", ".join(conditions_met),
                "Status": "BIG MOVE",
                "BB_Pos": f"{bb_pos:.2f}",
                "MA200_Dist": f"{dist_pct:.2f}%" if ma200_break else "—",
                "MACD_1H": f"H:{h_now:.3f}" if macd_break else "—",
            })
            if len(self.bigmove_signals) > MAX_DISPLAY_ROWS:
                self.bigmove_signals.pop()

    # ================================================================
    # SHORT BIG MOVE HUNTER
    # ================================================================

    def _maybe_trigger_shortmove(self, symbol, price, now):
        hist = list(self.history.get(symbol, []))
        if len(hist) < 6:
            return
        last_t = self.shortmove_last_trigger.get(symbol, 0)
        if now - last_t < 45:
            return
        past_1m = next((x for x in reversed(hist) if now - x[0] >= 60), hist[0])
        p_chg_1m = ((price - past_1m[1]) / past_1m[1]) * 100
        if p_chg_1m <= -0.35:
            self.shortmove_last_trigger[symbol] = now
            SHORT_EXECUTOR.submit(self._run_shortmove_analysis, symbol, price)

    def _ma200_breakdown_analysis(self, df: pd.DataFrame) -> tuple:
        if len(df) < 250:
            return False, 0, 0.0, 0.0

        df['ma200'] = df['close'].rolling(window=200).mean()
        closes = df['close'].values
        ma200 = df['ma200'].values

        if closes[-1] >= ma200[-1]:
            return False, 0, ma200[-1], 0.0
        if closes[-2] <= ma200[-2]:
            return False, 0, ma200[-1], 0.0

        bars_above = 0
        for i in range(2, min(len(closes), 300)):
            if not np.isnan(ma200[-i]) and closes[-i] > ma200[-i]:
                bars_above += 1
            else:
                break

        if bars_above < MA200_MIN_BARS_ABOVE:
            return False, bars_above, ma200[-1], 0.0

        distance_pct = ((closes[-1] - ma200[-1]) / ma200[-1]) * 100
        return True, bars_above, ma200[-1], distance_pct

    def _death_cross_analysis(self, df: pd.DataFrame) -> tuple:
        if len(df) < 210:
            return False, 0.0, 0.0

        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()

        ma50 = df['ma50'].values
        ma200 = df['ma200'].values

        if np.isnan(ma50[-1]) or np.isnan(ma200[-1]):
            return False, 0.0, 0.0

        if ma50[-1] >= ma200[-1]:
            return False, ma50[-1], ma200[-1]
        if ma50[-2] <= ma200[-2]:
            return False, ma50[-1], ma200[-1]

        return True, ma50[-1], ma200[-1]

    def _rsi_overbought_bearish_div(self, closes: pd.Series) -> tuple:
        if len(closes) < RSI_BEARISH_DIV_LOOKBACK + 5:
            return False, 0.0, False, False

        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        rsi_now = rsi.iloc[-1]
        is_overbought = rsi_now > RSI_OVERBOUGHT

        lookback = RSI_BEARISH_DIV_LOOKBACK
        price_high_recent = closes.iloc[-1]
        price_high_prev = closes.iloc[-lookback:-1].max()
        rsi_at_recent = rsi.iloc[-1]
        rsi_at_prev_high_idx = closes.iloc[-lookback:-1].idxmax()
        rsi_at_prev_high = rsi.loc[rsi_at_prev_high_idx] if rsi_at_prev_high_idx in rsi.index else rsi.iloc[-lookback]

        is_bearish_div = (price_high_recent > price_high_prev) and (rsi_at_recent < rsi_at_prev_high - 2)

        is_signal = is_overbought or is_bearish_div
        return is_signal, round(float(rsi_now), 1), is_overbought, is_bearish_div

    def _bb_upper_rejection(self, df: pd.DataFrame) -> tuple:
        if len(df) < 30:
            return False, 0.0

        df['sma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['upper'] = df['sma20'] + (df['std20'] * 2)
        df['lower'] = df['sma20'] - (df['std20'] * 2)
        df['bandwidth'] = (df['upper'] - df['lower']) / df['sma20'] * 100

        recent_bw = df['bandwidth'].iloc[-50:].dropna()
        if len(recent_bw) < 10:
            return False, 0.0

        last_close = df['close'].iloc[-1]
        last_open = df['open'].iloc[-1]
        last_upper = df['upper'].iloc[-1]

        bb_range = df['upper'].iloc[-1] - df['lower'].iloc[-1]
        bb_pos = (last_close - df['lower'].iloc[-1]) / bb_range if bb_range > 0 else 0.5

        touched_upper = False
        for i in range(1, BB_UPPER_REJECTION_BARS + 1):
            if len(df) >= i:
                if df['high'].iloc[-i] >= df['upper'].iloc[-i] * 0.99:
                    touched_upper = True
                    break

        bearish_candle = last_close < last_open
        below_upper = last_close < last_upper

        is_rejection = touched_upper and bearish_candle and below_upper and bb_pos > 0.7

        return is_rejection, round(bb_pos, 2)

    def _macd_bearish_crossover(self, closes: pd.Series) -> tuple:
        if len(closes) < 40:
            return False, 0.0, 0.0, 0.0

        exp1 = closes.ewm(span=12, adjust=False).mean()
        exp2 = closes.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        m_now = macd_line.iloc[-1]
        s_now = signal_line.iloc[-1]
        h_now = histogram.iloc[-1]
        m_prev = macd_line.iloc[-2]
        s_prev = signal_line.iloc[-2]
        h_prev = histogram.iloc[-2]

        fresh_bearish_cross = (m_prev >= s_prev) and (m_now < s_now)
        hist_turning_negative = h_prev >= 0 and h_now < 0
        hist_expanding_negative = h_now < 0 and h_now < h_prev and m_now < 0

        is_bearish = fresh_bearish_cross or hist_turning_negative or hist_expanding_negative

        return is_bearish, round(float(m_now), 4), round(float(s_now), 4), round(float(h_now), 4)

    def _run_shortmove_analysis(self, symbol, price):
        sym_clean = symbol.replace("USDT", "")
        now = time.time()

        with self.lock:
            if now - self.shortmove_sent.get(sym_clean, 0) < SHORT_COOLDOWN:
                return

        df_4h = self._fetch_klines_ohlc(symbol, "4h", 300)
        df_4h_short = self._fetch_klines_ohlc(symbol, "4h", 60)
        closes_1h = self._fetch_klines(symbol, "1h", 100)

        if df_4h is None or closes_1h is None:
            return

        ma200_break, bars_above, ma200_val, dist_pct = self._ma200_breakdown_analysis(df_4h)
        death_cross, ma50_val, ma200_dc_val = self._death_cross_analysis(df_4h)
        rsi_signal, rsi_now, is_overbought, is_bearish_div = self._rsi_overbought_bearish_div(closes_1h)
        bb_rejection, bb_pos = self._bb_upper_rejection(df_4h_short if df_4h_short is not None else df_4h)
        macd_bearish, macd_m, macd_s, macd_h = self._macd_bearish_crossover(closes_1h)

        conditions_met = []
        total_score = 0

        if ma200_break:
            conditions_met.append(f"MA200-Down({bars_above})")
            total_score += 35
            if bars_above > 50:
                total_score += 10

        if death_cross:
            conditions_met.append("Death-Cross")
            total_score += 30

        if rsi_signal:
            if is_overbought and is_bearish_div:
                conditions_met.append(f"RSI-OB+Div({rsi_now})")
                total_score += 30
            elif is_overbought:
                conditions_met.append(f"RSI-OB({rsi_now})")
                total_score += 15
            elif is_bearish_div:
                conditions_met.append(f"RSI-BearDiv({rsi_now})")
                total_score += 20

        if bb_rejection:
            conditions_met.append(f"BB-Reject({bb_pos})")
            total_score += 20

        if macd_bearish:
            conditions_met.append("MACD-Bear")
            total_score += 25

        if not conditions_met:
            return
        if len(conditions_met) < 2 and not ma200_break:
            return
        if total_score < 50:
            return

        t_str = datetime.now().strftime("%H:%M:%S")

        if bb_rejection and not ma200_break and not death_cross and len(conditions_met) == 1:
            with self.lock:
                self.shortmove_candidates[sym_clean] = {
                    "Time": t_str,
                    "Symbol": sym_clean,
                    "Price": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                    "Score": total_score,
                    "Conditions": ", ".join(conditions_met),
                    "Status": "Short Radar",
                    "BB_Pos": f"{bb_pos:.2f}",
                    "RSI": str(rsi_now),
                }
            return

        with self.lock:
            if now - self.shortmove_sent.get(sym_clean, 0) < SHORT_COOLDOWN:
                return
            self.shortmove_sent[sym_clean] = now

            for s in self.shortmove_signals[:5]:
                if s.get('Symbol') == sym_clean:
                    return

            self.shortmove_signals.insert(0, {
                "Time": t_str,
                "Symbol": sym_clean,
                "Price": f"{price:.4f}" if price < 1 else f"{price:.2f}",
                "Score": total_score,
                "Conditions": ", ".join(conditions_met),
                "Status": "SHORT MOVE",
                "BB_Pos": f"{bb_pos:.2f}",
                "MA200_Dist": f"{dist_pct:.2f}%" if ma200_break else "—",
                "RSI": str(rsi_now),
                "MACD_1H": f"H:{macd_h:.3f}" if macd_bearish else "—",
            })
            if len(self.shortmove_signals) > MAX_DISPLAY_ROWS:
                self.shortmove_signals.pop()


# ==================== WORKERS ====================
@st.cache_resource
def get_radar_instance():
    return MarketRadar()


def binance_worker(radar_obj):
    radar_obj.log(">>> FUTURES WORKER BASLADI")
    radar_obj.get_working_rest_url()
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
                        f"✅ Futures Fetch #{fetch_count} | Pairs: {radar_obj.total_pairs} | "
                        f"Signals: {len(radar_obj.signals)}"
                    )
            else:
                radar_obj.log(f"⚠️ Futures HTTP {r.status_code}")
        except Exception as e:
            radar_obj.log(f"❌ FUTURES WORKER HATA: {e}")
        time.sleep(FETCH_INTERVAL)


def spot_worker(radar_obj):
    radar_obj.log(">>> SPOT WORKER BASLADI")
    radar_obj.get_working_spot_url()
    fetch_count = 0
    while True:
        try:
            data = radar_obj.fetch_spot_ticker()
            if data:
                fetch_count += 1
                radar_obj.process_spot_ticker(data)
                if fetch_count % 5 == 0:
                    radar_obj.log(
                        f"💱 Spot Fetch #{fetch_count} | Pairs: {radar_obj.spot_total_pairs} | "
                        f"Spot Signals: {len(radar_obj.spot_signals)} | "
                        f"URL: {radar_obj.spot_base_url}"
                    )
            else:
                radar_obj.log("⚠️ Spot veri alınamadı - URL yeniden kontrol ediliyor...")
                radar_obj.get_working_spot_url()
        except Exception as e:
            radar_obj.log(f"❌ SPOT WORKER HATA: {e}")
            time.sleep(2)
            radar_obj.get_working_spot_url()
        time.sleep(SPOT_FETCH_INTERVAL)


# ==================== STREAMLIT UI ====================
st.set_page_config(layout="wide", page_title="Market Radar Pro")

st.markdown("""
<style>
.main { background-color: #0e1117; }

/* Status */
.status-live { color: #00ff88; font-weight: bold; border: 1px solid #00ff88; padding: 2px 10px; border-radius: 15px; font-size: 0.8rem; }
.status-offline { color: #ff4b4b; font-weight: bold; border: 1px solid #ff4b4b; padding: 2px 10px; border-radius: 15px; font-size: 0.8rem; }

/* Sayfa seçici */
.page-nav-container {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
    padding: 10px 0;
    border-bottom: 2px solid #222;
    flex-wrap: wrap;
}
.page-btn {
    padding: 8px 20px;
    border-radius: 8px;
    border: 1px solid #444;
    background: #1e2127;
    color: #aaa;
    cursor: pointer;
    font-size: 0.95rem;
    font-weight: bold;
    text-decoration: none;
    transition: all 0.2s;
}
.page-btn:hover { background: #2a2d35; color: #fff; border-color: #666; }
.page-btn.active-futures { background: #1a3a1a; color: #00ff88; border-color: #00ff88; }
.page-btn.active-longbig { background: #2a1d08; color: #f5b041; border-color: #f39c12; }
.page-btn.active-shortbig { background: #2a0808; color: #ff7675; border-color: #c0392b; }
.page-btn.active-spot { background: #0a1a2a; color: #74b9ff; border-color: #0984e3; }

/* Labels */
.pump-label  { background-color: #00ff88; color: black;  padding: 2px 8px; border-radius: 4px; font-weight: bold; }
.dump-label  { background-color: #ff4b4b; color: white;  padding: 2px 8px; border-radius: 4px; font-weight: bold; }
.buy-label   { background-color: #1a7f4b; color: #afffcf; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
.sell-label  { background-color: #7f1a1a; color: #ffcfcf; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
.mode-confirmed { background-color: #1abc9c; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
.mode-flash { background-color: #e67e22; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
.mode-macd  { background-color: #8e44ad; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
.mode-bigmove { background-color: #f39c12; color: #000; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
.mode-shortmove { background-color: #c0392b; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
.macd-tag   { background-color: #2c1654; color: #c39bd3; padding: 2px 7px; border-radius: 4px; font-size: 0.78rem; font-weight: bold; border: 1px solid #8e44ad; }
.bigmove-tag { background-color: #3d2208; color: #f5b041; padding: 2px 7px; border-radius: 4px; font-size: 0.78rem; font-weight: bold; border: 1px solid #f39c12; }
.shortmove-tag { background-color: #3d0808; color: #ff7675; padding: 2px 7px; border-radius: 4px; font-size: 0.78rem; font-weight: bold; border: 1px solid #c0392b; }

/* Spot signal labels - SADECE BALINA */
.spot-tag-mega-whale-buy   { background: #004d26; color: #00ff88; padding: 3px 10px; border-radius: 4px; font-weight: bold; border: 2px solid #00ff88; font-size: 0.9rem; }
.spot-tag-mega-whale-sell  { background: #4d0000; color: #ff7675; padding: 3px 10px; border-radius: 4px; font-weight: bold; border: 2px solid #ff4b4b; font-size: 0.9rem; }
.spot-tag-whale-buy   { background: #003d1f; color: #00ff88; padding: 3px 10px; border-radius: 4px; font-weight: bold; border: 1px solid #00b894; font-size: 0.88rem; }
.spot-tag-whale-sell  { background: #3d0000; color: #ff7675; padding: 3px 10px; border-radius: 4px; font-weight: bold; border: 1px solid #d63031; font-size: 0.88rem; }
.spot-tag-large-buy   { background: #0a1a2a; color: #74b9ff; padding: 3px 10px; border-radius: 4px; font-weight: bold; border: 1px solid #0984e3; font-size: 0.88rem; }
.spot-tag-large-sell  { background: #2a1a0a; color: #fdcb6e; padding: 3px 10px; border-radius: 4px; font-weight: bold; border: 1px solid #e17055; font-size: 0.88rem; }

/* Stat cards */
.stat-card { background-color: #1e2127; padding: 10px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #f1c40f; }
.debug-box { background-color: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 10px; font-family: monospace; font-size: 0.75rem; color: #aaa; max-height: 200px; overflow-y: auto; }

/* Tables */
table { width: 100%; border-collapse: collapse; }
th, td { white-space: nowrap; padding: 12px 15px; text-align: left; border-bottom: 1px solid #222; }
.sym-link { color: #f1c40f; text-decoration: none; font-weight: bold; font-size: 1.05rem; }
.sym-link:hover { color: #fff; }
.sym-link-spot { color: #74b9ff; text-decoration: none; font-weight: bold; font-size: 1.05rem; }
.sym-link-spot:hover { color: #fff; }
.green-arrow { color: #00ff88; font-weight: bold; }
.red-arrow   { color: #ff4b4b; font-weight: bold; }
.row-flash-pump { background-color: rgba(0, 255, 136, 0.22) !important; border-left: 5px solid #00ff88 !important; }
.row-flash-dump { background-color: rgba(255, 75,  75,  0.22) !important; border-left: 5px solid #ff4b4b !important; }
.row-conf-pump  { background-color: rgba(0, 255, 136, 0.08) !important; }
.row-conf-dump  { background-color: rgba(255, 75,  75,  0.08) !important; }
.row-macd       { background-color: rgba(142, 68, 173, 0.12) !important; border-left: 3px solid #8e44ad !important; }
.row-bigmove    { background-color: rgba(243, 156, 18, 0.15) !important; border-left: 4px solid #f39c12 !important; }
.row-shortmove  { background-color: rgba(192, 57, 43, 0.18) !important; border-left: 4px solid #c0392b !important; }

/* Spot rows - balina odaklı */
.row-spot-mega-whale-buy  { background-color: rgba(0, 255, 136, 0.28) !important; border-left: 6px solid #00ff88 !important; }
.row-spot-mega-whale-sell { background-color: rgba(255, 75, 75, 0.30) !important; border-left: 6px solid #ff4b4b !important; }
.row-spot-whale-buy       { background-color: rgba(0, 255, 136, 0.18) !important; border-left: 5px solid #00ff88 !important; }
.row-spot-whale-sell      { background-color: rgba(255, 75, 75, 0.22) !important; border-left: 5px solid #ff4b4b !important; }
.row-spot-large-buy       { background-color: rgba(0, 184, 148, 0.10) !important; border-left: 4px solid #00b894 !important; }
.row-spot-large-sell      { background-color: rgba(214, 48, 49, 0.12) !important; border-left: 4px solid #d63031 !important; }

/* MACD radar */
.macd-radar-card { background: #1a1030; border: 1px solid #8e44ad; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; }
.macd-radar-sym { color: #c39bd3; font-weight: bold; font-size: 1rem; }
.macd-radar-tag { color: #f0c3ff; font-size: 0.82rem; }
.macd-radar-time { color: #666; font-size: 0.72rem; }

/* BigMove radar */
.bigmove-card { background: #2a1d0a; border: 1px solid #f39c12; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }
.bigmove-sym { color: #f5b041; font-weight: bold; font-size: 1.1rem; }
.bigmove-cond { color: #f8c471; font-size: 0.85rem; }
.bigmove-score { color: #fff; font-weight: bold; font-size: 0.9rem; }
.bigmove-time { color: #888; font-size: 0.72rem; }
.bigmove-radar-card { background: #1a1508; border: 1px solid #7f8c8d; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; }
.bigmove-radar-sym { color: #d5dbdb; font-weight: bold; font-size: 1rem; }
.bigmove-radar-cond { color: #aab7b8; font-size: 0.82rem; }

/* ShortMove radar */
.shortmove-card { background: #1f0a0a; border: 1px solid #c0392b; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }
.shortmove-sym { color: #ff7675; font-weight: bold; font-size: 1.1rem; }
.shortmove-cond { color: #fab1a0; font-size: 0.85rem; }
.shortmove-radar-card { background: #1a0a08; border: 1px solid #7f3030; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; }
.shortmove-radar-sym { color: #e17055; font-weight: bold; font-size: 1rem; }
.shortmove-radar-cond { color: #d63031; font-size: 0.82rem; }

/* Spot stat card */
.spot-whale-badge { background: #003d1f; color: #00ff88; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: bold; border: 1px solid #00b894; }
.spot-mega-badge { background: #004d26; color: #00ff88; padding: 2px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold; border: 2px solid #00ff88; }

/* Progress bar */
.pressure-bar-container { display: flex; height: 8px; border-radius: 4px; overflow: hidden; width: 100%; min-width: 80px; }
.pressure-buy  { background: #00b894; height: 100%; }
.pressure-sell { background: #d63031; height: 100%; }
</style>
""", unsafe_allow_html=True)

radar = get_radar_instance()

# Thread yönetimi
if "futures_thread_started" not in st.session_state:
    t = threading.Thread(target=binance_worker, args=(radar,), daemon=True)
    t.start()
    st.session_state.futures_thread_started = True
    radar.log(">>> UI: Futures thread baslatildi")

if "spot_thread_started" not in st.session_state:
    t2 = threading.Thread(target=spot_worker, args=(radar,), daemon=True)
    t2.start()
    st.session_state.spot_thread_started = True
    radar.log(">>> UI: Spot thread baslatildi")

# ==================== SAYFA NAVİGASYONU ====================
params = st.query_params
current_page = params.get("page", "futures")

pages = {
    "futures":  ("📡 Futures Sinyaller", "active-futures"),
    "longbig":  ("🚀 Long Big Move",     "active-longbig"),
    "shortbig": ("🔻 Short Big Move",    "active-shortbig"),
    "spot":     ("🐋 Spot Whale Tracker",   "active-spot"),
}

nav_cols = st.columns(len(pages))
for i, (key, (label, active_cls)) in enumerate(pages.items()):
    is_active = key == current_page
    border_colors = {
        "futures": "#00ff88", "longbig": "#f39c12",
        "shortbig": "#c0392b", "spot": "#0984e3",
    }
    bg_colors = {
        "futures": "#1a3a1a", "longbig": "#2a1d08",
        "shortbig": "#2a0808", "spot": "#0a1a2a",
    }
    text_colors = {
        "futures": "#00ff88", "longbig": "#f5b041",
        "shortbig": "#ff7675", "spot": "#74b9ff",
    }
    if is_active:
        style = (
            f"background:{bg_colors[key]}; color:{text_colors[key]}; "
            f"border:2px solid {border_colors[key]}; border-radius:8px; "
            f"padding:8px 0; font-weight:bold; font-size:0.95rem; width:100%; cursor:default;"
        )
    else:
        style = (
            "background:#1e2127; color:#aaa; border:1px solid #444; border-radius:8px; "
            "padding:8px 0; font-weight:bold; font-size:0.95rem; width:100%; cursor:pointer;"
        )
    with nav_cols[i]:
        if nav_cols[i].button(label, key=f"nav_{key}", use_container_width=True, disabled=is_active):
            st.query_params["page"] = key
            st.rerun()

# ==================== ORTAK HEADER ====================
h1, h2, h3, h4, h5, h6 = st.columns([2, 1, 1, 1, 1, 1])
h1.title("📡 Market Radar Pro")

f_elapsed = time.time() - radar.last_heartbeat
s_elapsed = time.time() - radar.spot_last_heartbeat
futures_ok = f_elapsed < 10
spot_ok = s_elapsed < 15

status_html = ""
if futures_ok:
    status_html += '<span class="status-live">● FUTURES LIVE</span> '
else:
    status_html += '<span class="status-offline">● FUTURES OFF</span> '
if spot_ok:
    status_html += '<span class="status-live">● SPOT LIVE</span>'
else:
    status_html += '<span class="status-offline">● SPOT OFF</span>'

h2.markdown(f"<div style='margin-top:10px;'>{status_html}</div>", unsafe_allow_html=True)
h2.markdown('<a href="https://x.com/SinyalEngineer" target="_blank" style="color:white; text-decoration:none;">𝕏 @SinyalEngineer</a>', unsafe_allow_html=True)
h3.metric("Futures Pairs", radar.total_pairs)
h3.metric("Spot Pairs", radar.spot_total_pairs)
h4.metric("Futures Sinyaller", len(radar.signals))
h5.metric("Long Big Moves", len(radar.bigmove_signals))
h5.metric("Short Big Moves", len(radar.shortmove_signals))
h6.metric("Spot Whale Sinyaller", len(radar.spot_signals))

st.divider()

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
# SAYFA 1: FUTURES SİNYALLER
# ================================================================
if current_page == "futures":
    st.caption("⚡ Flash: Anlık hareket | 💎 Confirmed: 3dk+15dk | 📊 MACD: Paralel yükseliş (3-8 mum)")

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
    search_query = col_filters[2].text_input("🔍 Symbol Filtrele", placeholder="BTC...", key="search").upper()
    macd_only = col_filters[3].checkbox("Sadece MACD etiketli", value=False)

    st.divider()

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

    def render_futures_table(display_data, placeholder):
        with placeholder.container():
            with radar.lock:
                if display_data:
                    html = (
                        "<table><tr>"
                        "<th>Saat</th><th>Symbol (4H ↑/↓)</th><th>Fiyat</th>"
                        "<th>Momentum</th><th>15m Ref</th><th>Vol</th>"
                        "<th>Durum</th><th>Tür</th><th>MACD Pattern</th>"
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

        render_futures_table(display_data, main_placeholder)

        with macd_placeholder.container():
            with radar.lock:
                candidates = dict(radar.macd_candidates)
            if candidates:
                def _macd_sort_key(item):
                    tag = item[1].get("MACD Pattern", "Paralel(0)")
                    try: return int(tag.split("(")[1].rstrip(")"))
                    except: return 0
                sorted_c = sorted(candidates.items(), key=_macd_sort_key, reverse=True)
                for sym, info in sorted_c[:20]:
                    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
                    st.markdown(f"""<div class="macd-radar-card">
                        <a href="{tv_url}" target="_blank" class="macd-radar-sym">{sym}</a><br>
                        <span class="macd-radar-tag">{info["MACD Pattern"]}</span>
                        &nbsp;<span style="color:#aaa;font-size:0.8rem">{info["Fiyat"]}</span><br>
                        <span class="macd-radar-time">{info.get("Güncelleme", "N/A")}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.caption("MACD taranıyor...")

        time.sleep(1.5)

# ================================================================
# SAYFA 2: LONG BIG MOVE HUNTER
# ================================================================
elif current_page == "longbig":
    st.caption("🎯 Bollinger Squeeze + 4H MA200 Break + 1H MACD Resistance Break")

    st.markdown("""
    <div style="background-color:#1a1508; border-left:4px solid #f39c12; padding:12px 16px; border-radius:4px; margin-bottom:16px;">
        <b style="color:#f5b041;">Long Big Move Hunter Nasıl Çalışır?</b><br>
        <span style="color:#d5dbdb; font-size:0.9rem;">
        1. <b>BB Squeeze:</b> 4H Bollinger Bantları geçmiş 100 mumun en dar %5'lik diliminde mi?<br>
        2. <b>MA200 Break:</b> 4H fiyat 20+ mum (5 gün) MA200 altında kaldıktan sonra üzerine atıyor mu?<br>
        3. <b>MACD 1H Resistance:</b> 1H MACD histogram negatiften pozitife geçiyor veya önceki direnci kırıyor mu?<br>
        En az 2 koşulun birleşmesi + skor ≥ 50 gerekir. MA200 tek başına yeterlidir.
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f3 = st.columns([1, 1])
    bm_search = col_f1.text_input("🔍 Symbol Ara", placeholder="BTC...", key="bm_search").upper()
    min_score = 70  # Sabit minimum skor
    show_radar_only = col_f3.checkbox("Sadece Squeeze Radar", value=False)

    st.divider()

    col_bm_main, col_bm_radar = st.columns([3, 1])
    with col_bm_main:
        st.subheader("🚀 Long Big Move Sinyalleri")
        bm_main_placeholder = st.empty()
    with col_bm_radar:
        st.subheader("🔍 Squeeze Radar")
        bm_radar_placeholder = st.empty()

    def render_bigmove_table(data, placeholder):
        with placeholder.container():
            with radar.lock:
                if data:
                    html = (
                        "<table><tr>"
                        "<th>Saat</th><th>Symbol</th><th>Fiyat</th>"
                        "<th>Skor</th><th>Koşullar</th><th>MA200 Mesafe</th><th>MACD 1H</th>"
                        "</tr>"
                    )
                    for row in data:
                        sym = row['Symbol']
                        tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
                        score_color = "#00ff88" if row['Score'] >= 70 else "#f39c12" if row['Score'] >= 50 else "#e74c3c"
                        html += (
                            f"<tr class='row-bigmove'>"
                            f"<td>{row['Time']}</td>"
                            f"<td><a href='{tv_url}' target='_blank' class='sym-link'>{sym}</a></td>"
                            f"<td>{row['Price']}</td>"
                            f"<td style='color:{score_color}; font-weight:bold; font-size:1.1rem;'>{row['Score']}</td>"
                            f"<td><span class='bigmove-tag'>{row['Conditions']}</span></td>"
                            f"<td>{row['MA200_Dist']}</td>"
                            f"<td>{row['MACD_1H']}</td>"
                            f"</tr>"
                        )
                    html += "</table>"
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.info("Long Big Move sinyali bekleniyor... Piyasa taranıyor 🔭")

    while True:
        with radar.lock:
            bm_signals = list(radar.bigmove_signals)
            bm_candidates = dict(radar.bigmove_candidates)

        display_bm = bm_signals
        if bm_search:
            display_bm = [s for s in display_bm if bm_search in s['Symbol']]
        display_bm = [s for s in display_bm if s['Score'] >= min_score]
        if show_radar_only:
            display_bm = []

        render_bigmove_table(display_bm, bm_main_placeholder)

        with bm_radar_placeholder.container():
            radar_items = [(sym, info) for sym, info in bm_candidates.items()
                           if not bm_search or bm_search in sym]
            radar_items.sort(key=lambda x: x[1].get('Score', 0), reverse=True)
            if radar_items:
                for sym, info in radar_items[:15]:
                    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
                    st.markdown(f"""<div class="bigmove-radar-card">
                        <a href="{tv_url}" target="_blank" class="bigmove-radar-sym">{sym}</a><br>
                        <span class="bigmove-radar-cond">{info["Conditions"]}</span><br>
                        <span style="color:#888;font-size:0.8rem;">Skor: <b>{info["Score"]}</b> | BB-Poz: {info["BB_Pos"]} | {info["Time"]}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.caption("Squeeze taranıyor...")

        time.sleep(2)

# ================================================================
# SAYFA 3: SHORT BIG MOVE HUNTER
# ================================================================
elif current_page == "shortbig":
    st.caption("🔻 MA200 Breakdown + Death Cross + RSI Bearish Div + BB Upper Rejection + MACD Bear")

    st.markdown("""
    <div style="background-color:#1f0a0a; border-left:4px solid #c0392b; padding:12px 16px; border-radius:4px; margin-bottom:16px;">
        <b style="color:#ff7675;">Short Big Move Hunter Nasıl Çalışır?</b><br>
        <span style="color:#d5dbdb; font-size:0.9rem;">
        1. <b>MA200 Breakdown:</b> 4H fiyat 20+ mum MA200 üstünde kaldıktan sonra altına kırılıyor mu? (35 puan)<br>
        2. <b>Death Cross:</b> 4H 50MA, 200MA'nın altına geçiyor mu? (30 puan)<br>
        3. <b>RSI Overbought + Bearish Divergence:</b> 1H RSI &gt;70 VE/VEYA fiyat yüksek ama RSI düşüyor (15–30 puan)<br>
        4. <b>BB Upper Rejection:</b> 4H Bollinger üst bantından fiyat reddediliyor (20 puan)<br>
        5. <b>MACD 1H Bearish:</b> 1H MACD sinyal çizgisinin altına geçiyor (25 puan)<br>
        En az 2 koşul + skor ≥ 50. MA200 Breakdown tek başına yeterlidir.
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f3 = st.columns([1, 1])
    sm_search = col_f1.text_input("🔍 Symbol Ara", placeholder="BTC...", key="sm_search").upper()
    sm_min_score = 70  # Sabit minimum skor
    sm_show_radar = col_f3.checkbox("Sadece Short Radar", value=False)

    st.divider()

    col_sm_main, col_sm_radar = st.columns([3, 1])
    with col_sm_main:
        st.subheader("🔻 Short Big Move Sinyalleri")
        sm_main_placeholder = st.empty()
    with col_sm_radar:
        st.subheader("👁 Short Radar")
        sm_radar_placeholder = st.empty()

    def render_shortmove_table(data, placeholder):
        with placeholder.container():
            if data:
                html = (
                    "<table><tr>"
                    "<th>Saat</th><th>Symbol</th><th>Fiyat</th>"
                    "<th>Skor</th><th>Koşullar</th><th>MA200 Mesafe</th><th>RSI</th><th>MACD 1H</th>"
                    "</tr>"
                )
                for row in data:
                    sym = row['Symbol']
                    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
                    score_color = "#ff4b4b" if row['Score'] >= 70 else "#e17055" if row['Score'] >= 50 else "#e74c3c"
                    html += (
                        f"<tr class='row-shortmove'>"
                        f"<td>{row['Time']}</td>"
                        f"<td><a href='{tv_url}' target='_blank' class='sym-link'>{sym}</a></td>"
                        f"<td>{row['Price']}</td>"
                        f"<td style='color:{score_color}; font-weight:bold; font-size:1.1rem;'>{row['Score']}</td>"
                        f"<td><span class='shortmove-tag'>{row['Conditions']}</span></td>"
                        f"<td style='color:#ff7675;'>{row['MA200_Dist']}</td>"
                        f"<td style='color:#fab1a0;'>{row['RSI']}</td>"
                        f"<td>{row['MACD_1H']}</td>"
                        f"</tr>"
                    )
                html += "</table>"
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("Short Big Move sinyali bekleniyor... Düşüş fırsatları taranıyor 🔍")

    while True:
        with radar.lock:
            sm_signals = list(radar.shortmove_signals)
            sm_candidates = dict(radar.shortmove_candidates)

        display_sm = [s for s in sm_signals
                      if (not sm_search or sm_search in s['Symbol'])
                      and s['Score'] >= sm_min_score]
        if sm_show_radar:
            display_sm = []

        render_shortmove_table(display_sm, sm_main_placeholder)

        with sm_radar_placeholder.container():
            sm_radar_items = [(sym, info) for sym, info in sm_candidates.items()
                              if not sm_search or sm_search in sym]
            sm_radar_items.sort(key=lambda x: x[1].get('Score', 0), reverse=True)
            if sm_radar_items:
                for sym, info in sm_radar_items[:15]:
                    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
                    st.markdown(f"""<div class="shortmove-radar-card">
                        <a href="{tv_url}" target="_blank" class="shortmove-radar-sym">{sym}</a><br>
                        <span class="shortmove-radar-cond">{info["Conditions"]}</span><br>
                        <span style="color:#888;font-size:0.8rem;">
                            Skor: <b>{info["Score"]}</b> | BB-Poz: {info.get("BB_Pos","—")} |
                            RSI: {info.get("RSI","—")} | {info["Time"]}
                        </span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.caption("Short fırsatları taranıyor...")

        time.sleep(2)

# ================================================================
# SAYFA 4: SPOT WHALE TRACKER - SADECE BALINA
# ================================================================
elif current_page == "spot":
    st.caption("🐋 Binance Spot | Sadece Balina İşlemleri | $100K+ Large | $300K+ Whale | $1M+ Mega Whale")

    st.markdown("""
    <div style="background-color:#0a1a2a; border-left:4px solid #0984e3; padding:12px 16px; border-radius:4px; margin-bottom:16px;">
        <b style="color:#74b9ff;">Spot Whale Tracker Nasıl Çalışır?</b><br>
        <span style="color:#d5dbdb; font-size:0.9rem;">
        🐋 <b>Mega Whale:</b> $1M+ tek işlem — En büyük oyuncuların hareketi<br>
        🐋 <b>Whale:</b> $300K+ tek işlem — Büyük oyuncu hareketi<br>
        💰 <b>Large:</b> $100K+ işlemler + %3+ fiyat değişimi — Orta-büyük hareket<br>
        <b>Alım/Satım Baskısı:</b> Son 100 işlemdeki agresif alım (taker buy) vs satım oranı<br>
        <b>Not:</b> Sadece balina işlemi olan semboller gösterilir. Düşük hacimli/ufak işlemler filtrelenir.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Filtreler
    spot_col1, spot_col2, spot_col3 = st.columns([1, 1, 1])
    spot_search = spot_col1.text_input("🔍 Symbol Ara", placeholder="BTC...", key="spot_search").upper()
    spot_type_filter = spot_col2.multiselect(
        "Sinyal Tipi",
        ["🐋 MEGA WHALE BUY", "🐋 MEGA WHALE SELL", "🐋 MEGA WHALE",
         "🐋 WHALE BUY", "🐋 WHALE SELL", "🐋 WHALE",
         "💰 LARGE BUY", "💰 LARGE SELL", "💰 LARGE"],
        default=["🐋 MEGA WHALE BUY", "🐋 MEGA WHALE SELL", "🐋 MEGA WHALE",
                 "🐋 WHALE BUY", "🐋 WHALE SELL", "🐋 WHALE",
                 "💰 LARGE BUY", "💰 LARGE SELL", "💰 LARGE"],
        key="spot_type"
    )
    whale_only = spot_col3.checkbox("🐋 Sadece Whale/Mega", value=False)

    st.divider()

    # Ana layout
    col_spot_stats, col_spot_main = st.columns([1, 4])

    with col_spot_stats:
        st.subheader("📊 Whale Özeti")
        spot_stats_placeholder = st.empty()

    with col_spot_main:
        st.subheader("🐋 Spot Whale Sinyaller")
        spot_main_placeholder = st.empty()

    def get_spot_row_class(signal_color):
        mapping = {
            "mega-whale-buy":  "row-spot-mega-whale-buy",
            "mega-whale-sell": "row-spot-mega-whale-sell",
            "whale-buy":       "row-spot-whale-buy",
            "whale-sell":      "row-spot-whale-sell",
            "large-buy":       "row-spot-large-buy",
            "large-sell":      "row-spot-large-sell",
        }
        return mapping.get(signal_color, "row-spot-neutral")

    def format_usd(val):
        if val >= 1_000_000:
            return f"${val/1_000_000:.1f}M"
        if val >= 1_000:
            return f"${val/1_000:.0f}K"
        return f"${val:.0f}"

    def render_spot_table(data, placeholder):
        with placeholder.container():
            if data:
                html = (
                    "<table><tr>"
                    "<th>Saat</th>"
                    "<th>Symbol</th>"
                    "<th>Fiyat</th>"
                    "<th>24H Değ.</th>"
                    "<th>Sinyal</th>"
                    "<th>Max İşlem</th>"
                    "<th>Balina Sayısı</th>"
                    "<th>Alım/Satım</th>"
                    "<th>Hacim</th>"
                    "</tr>"
                )
                for row in data:
                    sym = row['Symbol']
                    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT"
                    r_cls = get_spot_row_class(row['SignalColor'])
                    tag_cls = f"spot-tag-{row['SignalColor']}"

                    chg = row['Chg']
                    chg_color = "#00ff88" if chg > 0 else "#ff4b4b"
                    chg_str = f"<span style='color:{chg_color}; font-weight:bold;'>{chg:+.2f}%</span>"

                    max_trade = row['MaxTradeUSD']
                    if row['HasMega']:
                        max_html = f"<span class='spot-mega-badge'>🐋 {format_usd(max_trade)}</span>"
                    elif row['HasWhale']:
                        max_html = f"<span class='spot-whale-badge'>🐋 {format_usd(max_trade)}</span>"
                    else:
                        max_html = format_usd(max_trade)

                    whale_count = row['WhaleCount']
                    large_count = row['LargeCount']

                    buy_ratio = row['BuyRatio']
                    sell_ratio = row['SellRatio']
                    pressure_html = (
                        f"<div class='pressure-bar-container'>"
                        f"<div class='pressure-buy' style='width:{buy_ratio}%;'></div>"
                        f"<div class='pressure-sell' style='width:{sell_ratio}%;'></div>"
                        f"</div>"
                        f"<small style='color:#00b894;'>{buy_ratio:.0f}%</small> / "
                        f"<small style='color:#d63031;'>{sell_ratio:.0f}%</small>"
                    )

                    vol_str = format_usd(row['QuoteVol'])

                    html += (
                        f"<tr class='{r_cls}'>"
                        f"<td style='color:#888; font-size:0.85rem;'>{row['Time']}</td>"
                        f"<td><a href='{tv_url}' target='_blank' class='sym-link-spot'>{sym}</a></td>"
                        f"<td style='font-family:monospace;'>{row['Price']}</td>"
                        f"<td>{chg_str}</td>"
                        f"<td><span class='{tag_cls}'>{row['SignalType']}</span></td>"
                        f"<td>{max_html}</td>"
                        f"<td style='text-align:center;'><b style='color:#f1c40f;'>{whale_count}</b> / <span style='color:#888;'>{large_count}</span></td>"
                        f"<td>{pressure_html}</td>"
                        f"<td style='color:#888;'>{vol_str}</td>"
                        f"</tr>"
                    )
                html += "</table>"
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("Balina sinyali bekleniyor... Spot piyasa taranıyor 🐋")

    while True:
        with radar.lock:
            spot_sigs = list(radar.spot_signals)

        # Filtrele
        display_spot = spot_sigs
        if spot_search:
            display_spot = [s for s in display_spot if spot_search in s['Symbol']]
        if spot_type_filter:
            display_spot = [s for s in display_spot if s['SignalType'] in spot_type_filter]
        if whale_only:
            display_spot = [s for s in display_spot if s['HasWhale'] or s['HasMega']]

        render_spot_table(display_spot, spot_main_placeholder)

        # İstatistik paneli
        with spot_stats_placeholder.container():
            total = len(spot_sigs)
            mega = sum(1 for s in spot_sigs if s['HasMega'])
            whales = sum(1 for s in spot_sigs if s['HasWhale'] and not s['HasMega'])
            large = sum(1 for s in spot_sigs if not s['HasWhale'] and not s['HasMega'])
            buys = sum(1 for s in spot_sigs if s['DominantSide'] == 'BUY')
            sells = sum(1 for s in spot_sigs if s['DominantSide'] == 'SELL')

            st.markdown(f"""
            <div style="background:#0a1a2a; border:1px solid #0984e3; border-radius:8px; padding:12px; margin-bottom:10px;">
                <div style="color:#74b9ff; font-weight:bold; margin-bottom:8px;">📊 Whale Özeti</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#aaa;">Toplam Sinyal</span>
                    <span style="color:#fff; font-weight:bold;">{total}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#00ff88;">🐋 Mega Whale</span>
                    <span style="color:#00ff88; font-weight:bold;">{mega}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#00b894;">🐋 Whale</span>
                    <span style="color:#00b894; font-weight:bold;">{whales}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#74b9ff;">💰 Large</span>
                    <span style="color:#74b9ff; font-weight:bold;">{large}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#00b894;">📈 Alım Ağırlıklı</span>
                    <span style="color:#00b894; font-weight:bold;">{buys}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#d63031;">📉 Satım Ağırlıklı</span>
                    <span style="color:#d63031; font-weight:bold;">{sells}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top aktif semboller (spot whale'ta)
            sym_counts = {}
            for s in spot_sigs[:50]:
                sym = s['Symbol']
                sym_counts[sym] = sym_counts.get(sym, 0) + 1
            top_syms = sorted(sym_counts.items(), key=lambda x: x[1], reverse=True)[:8]

            if top_syms:
                st.markdown("<div style='color:#74b9ff; font-weight:bold; margin-bottom:6px; margin-top:12px;'>🔥 En Aktif Semboller</div>", unsafe_allow_html=True)
                for sym, cnt in top_syms:
                    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT"
                    sym_sigs = [s for s in spot_sigs[:50] if s['Symbol'] == sym]
                    sym_buys = sum(1 for s in sym_sigs if s['DominantSide'] == 'BUY')
                    sym_sells = len(sym_sigs) - sym_buys
                    has_mega = any(s['HasMega'] for s in sym_sigs)
                    trend_color = "#00ff88" if has_mega else ("#00b894" if sym_buys >= sym_sells else "#d63031")
                    trend_icon = "🐋" if has_mega else ("↑" if sym_buys >= sym_sells else "↓")
                    st.markdown(f"""
                    <div style="background:#0d1520; border:1px solid #1a3a5c; border-radius:6px; padding:6px 10px; margin-bottom:4px;">
                        <a href="{tv_url}" target="_blank" class="sym-link-spot">{sym}</a>
                        <span style="float:right; color:{trend_color}; font-weight:bold;">{trend_icon} {cnt}x</span>
                    </div>
                    """, unsafe_allow_html=True)

        time.sleep(2)
