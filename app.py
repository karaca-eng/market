Content is user-generated and unverified.
import streamlit as st
import requests
import pandas as pd
import json
import os
import time
import threading
from datetime import datetime, timezone

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto Funding Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #08090d;
    --surface: #0f1117;
    --surface2: #161922;
    --border: #1f2133;
    --accent: #4ade80;
    --accent2: #f97316;
    --red: #f43f5e;
    --green: #4ade80;
    --yellow: #facc15;
    --blue: #60a5fa;
    --text: #dde1f0;
    --muted: #5a6080;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }

h1,h2,h3,h4 { font-family: 'IBM Plex Mono', monospace !important; }

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.logo { font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem; color: var(--accent); font-weight: 700; }
.status-dot { width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px; }
.dot-on  { background:var(--green);box-shadow:0 0 8px var(--green);animation:blink 1.5s infinite; }
.dot-off { background:var(--muted); }
@keyframes blink { 0%,100%{opacity:1}50%{opacity:.3} }

.stat-grid { display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px; }
.stat-box { background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center; }
.stat-val { font-family:'IBM Plex Mono',monospace;font-size:1.6rem;font-weight:700; }
.stat-lbl { font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:2px;margin-top:4px; }

.trade-card {
    background:var(--surface);border:1px solid var(--border);border-radius:10px;
    padding:16px 20px;margin-bottom:10px;
    font-family:'IBM Plex Mono',monospace;font-size:0.8rem;
    position:relative;overflow:hidden;
}
.trade-card::before { content:'';position:absolute;left:0;top:0;bottom:0;width:3px; }
.card-open::before  { background:var(--yellow); }
.card-win::before   { background:var(--green); }
.card-loss::before  { background:var(--red); }

.badge { display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.65rem;font-weight:700;letter-spacing:1px;text-transform:uppercase; }
.b-open { background:rgba(250,204,21,.12);color:var(--yellow);border:1px solid rgba(250,204,21,.3); }
.b-win  { background:rgba(74,222,128,.12);color:var(--green);border:1px solid rgba(74,222,128,.3); }
.b-loss { background:rgba(244,63,94,.12);color:var(--red);border:1px solid rgba(244,63,94,.3); }
.b-long { background:rgba(96,165,250,.12);color:var(--blue);border:1px solid rgba(96,165,250,.3); }
.b-short{ background:rgba(249,115,22,.12);color:var(--accent2);border:1px solid rgba(249,115,22,.3); }

.log-box {
    background:var(--surface);border:1px solid var(--border);border-radius:10px;
    padding:14px;height:260px;overflow-y:auto;
    font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:var(--muted);
}
.log-entry { padding:2px 0;border-bottom:1px solid var(--border); }
.log-green { color:var(--green); }
.log-red   { color:var(--red); }
.log-yellow{ color:var(--yellow); }

.stButton>button {
    background:transparent!important;border:1px solid var(--border)!important;
    color:var(--text)!important;font-family:'IBM Plex Mono',monospace!important;
    font-size:0.75rem!important;border-radius:6px!important;
}
.stButton>button:hover { border-color:var(--accent)!important;color:var(--accent)!important; }
.stTabs [data-baseweb="tab"] { font-family:'IBM Plex Mono',monospace!important;font-size:0.78rem!important; }
div[data-testid="stMetric"] { background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TRADES_FILE   = "trades.json"
LOG_FILE      = "bot_log.json"
CONFIG_FILE   = "bot_config.json"
MIN_RATE      = 0.20
MAX_OPEN      = 3
SCAN_INTERVAL = 60

# ── Persistence ───────────────────────────────────────────────────────────────
def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_trades():  return load_json(TRADES_FILE, [])
def save_trades(t): save_json(TRADES_FILE, t)
def load_logs():    return load_json(LOG_FILE, [])
def save_logs(l):   save_json(LOG_FILE, l[-300:])
def load_config():
    return load_json(CONFIG_FILE, {"running": False, "last_scan": None, "total_scans": 0})
def save_config(c): save_json(CONFIG_FILE, c)

def add_log(msg, level="info"):
    logs = load_logs()
    logs.append({"time": now_iso(), "msg": msg, "level": level})
    save_logs(logs)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

# ── Binance API ───────────────────────────────────────────────────────────────
def fetch_funding_rates():
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex", timeout=10)
        data = r.json()
        rows = []
        for item in data:
            sym = item.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            try:
                rate  = float(item.get("lastFundingRate", 0)) * 100
                mark  = float(item.get("markPrice", 0))
                rows.append({
                    "symbol": sym.replace("USDT", ""),
                    "funding_rate": round(rate, 4),
                    "annual_rate": round(rate * 3 * 365, 2),
                    "mark_price": mark,
                })
            except Exception:
                continue
        return sorted(rows, key=lambda x: abs(x["funding_rate"]), reverse=True)
    except Exception as e:
        add_log(f"API hatası: {e}", "error")
        return []

def get_price(symbol):
    try:
        r = requests.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}USDT", timeout=5)
        return float(r.json().get("price", 0))
    except Exception:
        return 0.0

# ── Strategy ──────────────────────────────────────────────────────────────────
def calculate_tp_sl(direction, entry, rate_abs):
    """Dynamic TP/SL — higher rate = wider targets, always 1:2 R:R"""
    if rate_abs >= 0.50:
        tp_pct, sl_pct = 4.0, 2.0
    elif rate_abs >= 0.30:
        tp_pct, sl_pct = 3.0, 1.5
    else:
        tp_pct, sl_pct = 2.0, 1.0

    if direction == "LONG":
        tp = round(entry * (1 + tp_pct / 100), 6)
        sl = round(entry * (1 - sl_pct / 100), 6)
    else:
        tp = round(entry * (1 - tp_pct / 100), 6)
        sl = round(entry * (1 + sl_pct / 100), 6)

    return tp, sl, tp_pct, sl_pct

# ── Bot core ──────────────────────────────────────────────────────────────────
def bot_scan():
    config = load_config()
    if not config.get("running"):
        return

    trades = load_trades()

    # 1. Check open trades for TP/SL hits
    for i, t in enumerate(trades):
        if t["status"] != "OPEN":
            continue
        current = get_price(t["symbol"])
        if current == 0:
            continue
        trades[i]["current_price"] = round(current, 6)
        direction = t["direction"]

        if direction == "LONG":
            pnl = (current - t["entry"]) / t["entry"] * 100
            if current >= t["tp"]:
                pnl_val = round((t["tp"] - t["entry"]) / t["entry"] * 100, 2)
                trades[i].update({"status":"WIN","close_price":t["tp"],"close_time":now_iso(),"pnl_pct":pnl_val})
                add_log(f"✅ WIN  {t['symbol']} LONG  +{pnl_val:.2f}%", "win")
            elif current <= t["sl"]:
                pnl_val = round((t["sl"] - t["entry"]) / t["entry"] * 100, 2)
                trades[i].update({"status":"LOSS","close_price":t["sl"],"close_time":now_iso(),"pnl_pct":pnl_val})
                add_log(f"❌ LOSS {t['symbol']} LONG  {pnl_val:.2f}%", "loss")
            else:
                trades[i]["pnl_pct"] = round(pnl, 2)
        else:
            pnl = (t["entry"] - current) / t["entry"] * 100
            if current <= t["tp"]:
                pnl_val = round((t["entry"] - t["tp"]) / t["entry"] * 100, 2)
                trades[i].update({"status":"WIN","close_price":t["tp"],"close_time":now_iso(),"pnl_pct":pnl_val})
                add_log(f"✅ WIN  {t['symbol']} SHORT +{pnl_val:.2f}%", "win")
            elif current >= t["sl"]:
                pnl_val = round((t["entry"] - t["sl"]) / t["entry"] * 100, 2)
                trades[i].update({"status":"LOSS","close_price":t["sl"],"close_time":now_iso(),"pnl_pct":pnl_val})
                add_log(f"❌ LOSS {t['symbol']} SHORT {pnl_val:.2f}%", "loss")
            else:
                trades[i]["pnl_pct"] = round(pnl, 2)

    # 2. Open new trades if slots available
    open_trades  = [t for t in trades if t["status"] == "OPEN"]
    open_symbols = {t["symbol"] for t in open_trades}
    slots = MAX_OPEN - len(open_trades)

    if slots > 0:
        rates = fetch_funding_rates()
        added = 0
        for row in rates:
            if added >= slots:
                break
            rate = row["funding_rate"]
            sym  = row["symbol"]
            if abs(rate) < MIN_RATE:
                break
            if sym in open_symbols:
                continue

            direction = "SHORT" if rate > 0 else "LONG"
            entry = get_price(sym)
            if entry == 0:
                continue

            tp, sl, tp_pct, sl_pct = calculate_tp_sl(direction, entry, abs(rate))
            trade = {
                "id": len(trades) + added + 1,
                "symbol": sym,
                "direction": direction,
                "entry": entry,
                "tp": tp, "sl": sl,
                "tp_pct": tp_pct, "sl_pct": sl_pct,
                "current_price": entry,
                "pnl_pct": 0.0,
                "status": "OPEN",
                "funding_rate": rate,
                "annual_rate": row["annual_rate"],
                "open_time": now_iso(),
                "close_time": None,
                "close_price": None,
            }
            trades.append(trade)
            open_symbols.add(sym)
            added += 1
            add_log(f"🚀 AÇILDI {sym} {direction} | Rate:{rate:+.3f}% TP:{tp_pct}% SL:{sl_pct}%", "open")

    # 3. Save
    config["last_scan"] = now_iso()
    config["total_scans"] = config.get("total_scans", 0) + 1
    save_config(config)
    save_trades(trades)

# ── Background thread ─────────────────────────────────────────────────────────
def background_loop():
    while True:
        try:
            bot_scan()
        except Exception as e:
            add_log(f"Loop hatası: {e}", "error")
        time.sleep(SCAN_INTERVAL)

if "bot_thread_started" not in st.session_state:
    th = threading.Thread(target=background_loop, daemon=True)
    th.start()
    st.session_state["bot_thread_started"] = True

# ── Stats ─────────────────────────────────────────────────────────────────────
def get_stats(trades):
    closed = [t for t in trades if t["status"] in ("WIN","LOSS")]
    wins   = [t for t in closed if t["status"] == "WIN"]
    open_t = [t for t in trades if t["status"] == "OPEN"]
    wr     = len(wins) / len(closed) * 100 if closed else 0
    tp     = sum(t.get("pnl_pct",0) for t in closed)
    return {
        "total": len(trades), "open": len(open_t),
        "closed": len(closed), "wins": len(wins),
        "losses": len(closed)-len(wins),
        "win_rate": round(wr,1),
        "total_pnl": round(tp,2),
        "avg_pnl": round(tp/len(closed),2) if closed else 0,
    }

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
config  = load_config()
running = config.get("running", False)
trades  = load_trades()
stats   = get_stats(trades)
logs    = load_logs()

dot_cls   = "dot-on" if running else "dot-off"
status_lbl = "ÇALIŞIYOR" if running else "DURDURULDU"
last_scan  = config.get("last_scan") or "—"
if last_scan != "—":
    last_scan = last_scan[11:19] + " UTC"

st.markdown(f"""
<div class="topbar">
  <div class="logo">🤖 AUTO FUNDING BOT</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:#5a6080;">
    <span class="status-dot {dot_cls}"></span>
    <span style="color:{'#4ade80' if running else '#5a6080'}">{status_lbl}</span>
    &nbsp;|&nbsp; Son tarama: {last_scan}
    &nbsp;|&nbsp; {config.get('total_scans',0)} tarama yapıldı
  </div>
</div>
""", unsafe_allow_html=True)

# Controls
c1, c2, c3, _ = st.columns([1,1,1,5])
with c1:
    btn_lbl = "⏸ DURDUR" if running else "▶ BAŞLAT"
    if st.button(btn_lbl):
        config["running"] = not running
        save_config(config)
        add_log("Bot başlatıldı" if not running else "Bot durduruldu")
        st.rerun()
with c2:
    if st.button("🔄 Yenile"):
        st.rerun()
with c3:
    if st.button("🗑️ Sıfırla"):
        save_trades([]); save_logs([])
        add_log("Sıfırlandı")
        st.rerun()

st.markdown("---")

# Stats bar
wr_color  = "#4ade80" if stats["win_rate"]>=55 else "#facc15" if stats["win_rate"]>=45 else "#f43f5e"
pnl_color = "#4ade80" if stats["total_pnl"]>=0 else "#f43f5e"

st.markdown(f"""
<div class="stat-grid">
  <div class="stat-box">
    <div class="stat-val" style="color:#60a5fa">{stats['open']}<span style="font-size:1rem;color:#5a6080">/{MAX_OPEN}</span></div>
    <div class="stat-lbl">Açık İşlem</div>
  </div>
  <div class="stat-box">
    <div class="stat-val" style="color:{wr_color}">{stats['win_rate']}%</div>
    <div class="stat-lbl">Başarı Oranı</div>
  </div>
  <div class="stat-box">
    <div class="stat-val" style="color:{pnl_color}">{stats['total_pnl']:+.1f}%</div>
    <div class="stat-lbl">Toplam PNL</div>
  </div>
  <div class="stat-box">
    <div class="stat-val" style="color:#a78bfa">{stats['wins']}W / {stats['losses']}L</div>
    <div class="stat-lbl">Win / Loss</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊  Açık İşlemler", "📜  Geçmiş & Log", "📈  Analiz"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    open_t = [t for t in trades if t["status"] == "OPEN"]
    if not open_t:
        st.info("Açık işlem yok. Bot çalışıyorsa 0.20%+ funding rate bulunca otomatik açacak.")
    for t in reversed(open_t):
        pnl = t.get("pnl_pct", 0)
        pc  = "#4ade80" if pnl>=0 else "#f43f5e"
        db  = "b-long" if t["direction"]=="LONG" else "b-short"
        st.markdown(f"""
        <div class="trade-card card-open">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <span style="font-size:1.05rem;color:#dde1f0;font-weight:700;">{t['symbol']}/USDT</span>
            <div><span class="badge {db}">{t['direction']}</span>&nbsp;<span class="badge b-open">AÇIK</span></div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;font-size:0.75rem;">
            <div><span style="color:#5a6080">Giriş</span><br/><b>{t['entry']}</b></div>
            <div><span style="color:#5a6080">Anlık</span><br/><b>{t.get('current_price','...')}</b></div>
            <div><span style="color:#4ade80">TP +{t.get('tp_pct','')}%</span><br/><b>{t['tp']}</b></div>
            <div><span style="color:#f43f5e">SL -{t.get('sl_pct','')}%</span><br/><b>{t['sl']}</b></div>
          </div>
          <div style="margin-top:10px;font-size:0.72rem;color:#5a6080;">
            PNL: <span style="color:{pc};font-weight:700">{pnl:+.2f}%</span>
            &nbsp;|&nbsp; Funding: <b style="color:#facc15">{t['funding_rate']:+.3f}%</b>
            &nbsp;|&nbsp; Yıllık: {t['annual_rate']:+.0f}%
            &nbsp;|&nbsp; Açılış: {t['open_time'][11:19]} UTC
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    col_h, col_l = st.columns([3,2])

    with col_h:
        st.markdown("**Kapanan İşlemler**")
        closed_t = [t for t in trades if t["status"] in ("WIN","LOSS")]
        if not closed_t:
            st.info("Henüz kapanan işlem yok.")
        for t in reversed(closed_t):
            s   = t["status"]
            cc  = "card-win" if s=="WIN" else "card-loss"
            bc  = "b-win" if s=="WIN" else "b-loss"
            em  = "✅" if s=="WIN" else "❌"
            pnl = t.get("pnl_pct",0)
            pc  = "#4ade80" if pnl>=0 else "#f43f5e"
            db  = "b-long" if t["direction"]=="LONG" else "b-short"
            st.markdown(f"""
            <div class="trade-card {cc}">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="color:#dde1f0;font-weight:700;">{em} {t['symbol']}/USDT</span>
                <div><span class="badge {db}">{t['direction']}</span>&nbsp;<span class="badge {bc}">{s}</span></div>
              </div>
              <div style="font-size:0.72rem;color:#5a6080;">
                Giriş: <b>{t['entry']}</b> → Kapanış: <b>{t.get('close_price','-')}</b>
                &nbsp;|&nbsp; PNL: <span style="color:{pc};font-weight:700">{pnl:+.2f}%</span>
                <br/>Funding: {t['funding_rate']:+.3f}%
                &nbsp;|&nbsp; {t['open_time'][11:19]} UTC
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_l:
        st.markdown("**Bot Logu**")
        lc = {"win":"log-green","loss":"log-red","open":"log-yellow","error":"log-red"}
        html = ""
        for e in reversed(logs[-60:]):
            ts  = e["time"][11:19]
            cls = lc.get(e["level"],"")
            html += f'<div class="log-entry"><span style="color:#5a6080">{ts}</span> <span class="{cls}">{e["msg"]}</span></div>'
        st.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    closed_t = [t for t in trades if t["status"] in ("WIN","LOSS")]
    if len(closed_t) < 2:
        st.info("Analiz için en az 2 kapanan işlem gerekli.")
    else:
        df = pd.DataFrame(closed_t)
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)
        df["cumulative_pnl"] = df["pnl_pct"].cumsum()
        df["trade_no"] = range(1, len(df)+1)

        st.markdown("**Kümülatif PNL (%)**")
        st.line_chart(df.set_index("trade_no")["cumulative_pnl"])

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Yön Performansı**")
            dir_s = df.groupby("direction").agg(
                İşlem=("pnl_pct","count"),
                Ort_PNL=("pnl_pct","mean"),
                Basari=("status", lambda x: f"{(x=='WIN').mean()*100:.0f}%")
            ).reset_index()
            dir_s["Ort_PNL"] = dir_s["Ort_PNL"].round(2)
            st.dataframe(dir_s, hide_index=True, use_container_width=True)
        with a2:
            st.markdown("**En Çok İşlem Yapılan Coinler**")
            sc = df["symbol"].value_counts().head(10).reset_index()
            sc.columns = ["Coin","İşlem"]
            st.dataframe(sc, hide_index=True, use_container_width=True)

        st.markdown("**Tüm Kapanan İşlemler**")
        cols = ["symbol","direction","entry","close_price","pnl_pct","status","funding_rate"]
        st.dataframe(
            df[cols].rename(columns={
                "symbol":"Coin","direction":"Yön","entry":"Giriş",
                "close_price":"Kapanış","pnl_pct":"PNL%",
                "status":"Sonuç","funding_rate":"Funding%"
            }),
            hide_index=True, use_container_width=True
        )

# Auto-refresh every 30s while running
if running:
    st.markdown("""
    <script>setTimeout(()=>window.location.reload(),30000);</script>
    """, unsafe_allow_html=True)
