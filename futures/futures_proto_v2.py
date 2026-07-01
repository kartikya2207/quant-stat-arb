"""Futures pipeline v2 — adds CORPORATE-ACTION adjustment (the smoke-test must-fix).
Corp factors from yfinance .splits, applied (multiplicatively, back-adjusted to
current scale) to the futures price panel BEFORE the additive roll back-adjustment.
Spot reference = NSE CASH bhavcopy (consistent, raw) so basis = true carry."""
import requests, zipfile, io, os, time, warnings
import numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")   # repo-relative, gitignored
os.makedirs(CACHE, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
     "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/"}
CUTOVER = pd.Timestamp("2024-07-08"); SYMS = ["RELIANCE", "SBIN"]; ROLL_BEFORE = 2

def _get(url):
    for a in range(3):
        try:
            r = requests.get(url, headers=H, timeout=40)
            if r.status_code == 200 and len(r.content) > 200:
                return r.content
            return None
        except Exception:
            time.sleep(1.0 * (a + 1))
    return None

def fetch_fo(d):                       # futures (reuse v1 cache: YYYYMMDD.csv)
    fn = os.path.join(CACHE, d.strftime("%Y%m%d") + ".csv")
    if os.path.exists(fn):
        try:
            df = pd.read_csv(fn)
        except pd.errors.EmptyDataError:
            return None
        if df.empty: return None
        df["date"] = pd.to_datetime(df["date"]); df["expiry"] = pd.to_datetime(df["expiry"]); return df
    url = (f"https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
           if d >= CUTOVER else
           f"https://archives.nseindia.com/content/historical/DERIVATIVES/{d.year}/{d.strftime('%b').upper()}/fo{d.strftime('%d')}{d.strftime('%b').upper()}{d.year}bhav.csv.zip")
    c = _get(url)
    if c is None:
        pd.DataFrame().to_csv(fn, index=False); return None
    raw = pd.read_csv(zipfile.ZipFile(io.BytesIO(c)).open(zipfile.ZipFile(io.BytesIO(c)).namelist()[0])); raw.columns=[x.strip() for x in raw.columns]
    if "INSTRUMENT" in raw.columns:
        f = raw[raw["INSTRUMENT"] == "FUTSTK"]
        out = pd.DataFrame({"symbol": f["SYMBOL"], "expiry": pd.to_datetime(f["EXPIRY_DT"], format="%d-%b-%Y"),
            "close": f["CLOSE"], "settle": f["SETTLE_PR"], "volume": f["CONTRACTS"], "oi": f["OPEN_INT"]})
    else:
        f = raw[raw["FinInstrmTp"] == "STF"]
        out = pd.DataFrame({"symbol": f["TckrSymb"], "expiry": pd.to_datetime(f["XpryDt"]),
            "close": f["ClsPric"], "settle": f["SttlmPric"], "volume": f["TtlTradgVol"], "oi": f["OpnIntrst"]})
    out["date"] = d; out.to_csv(fn, index=False); time.sleep(0.25)
    out["date"] = pd.to_datetime(out["date"]); out["expiry"] = pd.to_datetime(out["expiry"]); return out

def fetch_cm(d):                       # cash equity EQ close
    fn = os.path.join(CACHE, "cm_" + d.strftime("%Y%m%d") + ".csv")
    if os.path.exists(fn):
        try:
            df = pd.read_csv(fn)
        except pd.errors.EmptyDataError:
            return None
        if df.empty: return None
        df["date"] = pd.to_datetime(df["date"]); return df
    url = (f"https://archives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
           if d >= CUTOVER else
           f"https://archives.nseindia.com/content/historical/EQUITIES/{d.year}/{d.strftime('%b').upper()}/cm{d.strftime('%d')}{d.strftime('%b').upper()}{d.year}bhav.csv.zip")
    c = _get(url)
    if c is None:
        pd.DataFrame().to_csv(fn, index=False); return None
    raw = pd.read_csv(zipfile.ZipFile(io.BytesIO(c)).open(zipfile.ZipFile(io.BytesIO(c)).namelist()[0])); raw.columns=[x.strip() for x in raw.columns]
    if "SERIES" in raw.columns:
        e = raw[(raw["SERIES"] == "EQ") & (raw["SYMBOL"].isin(SYMS))]
        out = pd.DataFrame({"symbol": e["SYMBOL"], "cash": e["CLOSE"]})
    else:
        e = raw[(raw["SctySrs"] == "EQ") & (raw["TckrSymb"].isin(SYMS))]
        out = pd.DataFrame({"symbol": e["TckrSymb"], "cash": e["ClsPric"]})
    out["date"] = d; out.to_csv(fn, index=False); time.sleep(0.25)
    out["date"] = pd.to_datetime(out["date"]); return out

def splits_of(sym):
    s = yf.Ticker(sym + ".NS").splits
    s = s[s != 1.0]
    if len(s):
        s.index = pd.DatetimeIndex(s.index).tz_convert(None).normalize()
    return s

def cumfactor(dates, splits):           # back-adjust to current scale: x(1/f) for ex_date > t
    cf = pd.Series(1.0, index=pd.DatetimeIndex(dates))
    for exd, f in splits.items():
        cf[cf.index < exd] *= (1.0 / float(f))
    return cf.values

def active_expiry(exps, d):
    elig = sorted(e for e in exps if e >= d)
    if not elig: return None
    return elig[1] if (len(elig) > 1 and (elig[0] - d).days <= ROLL_BEFORE) else elig[0]

def build_continuous(panel):
    panel = panel.sort_values(["date", "expiry"]); by = panel.set_index(["date", "expiry"])
    rows = []
    for d in sorted(panel["date"].unique()):
        d = pd.Timestamp(d); exps = sorted(panel[panel["date"] == d]["expiry"].unique())
        ae = active_expiry(exps, d)
        if ae is None: continue
        r = by.loc[(d, ae)]; px = r["close"] if pd.notna(r["close"]) and r["close"] > 0 else r["settle"]
        rows.append({"date": d, "ae": ae, "raw_close": float(px), "oi": float(r["oi"])})
    cont = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    adj = cont["raw_close"].values.astype(float).copy(); rolls = []
    for i in range(1, len(cont)):
        if cont["ae"].iloc[i] != cont["ae"].iloc[i - 1]:
            dp = cont["date"].iloc[i - 1]; A = cont["ae"].iloc[i - 1]; B = cont["ae"].iloc[i]
            try: gap = float(by.loc[(dp, B), "close"]) - float(by.loc[(dp, A), "close"])
            except KeyError: gap = 0.0
            rolls.append((cont["date"].iloc[i].date(), A.date(), B.date(), round(gap, 2))); adj[:i] += gap
    cont["adj_close"] = adj
    return cont, rolls

WINDOWS = [("2019 (OLD csv)", "2019-06-03", "2019-07-31"),
           ("2024 (NEW udiff)", "2024-09-02", "2024-10-31")]
sp = {s: splits_of(s) for s in SYMS}
print("yfinance splits used:", {s: {str(k.date()): float(v) for k, v in sp[s].items()} for s in SYMS})

for wname, start, end in WINDOWS:
    print("\n" + "#" * 76 + f"\n# WINDOW {wname}: {start} -> {end}\n" + "#" * 76)
    days = pd.bdate_range(start, end); fo, cm = [], []
    for d in days:
        a = fetch_fo(d);  b = fetch_cm(d)
        if a is not None: fo.append(a[a["symbol"].isin(SYMS)])
        if b is not None: cm.append(b)
    fo = pd.concat(fo, ignore_index=True); cm = pd.concat(cm, ignore_index=True)
    for sym in SYMS:
        s_fo = fo[fo["symbol"] == sym].copy()
        # corp-adjusted panel (prices x cumfactor by date)
        s_fo_adj = s_fo.copy()
        cf = cumfactor(s_fo_adj["date"], sp[sym]); s_fo_adj["close"] = s_fo_adj["close"].values * cf; s_fo_adj["settle"] = s_fo_adj["settle"].values * cf
        raw_cont, rolls = build_continuous(s_fo)
        adj_cont, _ = build_continuous(s_fo_adj)
        # cash (raw + corp-adjusted)
        c = cm[cm["symbol"] == sym][["date", "cash"]].drop_duplicates("date").set_index("date")["cash"]
        c = c.reindex(raw_cont["date"].values)
        cfc = cumfactor(raw_cont["date"], sp[sym]); cash_adj = c.values * cfc
        # --- validations ---
        raw_ret = pd.Series(raw_cont["raw_close"].values).pct_change()
        adj_ret = pd.Series(adj_cont["adj_close"].values).pct_change()
        cashadj_ret = pd.Series(cash_adj).pct_change()
        basis = (raw_cont["raw_close"].values - c.values) / c.values * 100.0
        corr = pd.Series(adj_ret.values).corr(pd.Series(cashadj_ret.values))
        print(f"\n--- {sym} ({wname}) ---")
        print(f"  rolls: {rolls}")
        print(f"  max |daily ret|  RAW(unadj)={np.nanmax(np.abs(raw_ret)):.1%}   ADJUSTED={np.nanmax(np.abs(adj_ret)):.2%}  <-- fake-jump check")
        print(f"  basis (raw fut vs raw cash): mean={np.nanmean(basis):+.2f}%  median={np.nanmedian(basis):+.2f}%  max|={np.nanmax(np.abs(basis)):.2f}%")
        print(f"  return corr (adj future vs adj cash) = {corr:.4f}")
        print(f"  continuous close: NaNs={int(np.isnan(adj_cont['adj_close']).sum())}  min={adj_cont['adj_close'].min():.1f} max={adj_cont['adj_close'].max():.1f}")
print("\nDONE")
