#!/usr/bin/env python3

import time, base64, requests, pandas as pd
import json, numpy as np, re
import os
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ================== CONFIG ==================

API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", os.path.expanduser("~/.secrets/kalshi_private_key.pem"))
PRIVATE_KEY_PATH = "kalshi_private_key.pem"
BASE_URL = "https://api.elections.kalshi.com"
SESSION = requests.Session()
MARKETS_PATH = "/trade-api/v2/markets"

PAGE_LIMIT = 1000
MAX_ROWS   = 500000
# ============================================

SPORTS_PATTERN = re.compile(
    r'(NFL|NBA|MLB|NHL|NCAA|NCAAF|NCAAB|WNBA|EPL|MLS|UFC|NASCAR|F1|FORMULA1|PGA|LPGA|ATP|WTA|'
    r'SOCCER|CRICKET|RUGBY|HOCKEY|BASEBALL|BASKETBALL|FOOTBALL|'
    r'SINGLEGAME|SGLGAME|PARLAY|COMBO|SGP|PLAYERPROP|PROP)',
    re.IGNORECASE
)

def is_sports_market(m: dict) -> bool:
    parts = [str(m.get("ticker","")), str(m.get("event_ticker","")),
             str(m.get("title","")), str(m.get("category",""))]
    if SPORTS_PATTERN.search(" ".join(parts)):
        return True
    if " at " in str(m.get("title","")).lower():
        return True
    return False

with open(PRIVATE_KEY_PATH, "rb") as f:
    PRIVATE_KEY = serialization.load_pem_private_key(f.read(), password=None)

def signed_get(path: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
    method = "GET"
    ts = str(int(time.time() * 1000))
    msg = ts + method + path
    sig = base64.b64encode(
        PRIVATE_KEY.sign(msg.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256())
    ).decode()
    headers = {"KALSHI-ACCESS-KEY": API_KEY_ID, "KALSHI-ACCESS-TIMESTAMP": ts, "KALSHI-ACCESS-SIGNATURE": sig}
    return requests.get(BASE_URL + path, headers=headers, params=params, timeout=30)

def signed_get_with_session(path, params=None, session=SESSION):
    method = "GET"
    ts = str(int(time.time() * 1000))
    msg = ts + method + path
    sig = base64.b64encode(
        PRIVATE_KEY.sign(msg.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256())
    ).decode()
    headers = {"KALSHI-ACCESS-KEY": API_KEY_ID, "KALSHI-ACCESS-TIMESTAMP": ts, "KALSHI-ACCESS-SIGNATURE": sig}
    return session.get(BASE_URL + path, headers=headers, params=params, timeout=30)

def fetch_markets_closed_up_to_cap(max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    now = pd.Timestamp.utcnow().tz_localize(None)
    kept, seen, cursor, page = [], set(), None, 0
    while True:
        if max_pages is not None and page >= max_pages: break
        params = {"limit": PAGE_LIMIT}
        if cursor: params["cursor"] = cursor
        resp = signed_get(MARKETS_PATH, params=params); resp.raise_for_status()
        payload = resp.json()
        markets = payload.get("markets", []) or []
        cursor = payload.get("cursor"); page += 1

        added = 0
        for m in markets:
            mid = m.get("ticker")
            if not mid or mid in seen: continue
            if is_sports_market(m):    continue
            ct = m.get("close_time")
            try:
                ct = pd.to_datetime(ct, errors="coerce", utc=True).tz_localize(None) if ct is not None else None
            except Exception:
                ct = None
            if ct is not None and ct < now:
                seen.add(mid); kept.append(m); added += 1
                if len(kept) >= MAX_ROWS:
                    print(f"Page {page}: kept +{added} (total kept {len(kept)}). Reached cap {MAX_ROWS}.")
                    return normalize_c_records(kept)
        print(f"Page {page}: kept +{added} this page (total kept {len(kept)}), next_cursor={bool(cursor)}")
        if not cursor or not markets: break
    return normalize_c_records(kept)

def normalize_c_records(records):
    if not records or not isinstance(records, list): return records
    header_map = records[0]
    if not isinstance(header_map, dict): return records
    looks_like_c = all(isinstance(k,str) and k.startswith("C") for k in header_map.keys())
    if not looks_like_c: return records
    out = []
    for rec in records[1:]:
        if isinstance(rec, dict):
            out.append({header_map[k]: rec[k] for k in rec.keys() if k in header_map})
    return out

def fetch_event_fast(et, retries=3):
    delay = 0.5
    for _ in range(retries):
        try:
            r = signed_get_with_session(f"/trade-api/v2/events/{et}")
            if r.status_code == 429: time.sleep(delay); delay *= 2; continue
            r.raise_for_status()
            return r.json().get("event", {})
        except Exception:
            time.sleep(delay); delay *= 2
    return {}

def fetch_all_events_concurrent(event_tickers, max_workers=8):
    t0 = time.perf_counter(); event_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_event_fast, et): et for et in event_tickers}
        for fut in as_completed(futs):
            et = futs[fut]; event_map[et] = fut.result() or {}
    dt = time.perf_counter() - t0; n = max(1, len(event_tickers))
    print(f"Fetched {len(event_map)}/{n} events in {dt:.2f}s (avg {dt/n:.3f}s/event, workers={max_workers})")
    return event_map

def build_dataframe(markets: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for m in markets:
        rows.append({
            "ticker":           m.get("ticker"),
            "event_ticker":     m.get("event_ticker"),
            "market_type":      m.get("type") or m.get("market_type") or "binary",
            "title":            m.get("title"),
            "subtitle":         m.get("subtitle") or m.get("sub_title"),
            "yes_sub_title":    m.get("yes_sub_title"),
            "no_sub_title":     m.get("no_sub_title"),
            "open_time":        m.get("open_time"),
            "close_time":       m.get("close_time"),
            "expected_expiration_time": m.get("expected_expiration_time") or m.get("expiration_time"),
            "category":         m.get("category"),
            "status":           m.get("status"),
            "strike_type":      m.get("strike_type"),
            "cap":              m.get("cap"),
            "yes_price":        m.get("yes_price"),
            "no_price":         m.get("no_price"),
            "yes_bid":          m.get("yes_bid"),
            "yes_ask":          m.get("yes_ask"),
            "no_bid":           m.get("no_bid"),
            "no_ask":           m.get("no_ask"),
            "last_price":       m.get("last_price"),
            "volume":           m.get("volume"),
            "volume_24h":       m.get("volume_24h"),
            "open_interest":    m.get("open_interest"),
            "result":           m.get("result"),
            "settlement_value": m.get("settlement_value"),
        })
    df = pd.DataFrame(rows)
    for col in ["open_time","close_time","expected_expiration_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
    return df

def write_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    with open(PRIVATE_KEY_PATH, "rb") as f:
        serialization.load_pem_private_key(f.read(), password=None)
    print("Loaded PEM successfully.")

    print(f"Fetching up to {MAX_ROWS} CLOSED non-sports markets (pages of {PAGE_LIMIT}) from {BASE_URL}{MARKETS_PATH} ...")
    markets = fetch_markets_closed_up_to_cap()
    print(f"Collected {len(markets)} closed non-sports markets. Building DataFrame...")

    df = build_dataframe(markets)
    if df.empty:
        print(" No eligible markets found.")
        write_json({"events": []}, "events_only.json")
        write_json({"events": [], "markets": []}, "kalshi_final.json")
        write_json([], "markets_enriched.json")
        return

    # unique events
    if "event_ticker" not in df.columns or not df["event_ticker"].notna().any():
        print(" No event_ticker values found.")
        write_json({"events": []}, "events_only.json")
        write_json({"events": [], "markets": json.loads(df.to_json(orient='records', date_format='iso'))}, "kalshi_final.json")
        write_json(json.loads(df.to_json(orient='records', date_format='iso')), "markets_enriched.json")
        return

    unique_events = df["event_ticker"].dropna().unique().tolist()
    print(f"Fetching {len(unique_events)} event objects (concurrent)...")
    event_map = fetch_all_events_concurrent(unique_events, max_workers=8)

    # ---- Build events list (for events_only + bundle) ----
    events_list = []
    for et, data in event_map.items():
        if not data: continue
        events_list.append({
            "event_ticker":       data.get("ticker"),
            "series_ticker":      data.get("series_ticker"),
            "sub_title":          data.get("sub_title"),
            "title":              data.get("title"),
            "mutually_exclusive": data.get("mutually_exclusive", False),
            "category":           data.get("category"),
        })
    events_list.sort(key=lambda x: (x.get("series_ticker") or "", x.get("event_ticker") or ""))
    write_json({"events": events_list}, "events_only.json")
    print(f" Saved {len(events_list)} events to events_only.json")

    # ---- ENRICH the markets like the screenshot ----
    # map selected event fields onto each market via event_ticker
    df["series_ticker"]      = df["event_ticker"].map(lambda et: (event_map.get(et) or {}).get("series_ticker"))
    df["sub_title"]          = df.get("sub_title") if "sub_title" in df.columns else None
    # prefer event sub_title if present; fall back to market subtitle
    df["sub_title"]          = df["event_ticker"].map(lambda et: (event_map.get(et) or {}).get("sub_title")).fillna(df.get("subtitle"))
    df["mutually_exclusive"] = df["event_ticker"].map(lambda et: (event_map.get(et) or {}).get("mutually_exclusive"))

    # reorder columns to match the demo vibe
    front_cols = [
        "ticker","event_ticker","market_type","title","subtitle","yes_sub_title","no_sub_title",
        "open_time","close_time","expected_expiration_time",
        "series_ticker","sub_title","mutually_exclusive",
        "strike_type","cap","category","status",
        "yes_price","no_price","yes_bid","yes_ask","no_bid","no_ask",
        "last_price","volume","volume_24h","open_interest","result","settlement_value"
    ]
    # keep only those that exist + append any extras at end
    cols = [c for c in front_cols if c in df.columns] + [c for c in df.columns if c not in front_cols]
    df = df[cols].copy()

    # save enriched table
    markets_enriched = json.loads(df.to_json(orient="records", date_format="iso"))
    write_json(markets_enriched, "markets_enriched.json")
    print(f"✅ Saved {len(markets_enriched)} rows to markets_enriched.json")

    # ---- Final bundle (events + enriched markets) ----
    write_json({"events": events_list, "markets": markets_enriched}, "kalshi_final.json")
    print(" Saved kalshi_final.json (events + enriched markets)")

if __name__ == "__main__":
    main()