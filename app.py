import os
import requests
import logging
import time
import threading
import json
import uuid
import math
from datetime import datetime, timezone, timedelta

from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

###############################################################################
# If you have firebase.py => from firebase import db
# For demonstration, we'll mock with db=None
###############################################################################
db = None

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

###############################################################################
# Flask app
###############################################################################
app = Flask(__name__)
CORS(app)
limiter = Limiter(get_remote_address, app=app)

###############################################################################
# Configuration
###############################################################################
# 1) data.solanatracker.io calls => supply x-api-key in header
SOLTRACKER_DATA_API_KEY = "180c0541-91f5-4cc5-bb05-a9163df8d58e"  # put your real key
DATA_BASE_URL = "https://data.solanatracker.io"

# 2) RPC endpoint => ?api_key=... for JSON-RPC calls
SOLANATRACKER_RPC_URL = (
    "https://rpc-mainnet.solanatracker.io/?api_key=3275e8b9-7edb-4b56-a158-5ab0eeff47c9"
)

CACHE_DURATION = 300  # 5 minutes
DB_CACHE_DURATION = 3600  # 1 hour

###############################################################################
# Basic Utility Calls to Solana Tracker
###############################################################################

def fetch_pnl_data(wallet, max_retries=3):
    """
    GET /pnl/{wallet} with x-api-key
    Returns JSON with 'tokens' mapping and summary info (lifetime).
    We'll do showHistoricPnL/holdingCheck to get more details.
    """
    url = f"{DATA_BASE_URL}/pnl/{wallet}"
    headers = {"x-api-key": SOLTRACKER_DATA_API_KEY}
    params = {"showHistoricPnL": "true", "holdingCheck": "true"}

    for attempt in range(max_retries):
        try:
            logger.debug(f"[fetch_pnl_data] => attempt={attempt+1}, wallet={wallet}")
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ReadTimeout:
            logger.warning("[fetch_pnl_data] => readtimeout")
            if attempt < max_retries - 1:
                continue
            else:
                raise
        except requests.exceptions.RequestException as e:
            logger.warning(f"[fetch_pnl_data] => {e}, attempt={attempt+1}")
            if attempt < max_retries - 1:
                continue
            else:
                raise

def fetch_wallet_trades(owner, max_retries=3):
    """
    GET /wallet/{owner}/trades => returns a list of trades with 'time', 'from', 'to', 'volume'
    We'll do a single request for simplicity; if many trades, you might need pagination.
    """
    url = f"{DATA_BASE_URL}/wallet/{owner}/trades"
    headers = {"x-api-key": SOLTRACKER_DATA_API_KEY}
    params = {}  # e.g. {"cursor": "..."} for pagination

    for attempt in range(max_retries):
        try:
            logger.debug(f"[fetch_wallet_trades] => attempt={attempt+1}, wallet={owner}")
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            return r.json()  # expect: {"trades": [...], "nextCursor": "...", "hasNextPage": bool}
        except requests.exceptions.ReadTimeout:
            logger.warning("[fetch_wallet_trades] => readtimeout")
            if attempt < max_retries - 1:
                continue
            else:
                raise
        except requests.exceptions.RequestException as e:
            logger.warning(f"[fetch_wallet_trades] => {e}, attempt={attempt+1}")
            if attempt < max_retries - 1:
                continue
            else:
                raise

def fetch_wallet_basic(owner):
    """
    GET /wallet/{owner}/basic => quick info about tokens (lightweight).
    """
    url = f"{DATA_BASE_URL}/wallet/{owner}/basic"
    headers = {"x-api-key": SOLTRACKER_DATA_API_KEY}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()

def fetch_token_info_from_data(token_address):
    """
    GET /tokens/{tokenAddress} => returns token info (name, symbol, image, etc.)
    """
    url = f"{DATA_BASE_URL}/tokens/{token_address}"
    headers = {"x-api-key": SOLTRACKER_DATA_API_KEY}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()

###############################################################################
# JSON-RPC to your Solana Tracker RPC
###############################################################################

def rpc_call(payload, max_retries=3):
    """
    Generic JSON-RPC call => uses SOLANATRACKER_RPC_URL with ?api_key=...
    """
    for attempt in range(max_retries):
        try:
            r = requests.post(SOLANATRACKER_RPC_URL, json=payload, timeout=10)
            if r.status_code == 200:
                return r.json()
            else:
                logger.warning(f"[rpc_call] => status={r.status_code}, attempt={attempt+1}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"[rpc_call] => {e}, attempt={attempt+1}")
    return None

def precheck_wallet(wallet):
    """
    1) Check if wallet older than 30 days => getSignaturesForAddress
    2) If younger => see if token count>10 => getTokenAccountsByOwner
    """
    creation_payload = {
        "jsonrpc":"2.0",
        "id":1,
        "method":"getSignaturesForAddress",
        "params":[
            wallet, {"limit":1}
        ]
    }
    c_resp = rpc_call(creation_payload)
    if not c_resp or "result" not in c_resp or not c_resp["result"]:
        logger.info("[precheck] => no tx => fail")
        return False

    first_tx = c_resp["result"][0]
    blk_time = first_tx.get("blockTime")
    if not blk_time:
        return False
    age_days = (time.time() - blk_time)/(24*3600)
    if age_days>=30:
        return True

    # If younger => check token accounts
    token_payload = {
        "jsonrpc":"2.0",
        "id":2,
        "method":"getTokenAccountsByOwner",
        "params":[
            wallet,
            {"programId":"TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
            {"encoding":"jsonParsed"}
        ]
    }
    t_resp = rpc_call(token_payload)
    if not t_resp or "result" not in t_resp or "value" not in t_resp["result"]:
        return False

    arr = t_resp["result"]["value"]
    return (len(arr)>10)

###############################################################################
# Tier Checker => using minted token approach
###############################################################################

class TokenTierChecker:
    def __init__(self):
        self.token_mint = os.getenv("TIER_TOKEN_MINT","2KchKijPuwnwC92LPWVjFjRwB3WxKtzx9bbXZ7kRpump")

    def get_token_balance(self, wallet):
        payload = {
            "jsonrpc":"2.0",
            "id":1,
            "method":"getTokenAccountsByOwner",
            "params":[
                wallet,
                {"mint": self.token_mint},
                {"encoding":"jsonParsed"}
            ]
        }
        r = rpc_call(payload)
        if not r or "result" not in r or "value" not in r["result"]:
            return 0.0
        total=0.0
        for acct in r["result"]["value"]:
            parsed = acct.get("account",{}).get("data",{}).get("parsed")
            if not parsed:
                continue
            tinfo = parsed.get("info",{})
            tAmt  = tinfo.get("tokenAmount",{})
            raw   = tAmt.get("amount","0")
            dec   = tAmt.get("decimals",0)
            if raw.isdigit():
                minted= float(raw)/(10**dec)
                total+= minted
        return total

    def get_tier(self, wallet):
        bal = self.get_token_balance(wallet)
        logger.info(f"[TierChecker] => wallet={wallet}, bal={bal}")
        tiers = {
            "Tier 3":{
                "border_color":"#00C2FF",
                "background":"linear-gradient(135deg, #00C2FF, #0066FF)",
                "effects":["basic-glow"]
            },
            "Tier 2":{
                "border_color":"#00FFA3",
                "background":"linear-gradient(135deg, #00C2FF, #00FFA3)",
                "effects":["premium-glow","hover-animation"]
            },
            "Tier 1":{
                "border_color":"#DC1FFF",
                "background":"linear-gradient(135deg, #00FFA3, #DC1FFF, #3308F5)",
                "effects":["premium-glow","hover-animation","sparkle"]
            }
        }
        if bal>=10000:
            tier="Tier 1"
        elif bal>=5000:
            tier="Tier 2"
        else:
            tier="Tier 3"
        return {
            "tier": tier,
            "balance": bal,
            "styles": tiers[tier]
        }

token_tier_checker= TokenTierChecker()

###############################################################################
# Minimal TokenManager => caching name/symbol/icon from /tokens/<addr>
###############################################################################
class TokenManager:
    def __init__(self):
        self.cache={}
        self.cache_duration= timedelta(hours=1)
        self.lock= threading.Lock()
        self.max_cache_size=1000
        self.cache_hits=0
        self.cache_misses=0
        self.db=db

    def _cleanup_cache(self):
        now= datetime.now(timezone.utc)
        expired=[]
        for k,v in self.cache.items():
            if now - v[1]>=self.cache_duration:
                expired.append(k)
        for k in expired:
            del self.cache[k]
        if len(self.cache)> self.max_cache_size:
            sorted_items= sorted(self.cache.items(), key=lambda x:x[1][1])
            over= len(self.cache)- self.max_cache_size
            for k,_ in sorted_items[:over]:
                del self.cache[k]

    def _check_db_cache(self, token_address):
        if not self.db:
            return None
        try:
            doc= self.db.collection("token_info").document(token_address).get()
            if doc.exists:
                d= doc.to_dict()
                if d.get("timestamp") and (time.time()-d["timestamp"])<DB_CACHE_DURATION:
                    return d.get("info")
            return None
        except Exception as e:
            logger.error(f"[DB check] => {e}")
            return None

    def _save_db_cache(self, token_address, token_info):
        if not self.db:
            return
        try:
            ref= self.db.collection("token_info").document(token_address)
            ref.set({
                "info": token_info,
                "timestamp": time.time()
            })
        except Exception as e:
            logger.error(f"[DB save] => {e}")

    def get_token_info(self, token_address):
        with self.lock:
            now= datetime.now(timezone.utc)
            if token_address in self.cache:
                info,tstamp= self.cache[token_address]
                if now - tstamp< self.cache_duration:
                    self.cache_hits+=1
                    return info

            # check DB
            db_info= self._check_db_cache(token_address)
            if db_info:
                self.cache[token_address]= (db_info, now)
                self.cache_hits+=1
                self._cleanup_cache()
                return db_info

            self.cache_misses+=1
            # fetch => /tokens/<addr>
            try:
                details= fetch_token_info_from_data(token_address)
                token_obj= details.get("token",{})
                if token_obj:
                    info_obj= {
                        "name": token_obj.get("name","Unknown"),
                        "symbol": token_obj.get("symbol","???").upper(),
                        "icon": token_obj.get("image","/default-token.png")
                    }
                else:
                    info_obj= {
                        "name": token_address[:8]+"...",
                        "symbol": token_address[:4].upper(),
                        "icon": "/default-token.png"
                    }
                self.cache[token_address]= (info_obj, now)
                self._save_db_cache(token_address, info_obj)
                self._cleanup_cache()
                return info_obj
            except Exception as e:
                logger.error(f"[TokenManager fetch] => {e}")
                # fallback
                return {
                    "name": token_address[:8]+"...",
                    "symbol": token_address[:4].upper(),
                    "icon": "/default-token.png"
                }

    def get_cache_stats(self):
        tot= self.cache_hits+ self.cache_misses
        ratio= self.cache_hits/tot if tot>0 else 0
        return {
            "cache_size": len(self.cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_ratio": ratio
        }

    def clear_cache(self):
        with self.lock:
            self.cache.clear()
            self.cache_hits=0
            self.cache_misses=0
            if self.db:
                try:
                    for d in self.db.collection("token_info").stream():
                        d.reference.delete()
                except Exception as e:
                    logger.error(f"[DB clear] => {e}")

token_manager=TokenManager()

###############################################################################
# Basic lifetime PnL aggregator => ignoring year
###############################################################################
def process_pnl_data(pnl_data):
    """
    Returns best_token, worst_token, top 5 profit, top 5 loss, top 5 volume 
    from the lifetime PnL endpoint.
    """
    tokens_map= pnl_data.get("tokens",{})
    all_tokens=[]
    for addr, details in tokens_map.items():
        if isinstance(details, dict):
            all_tokens.append({
                "address": addr,
                "pnl": float(details.get("total",0)),         # lifetime total
                "volume": float(details.get("total_sold",0)) # lifetime volume
            })

    if not all_tokens:
        return {
            "best_token": None,
            "worst_token": None,
            "top_5_profit_tokens": [],
            "top_5_loss_tokens": [],
            "top_5_highest_volume_tokens": []
        }

    # sort
    profit_tokens= [t for t in all_tokens if t["pnl"]>0]
    profit_tokens.sort(key=lambda x:x["pnl"], reverse=True)
    top_5_profit= profit_tokens[:5]

    loss_tokens= [t for t in all_tokens if t["pnl"]<0]
    loss_tokens.sort(key=lambda x:x["pnl"])  # ascending => negative first
    top_5_loss= loss_tokens[:5]

    volume_tokens= sorted(all_tokens, key=lambda x:x["volume"], reverse=True)
    top_5_vol= volume_tokens[:5]

    best= max(all_tokens, key=lambda x:x["pnl"], default=None)
    worst= min(all_tokens, key=lambda x:x["pnl"], default=None)

    # gather mint addresses
    needed= set()
    if best:
        needed.add(best["address"])
    if worst:
        needed.add(worst["address"])
    for arr in (top_5_profit, top_5_loss, top_5_vol):
        for t in arr:
            needed.add(t["address"])

    # fetch token info
    addr2info = {}
    for a in needed:
        i = token_manager.get_token_info(a)
        addr2info[a]= i

    def build_obj(t):
        info= addr2info[t["address"]]
        return {
            "address": t["address"],
            "name": info["name"],
            "symbol": info["symbol"],
            "pnl": t["pnl"],
            "volume": t["volume"],
            "icon": info["icon"]
        }

    return {
        "best_token": build_obj(best) if best else None,
        "worst_token": build_obj(worst) if worst else None,
        "top_5_profit_tokens": [build_obj(x) for x in top_5_profit],
        "top_5_loss_tokens": [build_obj(x) for x in top_5_loss],
        "top_5_highest_volume_tokens": [build_obj(x) for x in top_5_vol],
    }

###############################################################################
# FIFO-based approach => 2 passes for trades
###############################################################################
def add_to_cost_basis(cost_basis_map, mint, qty, cost_per_unit):
    """
    Add 'qty' tokens at 'cost_per_unit' into FIFO queue.
    """
    if mint not in cost_basis_map:
        cost_basis_map[mint] = []
    cost_basis_map[mint].append({
        'qty': qty,
        'cost_per_unit': cost_per_unit
    })

def remove_from_cost_basis(cost_basis_map, mint, qty):
    """
    Remove 'qty' from FIFO cost basis. Return total cost of that removal.
    If partial or lacking, remove as much as we can.
    """
    if mint not in cost_basis_map or not cost_basis_map[mint]:
        return 0.0
    queue = cost_basis_map[mint]
    remaining = qty
    total_cost= 0.0

    while remaining>1e-9 and queue:
        front= queue[0]
        have= front['qty']
        if have<= remaining+1e-9:
            total_cost+= have* front['cost_per_unit']
            remaining-= have
            queue.pop(0)
        else:
            total_cost+= remaining* front['cost_per_unit']
            front['qty']-= remaining
            remaining= 0.0
    return total_cost

def compute_fifo_pnl_for_2024(all_trades, base_syms):
    """
    - Pass 1: Build cost basis from trades *before* 2024
    - Pass 2: compute PnL from trades *in* 2024
    Returns a dictionary with:
      {
        'per_token_2024': { mint: {...} },
        'total_realized_2024': float,
        'total_volume_2024': float,
        'worst_coin_year': {...}  # min realized
      }
    """
    start_2024_ms = int(datetime(2024,1,1).timestamp() * 1000)
    cost_basis_map= {}

    # pass1 => fill cost basis from trades < 2024
    for tr in all_trades:
        tx_ms= tr.get("time",0)
        if tx_ms>= start_2024_ms:
            continue
        vol_usd= float(tr.get("volume",{}).get("usd",0.0))
        tx_type = (tr.get("type") or "").lower()
        from_sym= tr.get("from",{}).get("token",{}).get("symbol")
        to_sym  = tr.get("to",{}).get("token",{}).get("symbol")
        from_addr= tr.get("from",{}).get("address")
        to_addr  = tr.get("to",{}).get("address")
        qty_buy  = float(tr.get("to",{}).get("amount",0.0))
        qty_sell = float(tr.get("from",{}).get("amount",0.0))

        if tx_type=="buy" or (from_sym in base_syms):
            # buy
            if qty_buy>1e-9:
                cost_unit= vol_usd/ qty_buy if qty_buy>1e-9 else 0
                add_to_cost_basis(cost_basis_map, to_addr, qty_buy, cost_unit)
        elif tx_type=="sell" or (to_sym in base_syms):
            # sell
            cost_removed= remove_from_cost_basis(cost_basis_map, from_addr, qty_sell)
        else:
            # swap => optional advanced logic
            pass

    # pass2 => compute 2024 PnL
    results_map= {}  # { mint: { 'realized':..., 'volume':..., 'buys':..., 'sells':...} }
    total_real=0.0
    total_vol=0.0

    def init_stats(m):
        if m not in results_map:
            results_map[m]= {
                'realized': 0.0,
                'volume': 0.0,
                'buys': 0,
                'sells': 0
            }

    for tr in all_trades:
        tx_ms= tr.get("time",0)
        if tx_ms< start_2024_ms:
            continue
        vol_usd= float(tr.get("volume",{}).get("usd",0.0))
        tx_type = (tr.get("type") or "").lower()
        from_sym= tr.get("from",{}).get("token",{}).get("symbol")
        to_sym  = tr.get("to",{}).get("token",{}).get("symbol")
        from_addr= tr.get("from",{}).get("address")
        to_addr  = tr.get("to",{}).get("address")
        qty_buy  = float(tr.get("to",{}).get("amount",0.0))
        qty_sell = float(tr.get("from",{}).get("amount",0.0))

        if tx_type=="buy" or (from_sym in base_syms):
            init_stats(to_addr)
            results_map[to_addr]['buys']+=1
            results_map[to_addr]['volume']+= vol_usd
            total_vol+= vol_usd

            cost_unit= vol_usd/ qty_buy if qty_buy>1e-9 else 0
            add_to_cost_basis(cost_basis_map, to_addr, qty_buy, cost_unit)

        elif tx_type=="sell" or (to_sym in base_syms):
            init_stats(from_addr)
            results_map[from_addr]['sells']+=1
            results_map[from_addr]['volume']+= vol_usd
            total_vol+= vol_usd

            cost_removed= remove_from_cost_basis(cost_basis_map, from_addr, qty_sell)
            realized = vol_usd - cost_removed
            results_map[from_addr]['realized']+= realized
            total_real+= realized
        else:
            # a swap => partial approach
            pass

    # find worst => min realized
    worst_mint= None
    worst_val= math.inf
    for m, st in results_map.items():
        if st['realized']< worst_val:
            worst_val= st['realized']
            worst_mint= m

    out= {
        "per_token_2024": results_map,
        "total_realized_2024": total_real,
        "total_volume_2024": total_vol,
        "worst_coin_year": None  # we'll fill next
    }
    if worst_mint is not None:
        info= token_manager.get_token_info(worst_mint)
        out["worst_coin_year"]= {
            "address": worst_mint,
            "name": info["name"],
            "symbol": info["symbol"],
            "icon": info["icon"],
            "realized": results_map[worst_mint]['realized'],
            "volume": results_map[worst_mint]['volume']
        }
    return out

###############################################################################
# WalletManager => optional Firestore-based caching of final results
###############################################################################
class WalletManager:
    def __init__(self):
        self.db= db
        self.cache_duration= DB_CACHE_DURATION

    def get_wallet_data(self, wallet):
        if not self.db:
            return None
        try:
            doc= self.db.collection("wallet_data").document(wallet).get()
            if doc.exists:
                d= doc.to_dict()
                if d.get("timestamp") and (time.time()- d["timestamp"])< self.cache_duration:
                    return d.get("data")
            return None
        except Exception as e:
            logger.error(f"[WM get] => {e}")
            return None

    def save_wallet_data(self, wallet, data):
        if not self.db:
            return
        try:
            ref= self.db.collection("wallet_data").document(wallet)
            ref.set({
                "data": data,
                "timestamp": time.time()
            })
        except Exception as e:
            logger.error(f"[WM save] => {e}")

wallet_manager= WalletManager()

###############################################################################
# Flask Routes
###############################################################################

@app.route("/api/check-tier/<wallet>", methods=["GET"])
@limiter.limit("30 per minute")
def check_tier(wallet):
    """
    Example => check tier from minted token approach
    """
    try:
        tier= token_tier_checker.get_tier(wallet)
        return jsonify(tier)
    except Exception as e:
        logger.error(f"[check-tier] => {e}")
        return jsonify({"error":"Failed to check tier"}), 500

@app.route("/api/top-memecoins/<wallet>", methods=["GET"])
def top_memecoins(wallet):
    """
    1) Precheck => older than 30 days or #tokens>10
    2) fetch PNL => best/worst tokens lifetime
    3) fetch trades => do two-pass FIFO approach for 2024
    4) store => DB
    """
    logger.info(f"[top-memecoins] => {wallet}")
    wallet= wallet.strip()

    try:
        # 1) precheck
        if not precheck_wallet(wallet):
            return jsonify({
                "error": "Wallet does not meet requirements",
                "message": "Wallet must be older than 30 days or hold >10 tokens."
            }), 400

        # 2) DB cache
        cached= wallet_manager.get_wallet_data(wallet)
        if cached:
            return jsonify(cached)

        # 3) fetch PnL => build best/worst tokens lifetime
        pnl_data= fetch_pnl_data(wallet)
        # filter tokens => at least 10 tokens
        tokens_arr= [
            a for a,d in pnl_data.get("tokens",{}).items()
            if isinstance(d, dict) and (d.get("held",0)>0 or d.get("total_sold",0)>0)
        ]
        if len(tokens_arr)<10:
            return jsonify({
                "error": "No tokens bought",
                "message": "Wallet must have at least 10 tokens"
            }), 400

        # lifetime-based token data
        out= process_pnl_data(pnl_data)

        # Tier
        tier_info= token_tier_checker.get_tier(wallet)
        out["tier_info"]= tier_info

        # 4) fetch trades => compute 2024 PnL
        trades_data= fetch_wallet_trades(wallet)
        all_trades= trades_data.get("trades", [])
        # Simple approach => base_syms
        base_syms= {"SOL","USDC","USDT"}
        year_stats= compute_fifo_pnl_for_2024(all_trades, base_syms)
        out["total_realized_year"] = year_stats["total_realized_2024"]
        out["total_volume_year"]   = year_stats["total_volume_2024"]
        out["worst_coin_year"]     = year_stats["worst_coin_year"]
        # We could parse 'per_token_2024' if we want

        # Additionally, if you want total # buys/sells for 2024:
        # e.g. year_stats["per_token_2024"][some_mint]['buys']
        total_buys_2024=0
        total_sells_2024=0
        for st in year_stats["per_token_2024"].values():
            total_buys_2024+= st['buys']
            total_sells_2024+= st['sells']
        out["total_buys_year"]= total_buys_2024
        out["total_sells_year"]= total_sells_2024

        # 5) Save => DB
        wallet_manager.save_wallet_data(wallet, out)
        return jsonify(out)

    except requests.exceptions.RequestException as e:
        logger.error(f"[top-memecoins] => request error {e}")
        return jsonify({"error": "Failed to fetch from API"}),503
    except Exception as e:
        logger.error(f"[top-memecoins] => {e}", exc_info=True)
        return jsonify({"error": str(e)}),500

@app.route("/api/cache/stats", methods=["GET"])
def cache_stats():
    return jsonify(token_manager.get_cache_stats())

@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    try:
        token_manager.clear_cache()
        return jsonify({"message":"Cache cleared"})
    except Exception as e:
        logger.error(f"[clear_cache] => {e}")
        return jsonify({"error":str(e)}),500

###############################################################################
# Additional Example => /api/wallet-basic/<owner>
###############################################################################
@app.route("/api/wallet-basic/<owner>", methods=["GET"])
def get_wallet_basic_api(owner):
    try:
        data= fetch_wallet_basic(owner.strip())
        return jsonify(data)
    except requests.exceptions.RequestException as e:
        logger.error(f"[wallet-basic] => {e}")
        return jsonify({"error":"Fetch error"}),503
    except Exception as e:
        logger.error(f"[wallet-basic] => {e}", exc_info=True)
        return jsonify({"error":str(e)}),500

###############################################################################
# Additional Example => /api/token-info/<token_address>
###############################################################################
@app.route("/api/token-info/<token_address>", methods=["GET"])
def get_token_info_api(token_address):
    try:
        details= fetch_token_info_from_data(token_address.strip())
        return jsonify(details)
    except requests.exceptions.RequestException as e:
        logger.error(f"[token-info] => {e}")
        return jsonify({"error":"Fetch error"}),503
    except Exception as e:
        logger.error(f"[token-info] => {e}", exc_info=True)
        return jsonify({"error":str(e)}),500

###############################################################################
# SHARE blueprint => example
###############################################################################
share_bp = Blueprint('share', __name__)

@share_bp.route('/api/share', methods=['POST'])
def create_share():
    try:
        s_id= str(uuid.uuid4())[:8]
        dat= request.get_json()
        share_data= {
            "data": dat,
            "tier_info": dat.get("tier_info"),
            "created_at": datetime.utcnow().isoformat(),
            "wallet": dat.get("wallet","anonymous")
        }
        if db:
            db.collection("shares").document(s_id).set(share_data)
        return jsonify({
            "success":True,
            "share_id": s_id
        })
    except Exception as e:
        return jsonify({"error":str(e)}),500

@share_bp.route("/api/share/<share_id>", methods=["GET"])
def get_share(share_id):
    try:
        if not db:
            return jsonify({"error":"DB not configured"}),500
        doc= db.collection("shares").document(share_id).get()
        if not doc.exists:
            return jsonify({"error":"Not found"}),404
        return jsonify({"success":True,"data": doc.to_dict()})
    except Exception as e:
        return jsonify({"error":str(e)}),500

app.register_blueprint(share_bp)

###############################################################################
# MAIN
###############################################################################
if __name__=="__main__":
    logger.info("[MAIN] Starting updated Flask app with FIFO approach to 2024 PnL.")
    app.run(debug=True, host="0.0.0.0", port=5000)
