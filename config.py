"""
Smart Trading Framework – Configuration
"""

# ── Timeframes ────────────────────────────────────────────────────────────────
BIAS_TF   = "4h"   # Daily / 4H bias timeframe (higher-timeframe)
TIMING_TF = "1h"   # 1H trigger timeframe
ENTRY_TF  = "15m"  # Entry/refinement timeframe

# ── Market Structure ──────────────────────────────────────────────────────────
SWING_LOOKBACK      = 10    # bars either side of pivot for swing detection
STRUCTURE_LOOKBACK  = 50    # bars to look back for BOS/CHoCH confirmation

# ── Order Blocks ──────────────────────────────────────────────────────────────
OB_LOOKBACK         = 15    # bars to search before a BOS for the originating OB
OB_MITIGATION_PCT   = 0.50  # OB considered mitigated when price reaches 50% of zone

# ── Fair Value Gaps ───────────────────────────────────────────────────────────
FVG_MIN_SIZE_PCT    = 0.05  # Minimum FVG size as % of price

# ── Liquidity ─────────────────────────────────────────────────────────────────
LIQ_LOOKBACK        = 20    # swing length for liquidity pivots
EQ_LEVEL_THRESH_PCT = 0.10  # Equal high/low threshold (%) for EQH/EQL detection
MAX_LIQ_LEVELS      = 6     # Maximum liquidity levels to track at once

# ── Premium / Discount ────────────────────────────────────────────────────────
PD_RANGE_LENGTH     = 50    # bars to measure range for P/D zone
PD_PREMIUM_PCT      = 0.75  # above this → premium  (top 25% of range)
PD_DISCOUNT_PCT     = 0.25  # below this → discount  (bottom 25% of range)

# ── Order Flow ────────────────────────────────────────────────────────────────
CVD_PERIOD          = 14    # period for CVD slope calculation
ABSORPTION_VOL_MULT = 1.5   # volume multiplier to qualify as absorption
ABSORPTION_BODY_MAX = 0.5   # max body size (as fraction of avg) for absorption
IMBALANCE_BODY_MULT = 2.0   # body multiplier for imbalance candle

# ── Institutional Phase ───────────────────────────────────────────────────────
PHASE_LOOKBACK      = 50    # bars for volatility / phase detection
ACCUM_VOL_THRESH    = 0.70  # stddev ratio below which volatility is "low"
DISTRIB_VOL_THRESH  = 1.30  # stddev ratio above which volatility is "high"

# ── Setup Scoring ─────────────────────────────────────────────────────────────
MIN_LONG_SCORE      = 4     # minimum score (out of 7) to fire long signal
MIN_SHORT_SCORE     = 4     # minimum score (out of 7) to fire short signal

# ── Data Fetching ─────────────────────────────────────────────────────────────
DATA_SOURCE         = "yfinance"   # "yfinance" | "ccxt"
DEFAULT_SYMBOL      = "BTC-USD"
DEFAULT_PERIOD      = "60d"        # yfinance period string
DEFAULT_INTERVAL    = "1h"         # yfinance interval string
