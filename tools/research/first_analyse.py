"""
Quick factor screening — fetch a panel once, compute every candidate TA-Lib
factor in memory, backtest each one, and keep only those whose Q1→Q{bins}
mean net returns are perfectly monotonic (ascending or descending).

No intermediate files are written — the panel lives in memory only.
Monotonic factors are written to results/first_analyse_factors.yaml.

Edit the globals below to configure the run.

Usage:
    python tools/research/first_analyse.py
"""

import os
import sys

import duckdb
import pandas as pd
import yaml

# Reuse the TA-Lib factor registry maintained in tools/process_data/pre_factor.py
_PROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "process_data")
sys.path.insert(0, os.path.abspath(_PROCESS_DIR))
from pre_factor import ALL_FACTOR_DEFS, compute_talib_factors  # noqa: E402

from alphapurify import AlphaPurifier, FactorAnalyzer  # noqa: E402
from alphapurify.FactorAnalyzer import ResearchConfig, AnalysisConfig  # noqa: E402

# ============================================================================
# Global configuration — edit these
# ============================================================================

# Stock universe
CODES = [
    "sh.600000", "sh.601799", "sh.600008", "sh.600009", "sh.600010",
    "sh.600011", "sh.600015", "sh.600016", "sh.600018", "sh.600019",
    "sh.600021", "sh.600023", "sh.600025", "sh.600026", "sh.600027",
    "sh.600028", "sh.600030", "sh.600031", "sh.600032", "sh.600036",
    "sh.600038", "sh.600039", "sh.600048", "sh.600050", "sh.600060",
    "sh.600061", "sh.600062", "sh.600066", "sh.600085", "sh.600089",
    "sh.600095", "sh.600098", "sh.600100", "sh.600104", "sh.600105",
    "sh.600109", "sh.600111", "sh.600115", "sh.600126", "sh.600131",
    "sh.600132", "sh.600141", "sh.600143", "sh.600150", "sh.600153",
    "sh.600157", "sh.600160", "sh.600161", "sh.600166", "sh.600170",
    "sh.600171", "sh.600177", "sh.600188", "sh.600196", "sh.600208",
    "sh.600219", "sh.600221", "sh.600233", "sh.600256", "sh.600276",
    "sh.600282", "sh.600292", "sh.600295", "sh.600298", "sh.600299",
    "sh.600312", "sh.600329", "sh.600332", "sh.600339", "sh.600346",
    "sh.600348", "sh.600350", "sh.600352", "sh.600362", "sh.600363",
    "sh.600369", "sh.600372", "sh.600377", "sh.600380", "sh.600390",
    "sh.600392", "sh.600398", "sh.600406", "sh.600415", "sh.600426",
    "sh.600435", "sh.600438", "sh.600460", "sh.600482", "sh.600483",
    "sh.600486", "sh.600489", "sh.600497", "sh.600499", "sh.600511",
    "sh.600515", "sh.600516", "sh.600517", "sh.600521", "sh.600522",
    "sh.600535", "sh.600536", "sh.600546", "sh.600547", "sh.600562",
    "sh.600566", "sh.600570", "sh.600578", "sh.600582", "sh.600583",
    "sh.600585", "sh.600588", "sh.600595", "sh.600598", "sh.600600",
    "sh.600601", "sh.600602", "sh.600606", "sh.600637", "sh.600642",
    "sh.600655", "sh.600660", "sh.600663", "sh.600674", "sh.600685",
    "sh.600688", "sh.600690", "sh.600699", "sh.600704", "sh.600707",
    "sh.600711", "sh.600737", "sh.600741", "sh.600754", "sh.600760",
    "sh.600763", "sh.600764", "sh.600765", "sh.600795", "sh.600801",
    "sh.600803", "sh.600808", "sh.600816", "sh.600820", "sh.600845",
    "sh.600848", "sh.600862", "sh.600863", "sh.600871", "sh.600873",
    "sh.600875", "sh.600879", "sh.600884", "sh.600885", "sh.600886",
    "sh.600887", "sh.600893", "sh.600900", "sh.600901", "sh.600905",
    "sh.600906", "sh.600909", "sh.600918", "sh.600919", "sh.600926",
    "sh.600927", "sh.600930", "sh.600938", "sh.600958", "sh.600967",
    "sh.600968", "sh.600970", "sh.600977", "sh.600985", "sh.600988",
    "sh.600989", "sh.600995", "sh.600998", "sh.600999", "sh.601000",
    "sh.601001", "sh.601006", "sh.601009", "sh.601012", "sh.601016",
    "sh.601018", "sh.601019", "sh.601058", "sh.601059", "sh.601066",
    "sh.601077", "sh.601088", "sh.601098", "sh.601099", "sh.601106",
    "sh.601108", "sh.601111", "sh.601112", "sh.601117", "sh.601118",
    "sh.601128", "sh.601136", "sh.601139", "sh.601155", "sh.601156",
    "sh.601162", "sh.601166", "sh.601169", "sh.601179", "sh.601186",
    "sh.601198", "sh.601211", "sh.601212", "sh.601216", "sh.601225",
    "sh.601228", "sh.601229", "sh.601233", "sh.601236", "sh.601238",
    "sh.601288", "sh.601298", "sh.601318", "sh.601319", "sh.601328",
    "sh.601360", "sh.601377", "sh.601390", "sh.601398", "sh.601399",
    "sh.601456", "sh.601555", "sh.601567", "sh.601577", "sh.601598",
    "sh.601600", "sh.601601", "sh.601607", "sh.601608", "sh.601611",
    "sh.601615", "sh.601618", "sh.601628", "sh.601633", "sh.601658",
    "sh.601665", "sh.601666", "sh.601668", "sh.601669", "sh.601688",
    "sh.601689", "sh.601696", "sh.601698", "sh.601699", "sh.601717",
    "sh.601727", "sh.601728", "sh.601766", "sh.601788", "sh.601800",
    "sh.601808", "sh.601816", "sh.601818", "sh.601825", "sh.601838",
    "sh.601857", "sh.601865", "sh.601866", "sh.601868", "sh.601872",
    "sh.601877", "sh.601878", "sh.601880", "sh.601881", "sh.601888",
    "sh.601898", "sh.601899", "sh.601901", "sh.601916", "sh.601919",
    "sh.601928", "sh.601939", "sh.601958", "sh.601966", "sh.601985",
    "sh.601988", "sh.601990", "sh.601991", "sh.601995", "sh.601997",
    "sh.601998", "sh.603000", "sh.603049", "sh.603077", "sh.603087",
    "sh.603092", "sh.603156", "sh.603179", "sh.603225", "sh.603233",
    "sh.603260", "sh.603288", "sh.603298", "sh.603308", "sh.603338",
    "sh.603341", "sh.603369", "sh.603392", "sh.603486", "sh.603529",
    "sh.603565", "sh.603568", "sh.603589", "sh.603596", "sh.603605",
    "sh.603606", "sh.603658", "sh.603659", "sh.603699", "sh.603737",
    "sh.603766", "sh.603786", "sh.603799", "sh.603806", "sh.603816",
    "sh.603833", "sh.603858", "sh.603899", "sh.603920", "sh.603939",
    "sh.603993", "sh.605589", "sz.000001", "sz.000002", "sz.000009",
    "sz.000021", "sz.000027", "sz.000032", "sz.000034", "sz.000039",
    "sz.000050", "sz.000060", "sz.000062", "sz.000063", "sz.000088",
    "sz.000100", "sz.000155", "sz.000157", "sz.000166", "sz.000301",
    "sz.000338", "sz.000400", "sz.000415", "sz.000423", "sz.000425",
    "sz.000429", "sz.000513", "sz.000519", "sz.000528", "sz.000537",
    "sz.000538", "sz.000539", "sz.000559", "sz.000582", "sz.000591",
    "sz.000598", "sz.000617", "sz.000623", "sz.000625", "sz.000629",
    "sz.000630", "sz.000651", "sz.000683", "sz.000703", "sz.000708",
    "sz.000709", "sz.000723", "sz.000725", "sz.000728", "sz.000729",
    "sz.000733", "sz.000737", "sz.000738", "sz.000739", "sz.000750",
    "sz.000768", "sz.000776", "sz.000783", "sz.000785", "sz.000786",
    "sz.000792", "sz.000800", "sz.000807", "sz.000825", "sz.000830",
    "sz.000831", "sz.000878", "sz.000883", "sz.000887", "sz.000893",
    "sz.000895", "sz.000898", "sz.000921", "sz.000932", "sz.000937",
    "sz.000938", "sz.000951", "sz.000959", "sz.000960", "sz.000963",
    "sz.000967", "sz.000975", "sz.000983", "sz.000987", "sz.000997",
    "sz.000999", "sz.001203", "sz.001221", "sz.001286", "sz.001386",
    "sz.001391", "sz.001696", "sz.001965", "sz.001979", "sz.002001",
    "sz.002007", "sz.002027", "sz.002032", "sz.002044", "sz.002050",
    "sz.002056", "sz.002064", "sz.002065", "sz.002074", "sz.002078",
    "sz.002085", "sz.002120", "sz.002126", "sz.002130", "sz.002131",
    "sz.002142", "sz.002152", "sz.002153", "sz.002155", "sz.002157",
    "sz.002179", "sz.002185", "sz.002195", "sz.002202", "sz.002203",
    "sz.002223", "sz.002230", "sz.002236", "sz.002241", "sz.002244",
    "sz.002252", "sz.002261", "sz.002262", "sz.002265", "sz.002266",
    "sz.002271", "sz.002273", "sz.002299", "sz.002304", "sz.002311",
    "sz.002312", "sz.002318", "sz.002335", "sz.002340", "sz.002352",
    "sz.002402", "sz.002407", "sz.002410", "sz.002414", "sz.002415",
    "sz.002422", "sz.002423", "sz.002429", "sz.002430", "sz.002436",
    "sz.002444", "sz.002461", "sz.002465", "sz.002472", "sz.002487",
    "sz.002493", "sz.002500", "sz.002508", "sz.002517", "sz.002532",
    "sz.002558", "sz.002568", "sz.002583", "sz.002600", "sz.002601",
    "sz.002602", "sz.002603", "sz.002608", "sz.002624", "sz.002625",
    "sz.002648", "sz.002670", "sz.002673", "sz.002683", "sz.002709",
    "sz.002714", "sz.002736", "sz.002739", "sz.002773", "sz.002797",
    "sz.002831", "sz.002841", "sz.002926", "sz.002939", "sz.002945",
    "sz.002966", "sz.002984", "sz.003022", "sz.003035", "sz.003816",
]

# Date range (inclusive)
START_DATE = "2025-07-01"
END_DATE = "2026-07-02"

# Candidate factors to screen — default: every TA-Lib factor in
# pre_factor.ALL_FACTOR_DEFS. Trim this list to focus on a subset.
FACTORS = [name for name, _, _ in ALL_FACTOR_DEFS]

# Per-factor preprocessing — set either to None to skip that step
WINSORIZE = "mad"
STANDARDIZE = "zscore"

# Backtest hyperparameters
REBALANCE_PERIODS = ("W",)        # one period is enough for fast screening
RETURN_HORIZONS = (5,)            # IC horizons — kept minimal for speed
BINS = 5
WARMUP_DAYS = 60                  # calendar days pre-fetched for TA-Lib seeding

# Paths
DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "quant.db")
)
_TABLE = "stock_kline_d"

OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "results", "first_analyse_factors.yaml")
)

# ============================================================================


def fetch_panel(db_path: str, codes, start: str, end: str, warmup_days: int) -> pd.DataFrame:
    """Query OHLCV from the local DuckDB kline table."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")
    if not codes:
        raise ValueError("'codes' is empty.")

    fetch_start = start
    if warmup_days > 0:
        fetch_start = (
            pd.to_datetime(start) - pd.Timedelta(days=warmup_days)
        ).strftime("%Y-%m-%d")

    codes_literal = ", ".join(f"'{c}'" for c in codes)
    query = f"""
        SELECT code, date, open, high, low, close, volume
        FROM {_TABLE}
        WHERE code IN ({codes_literal})
          AND date >= DATE '{fetch_start}'
          AND date <= DATE '{end}'
        ORDER BY code, date
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(query).fetchdf()
    finally:
        con.close()

    if df.empty:
        raise ValueError(
            f"No rows returned. Check codes (e.g. {codes[:3]}) and date range "
            f"[{fetch_start}, {end}] against table '{_TABLE}'."
        )

    df = df.rename(columns={"code": "symbol", "date": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def compute_factors(df: pd.DataFrame, factor_names: list[str]):
    """Apply TA-Lib factors per symbol. Returns (df, produced_cols)."""
    name_to_idx = {n: i for i, (n, _, _) in enumerate(ALL_FACTOR_DEFS)}
    indices = sorted({name_to_idx[n] for n in factor_names if n in name_to_idx})
    parts = []
    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values("datetime")
        parts.append(compute_talib_factors(g, indices))
    out = pd.concat(parts, ignore_index=True)
    produced = [ALL_FACTOR_DEFS[i][0] for i in indices]
    return out, produced


def trim_warmup(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    return df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)].reset_index(drop=True)


def is_strictly_monotonic(seq: list[float]) -> tuple[bool, str]:
    """Strict monotonicity check — ascending OR descending both count."""
    if len(seq) < 2:
        return False, "none"
    inc = all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
    dec = all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
    if inc:
        return True, "ascending"
    if dec:
        return True, "descending"
    return False, "none"


def backtest_factor(panel: pd.DataFrame, factor_name: str) -> FactorAnalyzer:
    """Run AlphaPurifier preprocessing + FactorAnalyzer backtest in memory."""
    pre = AlphaPurifier(
        panel, factor_name=factor_name, trade_date_col="datetime", symbol_col="symbol"
    )
    if WINSORIZE:
        pre = pre.winsorize(method=WINSORIZE)
    if STANDARDIZE:
        pre = pre.standardize(method=STANDARDIZE)
    proc = pre.to_result()

    research_cfg = ResearchConfig()
    research_cfg.rebalance_periods = REBALANCE_PERIODS
    research_cfg.return_horizons = RETURN_HORIZONS
    research_cfg.bins = BINS
    fa = FactorAnalyzer(
        base_df=proc,
        trade_date_col="datetime",
        symbol_col="symbol",
        price_col="close",
        factor_name=factor_name,
        research_cfg=research_cfg,
        analysis_cfg=AnalysisConfig(),
    )
    fa.run()
    return fa


def extract_quantile_means(fa: FactorAnalyzer) -> dict:
    """Per rebalance period: mean of ret_net_q columns as a list, Q1…Q{bins}."""
    out = {}
    for period, df_res in fa.returns_dict.items():
        cols = [f"ret_net_{q}" for q in range(1, fa.bins + 1)]
        if any(c not in df_res.columns for c in cols):
            continue
        out[period] = [float(df_res[c].mean()) for c in cols]
    return out


def extract_ls_stats(fa: FactorAnalyzer) -> dict:
    """Pull Long-Short summary stats per period from ls_stats_panel."""
    out = {}
    panel = fa.ls_stats_panel
    if panel is None or len(panel) == 0:
        return out
    for _, row in panel.iterrows():
        p = row.get("period")
        out[p] = {
            "ls_ann_return": float(row.get("Ann. Return", float("nan"))),
            "ls_ann_sharpe": float(row.get("Ann. Sharpe", float("nan"))),
            "ls_max_drawdown": float(row.get("Max Drawdown", float("nan"))),
        }
    return out


def main():
    print(f"[screen] symbols: {len(CODES)}  range: [{START_DATE}, {END_DATE}]")
    print(f"[screen] factors: {len(FACTORS)}  bins: {BINS}  rebalance: {REBALANCE_PERIODS}")

    df = fetch_panel(DB_PATH, CODES, START_DATE, END_DATE, WARMUP_DAYS)
    print(f"[screen] panel rows={len(df)}  (incl. {WARMUP_DAYS}d warmup)")

    df, produced = compute_factors(df, FACTORS)
    df = trim_warmup(df, START_DATE, END_DATE)
    print(f"[screen] after warmup trim: rows={len(df)}  factor cols={len(produced)}")

    usable = [f for f in FACTORS if f in df.columns]
    dropped = [f for f in FACTORS if f not in df.columns]
    if dropped:
        print(f"[screen] WARNING: factors not produced (skipped): {dropped}")
    print(f"[screen] backtesting {len(usable)} factors ...\n")

    results = []
    for i, factor_name in enumerate(usable, 1):
        try:
            fa = backtest_factor(df, factor_name)
        except Exception as e:  # per-factor failure should not abort the screen
            print(f"[screen] ({i}/{len(usable)}) {factor_name:<22} ERROR: {e}")
            continue

        q_means = extract_quantile_means(fa)
        ls_stats = extract_ls_stats(fa)

        monotonic_periods = []
        for period, means in q_means.items():
            ok, direction = is_strictly_monotonic(means)
            if not ok:
                continue
            monotonic_periods.append({
                "period": str(period),
                "direction": direction,
                "quantile_means": {f"Q{q + 1}": means[q] for q in range(len(means))},
                **ls_stats.get(period, {}),
            })

        tag = f"MONO[{','.join(p['period'] for p in monotonic_periods)}]" if monotonic_periods else "----"
        print(f"[screen] ({i}/{len(usable)}) {factor_name:<22} {tag}")

        if monotonic_periods:
            results.append({"factor": factor_name, "periods": monotonic_periods})

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    payload = {
        "meta": {
            "codes": CODES,
            "start": START_DATE,
            "end": END_DATE,
            "bins": BINS,
            "rebalance_periods": list(REBALANCE_PERIODS),
            "return_horizons": list(RETURN_HORIZONS),
            "winsorize": WINSORIZE,
            "standardize": STANDARDIZE,
            "n_screened": len(usable),
            "n_monotonic": len(results),
        },
        "factors": results,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    print(f"\n[screen] done. {len(results)}/{len(usable)} monotonic -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
