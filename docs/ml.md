# ML 多因子合成（`ml/`）

用 LightGBM 在滚动窗口（walk-forward）上把 59 个 TA-Lib 因子合成为一个 `alpha_composite`，输出可以喂给 `tools/research/backtest.py` 做单因子回测。

> 这套流程存在的根本原因：AlphaPurify 本身是**单因子**框架（`FactorAnalyzer` 一次只看一列 alpha）。要做"多因子合成"，要么 IC 加权、要么 ML。IC 加权如果在全样本上算 IC，会泄漏未来；ML 配合 walk-forward 可以把"用历史选因子权重"这件事严格限制在每个测试月之前。

---

## 1. 整体流程

```
              ┌─ TA-Lib 59 个因子（在 fetch_data.py 里即时算）
              │
全因子 panel ─┼─ 基本面 11 个因子（来自 stock_fundamental，PIT via pub_date）
(all_factors  │
.parquet)     ├─ 估值 3 个因子 peTTM/pb/psTTM（来自 stock_valuation_d）
              │
              └─ 行业列（来自 stock_industry，用于中性化，不进模型）
        │
        │  fetch_data.py 拼 panel：PIT merge + valuation join + industry join
        ▼
┌──────────────────────────────────────────────┐
│  ml/run.py                                    │
│  ├─ attach_forward_return(horizon=5)         │  ← 给每行加 fut_ret_5 标签
│  ├─ for each test month (walk-forward):      │
│  │   ├─ train = panel[start .. test-5d-1]    │  ← expanding，purge 5 个交易日
│  │   ├─ test  = panel[test+5d .. 月末]       │  ← embargo 5 个交易日
│  │   ├─ fit winsorize → 行业去均值 → zscore  │  ← 三步预处理，参数仅在 train 上 fit
│  │   ├─ fit LightGBM (full train, 固定 100 棵)│  ← 无 holdout，无 early stopping
│  │   └─ predict on test → 一列 pred          │
│  └─ concat 所有月的 pred → oos_preds         │
└──────────────────────────────────────────────┘
        │
        │  save_oas_parquet
        ▼
results/ml_oos.parquet  (datetime, symbol, close, alpha_composite)
        │
        │  backtest.py --config data/ml_composite.yaml
        ▼
results/ml_report/backtest/*.html
```

输入：`data/all_factors.parquet`（520 票 × ~2.5 年 × **73 因子** + 行业列）。
输出：`results/ml_oos.parquet` + 5 份 HTML 回测报告。

---

## 2. 核心原理（防泄漏的几个关键点）

### 2.1 Walk-forward + Expanding window

按月推进测试集。每个月用**所有更早的数据**训练（expanding，不是 rolling），所以训练量随时间增长。第一个测试月要求至少有 6 个月训练数据（`min_train_months=6`），太早的月直接跳过。

### 2.2 Purge + Embargo（用**交易日**算，不是自然日）

标签是"未来 5 日收益"——它会偷看未来 5 个 bar。如果 train 的最后一行和 test 的第一行只隔一个周末，那么 train 最后一行的标签实际上**包含了 test 第一天的收益**，这就是泄漏。

- **Purge**：train 的最后一行 = 测试月第一个交易日往前数第 `purge_days+1` 个交易日。这样 train 末行的 5 日 forward return 一定在测试月开始之前结算完。
- **Embargo**：test 的第一行 = 测试月第一个交易日往后数 `embargo_days` 个交易日。即便训练用了某些跨越到 test 头部的标签，也把它们从 test 里剔掉。

> 一定要用**交易日**而不是自然日。`pandas.Timedelta(days=5)` 跨周末时只覆盖 3 个交易日，标签还是会泄漏。`generate_walkforward_splits` 用的是 `trading_days` 列表索引，保证 5 个真实交易日。

### 2.3 每个窗口独立预处理（参数 fit 在 train 上）

每个测试月单独算一次 winsorize（MAD，k=5）阈值和 zscore 均值方差，**只用 train 切片**算，然后 transform 到 train 和 test。绝不用全局 panel 统计量——那会泄漏 test 的分布信息。

### 2.4 不用 holdout，固定 n_estimators

LightGBM 默认 100 棵树（`n_estimators=100`），不分内部 holdout、不开 early stopping。原因：

- 数据有限，holdout 那 20% 训练数据要么拿来建树要么浪费；
- early stopping 的作用是"挑停止点"，但月度 walk-forward 本身就是更可靠的 OOS 评估——直接用 OOS IC 反推超参比 holdout RMSE 更接近最终目标；
- 想调 `n_estimators`：sweep [50, 100, 200]，看哪个 OOS mean IC 更高、t-stat 更显著。

### 2.5 行业中性化（必做）

A 股行业间基本面差异极大（银行 ROE 11% 算低，软件 11% 算高；地产负债率 80% 正常，医药 50% 都嫌高）。**不中性化等于让模型先学"行业差异"**，而不是"行业内谁更优秀"——会浪费一大半信号容量。

中性化在 `ml/dataset.py` 的 `transform` 里做，顺序：**winsorize → 行业去均值 → zscore**。

实现：
- `fit_preprocessor` 在 train 切片上算每个 (date, industry) 的因子均值
- 然后 groupby industry 再取均值，得到每个 (industry, factor) 的单一均值（跨 train 日期平均，防 date-specific leak）
- `transform` 时，每行用其 industry 的均值减一次

如果 panel 里没有 `industry` 列，中性化自动跳过（向后兼容）。

### 2.6 Label = forward 5 日收益（回归）

直接做回归，目标是 `close[t+5]/close[t] - 1`。比分类（涨/跌）保留更多信息，也跟 IC、quantile spread 的评估口径一致。

### 2.7 PIT merge：用 pub_date 不用 report_date

季度基本面是季频，价格是日频。要把季度值"贴"到日频行上，必须做 forward-fill。**关键：用 `pub_date`（披露日）而不是 `report_date`（报告期）**。

年报最晚 4/30 披露，但报告期是去年 12/31。如果用 `report_date` 做 join，1/1 ~ 4/30 这段时间的"基本面"实际上是当时还没人知道的未来数据——经典 look-ahead。

`tools/research/fetch_data.py:_pit_merge_fundamentals` 用 `pd.merge_asof(date, pub_date, direction='backward')`，保证每行只能看到当时已披露的最新季报。

---

## 3. 代码结构

| 文件 | 作用 |
|---|---|
| `tools/fetch_data/get_fundamental.py` | 拉季度基本面（`fetch_fundamental`）+ 日频估值（`fetch_valuation`），BaostockClient 限流，断点续拉 |
| `tools/research/fetch_data.py` | 拼 panel：OHLCV + TA-Lib + 基本面（PIT merge）+ 估值（直接 join）+ 行业 |
| `ml/dataset.py` | 加载 parquet、attach forward return、walk-forward 切分、per-window winsorize+中性化+zscore |
| `ml/model.py` | LightGBM 训练 / 预测。默认浅树（`max_depth=3, num_leaves=8, n_estimators=100`）+ L2 正则 |
| `ml/walkforward.py` | 主循环。每月切窗口、训练、预测、记录 per-date IC |
| `ml/evaluate.py` | 汇总 OOS pred → mean IC、t-stat、IR、quantile means、写 parquet |
| `ml/run.py` | CLI 入口，串起上面的步骤 |

### LightGBM 默认超参（`default_lgbm_params`）

```python
max_depth=3, num_leaves=8          # 浅树，短面板深了立刻过拟合
learning_rate=0.05, n_estimators=100   # 固定 100 棵，无 early stopping
reg_alpha=0.1, reg_lambda=1.0      # L2 为主
min_child_samples=200              # 叶子最小样本数，再抗过拟合
feature_fraction=0.8, bagging_fraction=0.8
```

调参必须有 OOS 证据，不要只因为训练集 IC 好看就加深。

---

## 4. 使用方法

### 4.0 拉数据（一次性）

确保 DuckDB 里有 OHLCV、基本面、估值、行业四类数据：

```bash
# 1. OHLCV + 行业（已有，跳过；首次拉看 tools/fetch_data/get_stocks_from_index.py）

# 2. 季度基本面（已有 10 年 × 3189 票；要补新季度跑这条）
python tools/fetch_data/get_fundamental.py --fundamental

# 3. 日频估值（peTTM/pb/psTTM/isST，新拉，~520 calls）
python tools/fetch_data/get_fundamental.py --valuation --start 2024-01-01
```

估值每次约 520 API calls，远低于 baostock 45000/天上限。BaostockClient 自带 0.1s 间隔 + 全局锁 + 日计数器 + 重试，不会撞限流。

### 4.1 准备全因子 panel

写一个 yaml（参考 `data/all_factors.yaml`），关键配置：

```yaml
codes: [sh.600000, sz.000001, ...]      # 520 票
date:
  - start: '2024-01-01'
  - end: '2026-07-03'

include_fundamentals: true               # 拉 stock_fundamental（11 个基本面）
include_valuation: true                  # 拉 stock_valuation_d（3 个估值）
drop_st: true                            # stock_kline_d 已在 fetch 时过滤 isST=1，这里基本是 no-op

alpha:                                    # 只列 TA-Lib 因子名（59 个）
  - sma
  - macd
  # ... 完整 59 个见 data/all_factors.yaml
```

跑：

```bash
python tools/research/fetch_data.py \
    --config data/all_factors.yaml \
    --save data/all_factors.parquet \
    --warmup 60
```

`--warmup 60` 让 TA-Lib 在 `start` 之前预热 60 个自然日，避免头几周因子值是 NaN。

输出列：`datetime, symbol, close, volume, <59 TA-Lib>, <11 基本面>, <3 估值>, industry`（共 76 列）。

### 4.2 跑 ML walk-forward 合成

```bash
python -m ml.run \
    --panel data/all_factors.parquet \
    --save-oas results/ml_oos.parquet \
    --save-report results/ml_report \
    --horizon 5 \
    --min-train-months 6 \
    --purge-days 5 \
    --embargo-days 5
```

控制台会逐月打印：

```
[wf] (1/24) 2024-07  n_train= 60000  n_test=  9600  OOS_IC=-0.0421
[wf] (2/24) 2024-08  n_train= 69600  n_test=  8800  OOS_IC=+0.0185
...
```

2.5 年 ≈ 30 个月，`min_train_months=6` 之后还有 24 个 OOS 月，足够算 t-stat 和 IR。

结束后打印汇总：

```
======================================================
OOS Summary (walk-forward, expanding window, purge+embargo)
======================================================
  observations:     200000
  test dates:          480
  mean IC:        -0.0365
  std IC:          0.1100
  IR:             -0.3318
  t-stat:         -2.1000
  p-value:          0.037
  pooled spearman: -0.0412

  quantile means (avg fut_ret per bin):
     Q1: +0.00120
     Q2: +0.00050
     Q3: -0.00010
     Q4: -0.00080
     Q5: -0.00150

Per-month IC:
test_month  n_train  n_test  mean_ic
  2024-07    60000    9600  -0.0421
  2024-08    69600    8800  +0.0185
  ...
```

文件输出：
- `results/ml_oos.parquet` — `datetime, symbol, close, alpha_composite`，可以直接喂给 `backtest.py`
- `results/ml_report/month_log.csv` — 每月的 n_train / n_test / mean_ic
- `results/ml_report/stats.txt` — 上面那段汇总的文本版

### 4.4 喂给回测引擎

```bash
python tools/research/backtest.py \
    --data results/ml_oos.parquet \
    --save results/ml_report/backtest \
    --config data/ml_composite.yaml \
    --factor alpha_composite
```

`data/ml_composite.yaml` 长这样（**注意 winsorize / standardize 留空**，因为合成的时候已经处理过了，再做一次会过度压缩）：

```yaml
alpha:
  - alpha_composite

winsorize: []
standardize: []

trace:
  rebalance_period: W
  dates:
    - "2026-06-12"   # 必须落在 ml_oos.parquet 的时间范围内
```

输出 5 份 HTML：
- `single_fac_ic.html` — IC 时间序列、t-stat、IR
- `long_return.html` / `short_return.html` / `long_short_return.html` — 多空分组累计收益
- `trace_<date>.html` — 该日截面权重/收益快照

---

## 5. 怎么读结果

### 5.1 OOS Summary

| 指标 | 含义 | 判断标准 |
|---|---|---|
| `mean IC` | 月度截面 spearman IC 的均值 | 绝对值 ≥ 0.02 才有点用，≥ 0.05 算不错 |
| `std IC` | 月度 IC 的波动 | 越小越好，大于 0.1 说明信号很不稳 |
| `IR` | mean/std | 绝对值 ≥ 0.3 算合格 |
| `t-stat` | `mean IC` 是否显著不为 0 | \|t\| ≥ 2 才有统计意义 |
| `quantile_means` | 按 pred 分 5 组的平均未来收益 | 应该单调（Q1→Q5 或 Q5→Q1），不单调说明模型没学到分层 |

### 5.2 信号方向

`mean IC < 0` 不一定是 bug——只是说模型预测分高的股票未来收益低。这种情况下：
- `long_return` 是亏的
- `short_return` 是赚的
- 想测"反向"，把 `alpha_composite` 取负号再回测一次：

```bash
python -c "
import pandas as pd
df = pd.read_parquet('results/ml_oos.parquet')
df['alpha_composite'] = -df['alpha_composite']
df.to_parquet('results/ml_oos_neg.parquet', index=False)
"
python tools/research/backtest.py \
    --data results/ml_oos_neg.parquet \
    --save results/ml_report/backtest_neg \
    --config data/ml_composite.yaml \
    --factor alpha_composite
```

### 5.3 month_log.csv 怎么用

```python
import pandas as pd
log = pd.read_csv("results/ml_report/month_log.csv")
print(log[["test_month", "mean_ic"]])
# 如果最近几个月 mean_ic 明显比早期低（比如早期 +0.04，最近 -0.02），
# 说明因子在衰减，模型需要更频繁重训，或者考虑 rolling window
```

---

## 6. 已知的坑

1. **`fetch_data.py` 的 yaml 里 `alpha:` 只列 TA-Lib 因子**（59 个）。基本面和估值因子通过 `include_fundamentals` / `include_valuation` 自动加，不要写到 `alpha:` 下，否则 `resolve_factor_indices` 会报 `Unknown TA-Lib factor 'roe'`。
2. **trace dates 必须落在 OOS 数据范围内**。OOS 的范围是第一个测试月的 embargo 之后到 panel 最后一天往前 `horizon` 天（标签不 NaN 的部分）。超出会报 "date not exists"。
3. **`ml_composite.yaml` 的 winsorize/standardize 一定要留空**。复合信号在 walk-forward 里已经做过 per-window winsorize+中性化+zscore，再压一次会丢信息。
4. **不要重新引入 holdout / early stopping**，除非有 OOS 证据说明它更好。即便要试，先用 `n_estimators=200 + early_stopping=50` 跑一遍 OOS，再和 `n_estimators=100` 默认版对比 mean IC。
5. **拉估值前必须先建表**。`stock_valuation_d` 表在 `core/data/schema.sql` 里，第一次调 `get_duckdb_connection()` 会自动 `CREATE TABLE IF NOT EXISTS`。如果直接跑 `--valuation` 报 "Table stock_valuation_d does not exist"，先随便跑一个会触发 `_initialize_schema()` 的脚本（比如 `get_fundamental.py` 不带参数）。
6. **行业中性化需要 panel 里有 `industry` 列**。`fetch_data.py` 默认会 join `stock_industry`，但如果某只票在 industry 表里找不到，会被填成 `"UNKNOWN"`，所有 UNKNOWN 票共用一个均值——可能影响中性化效果。建议跑前确认 `SELECT COUNT(*) FROM stock_industry WHERE industry = ''` 接近 0。

---

## 7. 想改进的几个方向

- **特征工程**：把原始 TA-Lib 因子做行业中性化、市值中性化后再喂模型，通常能显著提升 IR。
- **Ridge / Lasso baseline**：在同样的 walk-forward 框架下跑一个线性模型做对照。如果线性模型 OOS IC 比 LightGBM 还高，说明树模型在过拟合，应该再压超参（`max_depth` ↓ 或 `n_estimators` ↓）。
- **行业 one-hot**：把行业作为分类特征加进去，LightGBM 处理类别特征比 one-hot 更高效。
- **多 horizon 集成**：同时训 1d / 5d / 10d 三个 horizon 的模型，pred 取平均或再学一层权重。
- **Rolling window 替代 expanding**：当数据 ≥ 5 年时，固定窗口（比如最近 24 个月）能更快适应 regime 变化。短数据下还是 expanding 占优。
