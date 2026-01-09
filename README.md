# Combined Multi-Strategy Trading System

## Overview

This system aggregates five independent trading strategies into a single portfolio, each operating with its own capital allocation and position management. Starting with a $100,000 bankroll (which we just hard-coded, just change it to whatever bankroll we actually have), the system diversifies across different market inefficiencies to achieve more stable risk-adjusted returns.

---

## Capital Allocation

| Strategy | Allocation | Rationale |
|----------|------------|-----------|
| **Mean Reverting Strategies** | 35% | Exploits price inefficiencies that revert to equilibrium |
| → Pairs Trading (L & O) | 9.21% | Lower Sharpe weight (0.5 ratio) |
| → Stock V Mean Reversion | 12.89% | Higher Sharpe weight (0.7 ratio) |
| → Stock Q Mean Reversion | 12.89% | Higher Sharpe weight (0.7 ratio) |
| **HMM Volatility Regime** | 30% | Adapts to market conditions dynamically |
| **Seasonality Trading** | 15% | Calendar-based predictable patterns |
| **Cash Reserve** | 20% | Liquidity buffer / risk management |

Within the 35% mean reversion allocation, capital is distributed by Sharpe ratio: `pairs:V:Q = 0.5:0.7:0.7`

---

## Strategy Descriptions

### 1. Pairs Trading (Stocks L & O)

**Concept:** Stocks L and O are cointegrated—they move together over time. When their spread deviates from the mean, we bet on reversion.

**Mechanism:**
- Calculate hedge ratio via OLS regression: `L = α + β × O`
- Compute spread: `Spread = L - β × O`
- Normalize to z-score using rolling 1000-day window
- **Entry:** |z| > 1.0 (spread extended)
- **Exit:** z crosses 0 (spread normalized) or 150 days max hold

**Trades:**
- `z < -1.0` → Long spread (buy L, short O)
- `z > +1.0` → Short spread (short L, buy O)

---

### 2. Stock V Mean Reversion

**Concept:** Stock V exhibits mean-reverting behavior after momentum extremes, driven by retail trading creating temporary inefficiencies.

**Mechanism:**
- Calculate 22-day momentum (log price change)
- Normalize by 22-day rolling volatility → z-score
- Trade **against** momentum (contrarian)
- **Entry:** |z| > 2.0
- **Exit:** |z| < 1.0 (hysteresis to reduce whipsaw)

**Trades:**
- High momentum (z > 2.0) → Short (expect reversal down)
- Low momentum (z < -2.0) → Long (expect reversal up)

---

### 3. Stock Q Mean Reversion

**Concept:** Stock Q follows a random walk with drift. The residuals around the linear trend are mean-reverting (Ornstein-Uhlenbeck process).

**Mechanism:**
- Fit expanding window linear trend: `P_t = α + β×t + residual`
- Calculate z-score of deviation from expected price
- **Entry:** |z| > 1.5σ (price significantly off trend)
- **Exit:** z crosses 0 (price returns to trend)

**Trades:**
- Price below trend (z < -1.5) → Long
- Price above trend (z > +1.5) → Short

---

### 4. HMM Volatility Regime Strategy

**Concept:** Markets cycle through volatility regimes. Different stocks outperform in different regimes. A Hidden Markov Model detects the current regime.

**Mechanism:**
- Fit 3-state Gaussian HMM on absolute market returns
- States labeled by variance: **LOW**, **MID**, **HIGH**
- Switch portfolio allocation when regime changes

**Regime Portfolios:**
| Regime | Long | Short |
|--------|------|-------|
| HIGH volatility | N, M, K | D |
| MID volatility | Y, V, C, K | R, I |
| LOW volatility | V | E |

---

### 5. Seasonality Trading (Stocks C, K, M)

**Concept:** Certain stocks exhibit calendar-based seasonal patterns with higher returns in specific quarters.

**Mechanism:**
- Simple calendar rule: hold during favorable quarters, flat otherwise
- Equal weight across seasonal stocks

**Seasonal Schedule:**
| Stock | Active Quarters |
|-------|-----------------|
| C | Q1 (Jan-Mar) + Q4 (Oct-Dec) |
| K | Q4 only (Oct-Dec) |
| M | Q4 only (Oct-Dec) |

---

## Architecture

| Layer | Strategy | Allocation | Capital | Assets |
|-------|----------|------------|---------|--------|
| **Mean Reversion** | Pairs Trading | 9.21% | $9,211 | L & O |
| | Stock V Mean Reversion | 12.89% | $12,895 | V |
| | Stock Q Mean Reversion | 12.89% | $12,895 | Q |
| **Regime-Based** | HMM Volatility | 30.00% | $30,000 | Dynamic |
| **Calendar-Based** | Seasonality | 15.00% | $15,000 | C, K, M |
| **Reserve** | Cash | 20.00% | $20,000 | — |
| | **Total** | **100%** | **$100,000** | |





**Key Design Principle:** Each strategy maintains **independent positions**. Position sizing within each strategy is based only on that strategy's allocated capital, not the aggregate portfolio. This prevents strategies from interfering with each other.

---

## Performance Summary

| Metric | Combined Portfolio |
|--------|-------------------|
| Initial Capital | $100,000 |
| Final Value | $211,263 |
| Total Return | 111.3% |
| Sharpe Ratio | 0.67 |
| Max Drawdown | -17.5% |

**Diversification Benefit:** The combined max drawdown (-17.5%) is significantly lower than individual strategies (e.g., HMM alone: -47.7%), demonstrating the risk reduction from strategy diversification.

---

## Files

| File | Description |
|------|-------------|
| `combined_strategy.py` | Main executable with all strategies |
| `simulated_prices.csv` | Input price data (25 stocks, ~10 years) |
| `combined_strategy_backtest.csv` | Daily equity curves for all strategies |
| `combined_strategy_results.png` | Visualization of performance |

---

## Usage

```bash
python combined_strategy.py
