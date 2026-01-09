"""
Combined Trading Strategy
=========================
Aggregates 5 sub-strategies with independent capital allocations:
  1. Pairs Trading (L & O)      - 9.21% of capital
  2. Stock V Mean Reversion     - 12.89% of capital
  3. Stock Q Mean Reversion     - 12.89% of capital
  4. HMM Volatility Regime      - 30.00% of capital
  5. Seasonality Trading (C,K,M)- 15.00% of capital
  6. Cash Reserve               - 20.00% of capital

Each strategy maintains its own sub-position and trades independently.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

INITIAL_CAPITAL = 100000
TRANSACTION_COST_BPS = 10  # 10 basis points

# Capital allocation
MEAN_REVERSION_PCT = 0.35  # 35% to mean reverting strategies
HMM_PCT = 0.30             # 30% to HMM volatility regime
SEASONALITY_PCT = 0.15     # 15% to seasonality
CASH_RESERVE_PCT = 0.20    # 20% cash reserve

# Within mean reversion, ratio is pairs:V:Q = 0.5:0.7:0.7
PAIRS_RATIO = 0.5
STOCK_V_RATIO = 0.7
STOCK_Q_RATIO = 0.7
MEAN_REV_TOTAL_RATIO = PAIRS_RATIO + STOCK_V_RATIO + STOCK_Q_RATIO

# Strategy-specific parameters
PAIRS_WINDOW = 1000
PAIRS_ENTRY_THRESH = 1.0
PAIRS_EXIT_THRESH = 0.0
PAIRS_MAX_HOLDING = 150

STOCK_V_LOOKBACK = 22
STOCK_V_VOL_WIN = 22
STOCK_V_Z_ENTRY = 2.0
STOCK_V_Z_EXIT = 1.0

STOCK_Q_MIN_HISTORY = 1000
STOCK_Q_ENTRY_THRESH = 1.5
STOCK_Q_EXIT_THRESH = 0.0

HMM_N_STATES = 3
HMM_N_ITER = 3000
HMM_N_RESTARTS = 30
HMM_EXCLUDED = ['F', 'G', 'J']

SEASONAL_STOCKS = {'C': [1, 4], 'K': [4], 'M': [4]}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calc_sharpe(returns, ann=252):
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0
    return np.sqrt(ann) * returns.mean() / returns.std()


def calc_max_drawdown(equity):
    """Calculate maximum drawdown."""
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def parse_date(date_str):
    """Parse Y0-M01-D02 format to year, month, day."""
    parts = date_str.split('-')
    year = int(parts[0][1:])
    month = int(parts[1][1:])
    day = int(parts[2][1:])
    return year, month, day


def get_quarter(month):
    """Get quarter from month."""
    if month <= 3:
        return 1
    elif month <= 6:
        return 2
    elif month <= 9:
        return 3
    else:
        return 4


# =============================================================================
# SUB-STRATEGY CLASSES
# =============================================================================

class PairsTradingStrategy:
    """Pairs trading between stocks L and O using rolling window parameters."""

    def __init__(self, capital, window=1000, entry_thresh=1.0, exit_thresh=0.0,
                 max_holding=150, tc_bps=10):
        self.initial_capital = capital
        self.window = window
        self.entry_thresh = entry_thresh
        self.exit_thresh = exit_thresh
        self.max_holding = max_holding
        self.tc_rate = tc_bps / 10000
        self.reset()

    def reset(self):
        self.position = 0
        self.shares_L = 0
        self.shares_O = 0
        self.cash = self.initial_capital
        self.entry_day = 0
        self.entry_price_L = 0
        self.entry_price_O = 0
        self.entry_z = 0
        self.trades = []
        self.equity_history = []

    def step(self, day, price_L, price_O, prices_L_history, prices_O_history):
        """Process one day of trading."""
        # Need enough history for rolling window
        if day < self.window:
            self.equity_history.append(self.cash)
            return

        # Calculate rolling parameters
        L_window = prices_L_history[day-self.window:day]
        O_window = prices_O_history[day-self.window:day]

        model = OLS(L_window, add_constant(O_window)).fit()
        hedge_ratio = model.params[1]

        spread_window = L_window - hedge_ratio * O_window
        spread_mean = spread_window.mean()
        spread_std = spread_window.std()

        current_spread = price_L - hedge_ratio * price_O
        z = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0

        # Position sizing
        target_shares_L = int(self.initial_capital * 0.4 / price_L)
        target_shares_O = int(target_shares_L * hedge_ratio)

        # Check max holding exit
        if self.position != 0 and (day - self.entry_day) >= self.max_holding:
            self._close_position(day, price_L, price_O, z, 'max_holding')

        # Trading logic
        if self.position == 0:
            if z > self.entry_thresh:  # Short spread
                self.position = -1
                self.shares_L = -target_shares_L
                self.shares_O = target_shares_O
                self.entry_price_L, self.entry_price_O = price_L, price_O
                self.entry_day, self.entry_z = day, z
                self.cash += target_shares_L * price_L * (1 - self.tc_rate)
                self.cash -= target_shares_O * price_O * (1 + self.tc_rate)
            elif z < -self.entry_thresh:  # Long spread
                self.position = 1
                self.shares_L = target_shares_L
                self.shares_O = -target_shares_O
                self.entry_price_L, self.entry_price_O = price_L, price_O
                self.entry_day, self.entry_z = day, z
                self.cash -= target_shares_L * price_L * (1 + self.tc_rate)
                self.cash += target_shares_O * price_O * (1 - self.tc_rate)
        elif self.position == 1 and z >= self.exit_thresh:
            self._close_position(day, price_L, price_O, z, 'signal')
        elif self.position == -1 and z <= -self.exit_thresh:
            self._close_position(day, price_L, price_O, z, 'signal')

        # Record equity
        equity = self.cash + self.shares_L * price_L + self.shares_O * price_O
        self.equity_history.append(equity)

    def _close_position(self, day, price_L, price_O, z, reason):
        if self.position == 1:
            pnl = self.shares_L * (price_L - self.entry_price_L)
            pnl += (-self.shares_O) * (self.entry_price_O - price_O)
            self.cash += self.shares_L * price_L * (1 - self.tc_rate)
            self.cash -= (-self.shares_O) * price_O * (1 + self.tc_rate)
        else:
            pnl = (-self.shares_L) * (self.entry_price_L - price_L)
            pnl += self.shares_O * (price_O - self.entry_price_O)
            self.cash -= (-self.shares_L) * price_L * (1 + self.tc_rate)
            self.cash += self.shares_O * price_O * (1 - self.tc_rate)

        self.trades.append({
            'entry_day': self.entry_day, 'exit_day': day,
            'type': 'LONG' if self.position == 1 else 'SHORT',
            'pnl': pnl, 'reason': reason
        })
        self.position = 0
        self.shares_L = self.shares_O = 0

    def force_close(self, day, price_L, price_O):
        if self.position != 0:
            self._close_position(day, price_L, price_O, 0, 'end')


class StockVMeanReversion:
    """Mean reversion strategy for stock V using momentum z-score."""

    def __init__(self, capital, lookback=22, vol_win=22, z_entry=2.0, z_exit=1.0, tc_bps=10):
        self.initial_capital = capital
        self.lookback = lookback
        self.vol_win = vol_win
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.tc_rate = tc_bps / 10000
        self.reset()

    def reset(self):
        self.position = 0
        self.shares = 0
        self.cash = self.initial_capital
        self.entry_price = 0
        self.trades = []
        self.equity_history = []

    def step(self, day, price, log_prices):
        """Process one day."""
        min_days = max(self.lookback, self.vol_win) + 1
        if day < min_days:
            self.equity_history.append(self.cash)
            return

        # Calculate momentum and z-score
        log_ret = np.diff(log_prices[:day+1])
        mom = log_prices[day] - log_prices[day - self.lookback]
        vol = np.std(log_ret[-self.vol_win:])
        z = mom / (vol * np.sqrt(self.lookback)) if vol > 0 else 0

        # Position sizing
        shares_to_trade = int(self.initial_capital * 0.8 / price)

        # Trading logic (mean reversion: trade against momentum)
        if self.position == 0:
            if z > self.z_entry:  # Overbought -> short
                self.position = -1
                self.shares = -shares_to_trade
                self.cash += shares_to_trade * price * (1 - self.tc_rate)
                self.entry_price = price
            elif z < -self.z_entry:  # Oversold -> long
                self.position = 1
                self.shares = shares_to_trade
                self.cash -= shares_to_trade * price * (1 + self.tc_rate)
                self.entry_price = price
        elif self.position == 1 and z > -self.z_exit:
            self._close(day, price, 'signal')
        elif self.position == -1 and z < self.z_exit:
            self._close(day, price, 'signal')

        equity = self.cash + self.shares * price
        self.equity_history.append(equity)

    def _close(self, day, price, reason):
        if self.position == 1:
            pnl = self.shares * (price - self.entry_price)
            self.cash += self.shares * price * (1 - self.tc_rate)
        else:
            pnl = (-self.shares) * (self.entry_price - price)
            self.cash -= (-self.shares) * price * (1 + self.tc_rate)
        self.trades.append({'exit_day': day, 'pnl': pnl, 'reason': reason})
        self.position = 0
        self.shares = 0

    def force_close(self, day, price):
        if self.position != 0:
            self._close(day, price, 'end')


class StockQMeanReversion:
    """Mean reversion for stock Q using expanding window trend estimation."""

    def __init__(self, capital, min_history=1000, entry_thresh=1.5, exit_thresh=0.0, tc_bps=10):
        self.initial_capital = capital
        self.min_history = min_history
        self.entry_thresh = entry_thresh
        self.exit_thresh = exit_thresh
        self.tc_rate = tc_bps / 10000
        self.reset()

    def reset(self):
        self.position = 0
        self.shares = 0
        self.cash = self.initial_capital
        self.entry_price = 0
        self.entry_z = 0
        self.trades = []
        self.equity_history = []

    def step(self, day, price, prices_history):
        if day < self.min_history:
            self.equity_history.append(self.cash)
            return

        # Fit trend on historical data
        hist_prices = prices_history[:day]
        hist_t = np.arange(day)
        model = OLS(hist_prices, add_constant(hist_t)).fit()
        alpha, beta = model.params[0], model.params[1]

        expected = alpha + beta * day
        residuals = hist_prices - (alpha + beta * hist_t)
        z = (price - expected) / residuals.std() if residuals.std() > 0 else 0

        shares_to_trade = int(self.initial_capital * 0.8 / price)

        if self.position == 0:
            if z > self.entry_thresh:  # Above trend -> short
                self.position = -1
                self.shares = -shares_to_trade
                self.cash += shares_to_trade * price * (1 - self.tc_rate)
                self.entry_price, self.entry_z = price, z
            elif z < -self.entry_thresh:  # Below trend -> long
                self.position = 1
                self.shares = shares_to_trade
                self.cash -= shares_to_trade * price * (1 + self.tc_rate)
                self.entry_price, self.entry_z = price, z
        elif self.position == 1 and z >= self.exit_thresh:
            self._close(day, price, 'signal')
        elif self.position == -1 and z <= -self.exit_thresh:
            self._close(day, price, 'signal')

        equity = self.cash + self.shares * price
        self.equity_history.append(equity)

    def _close(self, day, price, reason):
        if self.position == 1:
            pnl = self.shares * (price - self.entry_price)
            self.cash += self.shares * price * (1 - self.tc_rate)
        else:
            pnl = (-self.shares) * (self.entry_price - price)
            self.cash -= (-self.shares) * price * (1 + self.tc_rate)
        self.trades.append({'exit_day': day, 'pnl': pnl, 'reason': reason})
        self.position = 0
        self.shares = 0

    def force_close(self, day, price):
        if self.position != 0:
            self._close(day, price, 'end')


class HMMVolatilityStrategy:
    """HMM-based volatility regime strategy with regime-dependent portfolios."""

    STRATEGY = {
        'HIGH': {'long': ['N', 'M', 'K'], 'short': ['D']},
        'MID': {'long': ['Y', 'V', 'C', 'K'], 'short': ['R', 'I']},
        'LOW': {'long': ['V'], 'short': ['E']}
    }

    def __init__(self, capital, n_states=3, n_iter=3000, n_restarts=30, excluded=None, tc_bps=10):
        self.initial_capital = capital
        self.n_states = n_states
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.excluded = excluded or ['F', 'G', 'J']
        self.tc_rate = tc_bps / 10000
        self.hmm_model = None
        self.state_mapping = None
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.positions = {}  # {stock: shares}
        self.current_regime = None
        self.equity_history = []
        self.regime_history = []

    def fit_hmm(self, returns):
        """Fit HMM on absolute returns."""
        x = np.abs(returns.values)
        x_mean, x_std = x.mean(), x.std()
        X = ((x - x_mean) / (x_std + 1e-12)).reshape(-1, 1)

        best_model, best_ll = None, -np.inf
        for seed in range(self.n_restarts):
            model = GaussianHMM(n_components=self.n_states, covariance_type="diag",
                               n_iter=self.n_iter, random_state=seed, tol=1e-6)
            model.fit(X)
            ll = model.score(X)
            if ll > best_ll:
                best_ll, best_model = ll, model

        self.hmm_model = best_model
        state_vars = best_model.covars_.reshape(-1)
        order = np.argsort(state_vars)
        self.state_mapping = {int(order[0]): "LOW", int(order[1]): "MID", int(order[2]): "HIGH"}
        self.x_mean, self.x_std = x_mean, x_std

        return best_model

    def predict_regime(self, mkt_return):
        """Predict regime for a single observation."""
        x = (np.abs(mkt_return) - self.x_mean) / (self.x_std + 1e-12)
        state = self.hmm_model.predict(np.array([[x]]))[0]
        return self.state_mapping[int(state)]

    def step(self, day, prices_dict, mkt_return):
        """Process one day."""
        regime = self.predict_regime(mkt_return)
        self.regime_history.append(regime)

        # Rebalance if regime changed
        if regime != self.current_regime:
            self._rebalance(regime, prices_dict)
            self.current_regime = regime

        # Calculate equity
        equity = self.cash
        for stock, shares in self.positions.items():
            equity += shares * prices_dict[stock]
        self.equity_history.append(equity)

    def _rebalance(self, regime, prices_dict):
        """Close old positions and open new ones for the regime."""
        # Close all existing positions
        for stock, shares in self.positions.items():
            if shares > 0:
                self.cash += shares * prices_dict[stock] * (1 - self.tc_rate)
            elif shares < 0:
                self.cash -= abs(shares) * prices_dict[stock] * (1 + self.tc_rate)
        self.positions = {}

        # Open new positions
        long_stocks = self.STRATEGY[regime]['long']
        short_stocks = self.STRATEGY[regime]['short']

        # Equal weight among long and short sides
        n_legs = (1 if long_stocks else 0) + (1 if short_stocks else 0)
        capital_per_leg = self.cash / n_legs if n_legs > 0 else 0

        for stock in long_stocks:
            shares = int(capital_per_leg / len(long_stocks) / prices_dict[stock])
            self.positions[stock] = shares
            self.cash -= shares * prices_dict[stock] * (1 + self.tc_rate)

        for stock in short_stocks:
            shares = int(capital_per_leg / len(short_stocks) / prices_dict[stock])
            self.positions[stock] = -shares
            self.cash += shares * prices_dict[stock] * (1 - self.tc_rate)

    def force_close(self, prices_dict):
        """Close all positions at end."""
        for stock, shares in self.positions.items():
            if shares > 0:
                self.cash += shares * prices_dict[stock] * (1 - self.tc_rate)
            elif shares < 0:
                self.cash -= abs(shares) * prices_dict[stock] * (1 + self.tc_rate)
        self.positions = {}


class SeasonalityStrategy:
    """Calendar-based seasonality strategy for stocks C, K, M."""

    def __init__(self, capital, seasonal_stocks=None, tc_bps=10):
        self.initial_capital = capital
        self.seasonal_stocks = seasonal_stocks or {'C': [1, 4], 'K': [4], 'M': [4]}
        self.tc_rate = tc_bps / 10000
        # Equal weight per stock
        self.capital_per_stock = capital / len(self.seasonal_stocks)
        self.reset()

    def reset(self):
        self.sub_strategies = {}
        for stock in self.seasonal_stocks:
            self.sub_strategies[stock] = {
                'position': 0,
                'shares': 0,
                'cash': self.capital_per_stock,
                'entry_price': 0
            }
        self.equity_history = []

    def step(self, day, prices_dict, quarter):
        """Process one day for all seasonal stocks."""
        for stock, quarters in self.seasonal_stocks.items():
            sub = self.sub_strategies[stock]
            price = prices_dict[stock]
            should_be_in = quarter in quarters

            if sub['position'] == 0 and should_be_in:
                # Enter position
                shares = int(sub['cash'] * 0.95 / price)
                sub['shares'] = shares
                sub['cash'] -= shares * price * (1 + self.tc_rate)
                sub['position'] = 1
                sub['entry_price'] = price
            elif sub['position'] == 1 and not should_be_in:
                # Exit position
                sub['cash'] += sub['shares'] * price * (1 - self.tc_rate)
                sub['position'] = 0
                sub['shares'] = 0

        # Calculate total equity
        equity = sum(
            sub['cash'] + sub['shares'] * prices_dict[stock]
            for stock, sub in self.sub_strategies.items()
        )
        self.equity_history.append(equity)

    def force_close(self, prices_dict):
        """Close all positions."""
        for stock, sub in self.sub_strategies.items():
            if sub['position'] == 1:
                sub['cash'] += sub['shares'] * prices_dict[stock] * (1 - self.tc_rate)
                sub['position'] = 0
                sub['shares'] = 0


# =============================================================================
# COMBINED STRATEGY
# =============================================================================

class CombinedStrategy:
    """
    Master strategy that coordinates all sub-strategies with proper capital allocation.
    """

    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital

        # Calculate capital allocations
        mean_rev_capital = initial_capital * MEAN_REVERSION_PCT
        pairs_capital = mean_rev_capital * (PAIRS_RATIO / MEAN_REV_TOTAL_RATIO)
        stock_v_capital = mean_rev_capital * (STOCK_V_RATIO / MEAN_REV_TOTAL_RATIO)
        stock_q_capital = mean_rev_capital * (STOCK_Q_RATIO / MEAN_REV_TOTAL_RATIO)
        hmm_capital = initial_capital * HMM_PCT
        seasonality_capital = initial_capital * SEASONALITY_PCT
        cash_reserve = initial_capital * CASH_RESERVE_PCT

        print(f"Capital Allocation:")
        print(f"  Pairs Trading (L & O):     ${pairs_capital:>10,.2f} ({pairs_capital/initial_capital*100:.2f}%)")
        print(f"  Stock V Mean Reversion:    ${stock_v_capital:>10,.2f} ({stock_v_capital/initial_capital*100:.2f}%)")
        print(f"  Stock Q Mean Reversion:    ${stock_q_capital:>10,.2f} ({stock_q_capital/initial_capital*100:.2f}%)")
        print(f"  HMM Volatility Regime:     ${hmm_capital:>10,.2f} ({hmm_capital/initial_capital*100:.2f}%)")
        print(f"  Seasonality (C, K, M):     ${seasonality_capital:>10,.2f} ({seasonality_capital/initial_capital*100:.2f}%)")
        print(f"  Cash Reserve:              ${cash_reserve:>10,.2f} ({cash_reserve/initial_capital*100:.2f}%)")
        print(f"  {'='*45}")
        print(f"  Total:                     ${initial_capital:>10,.2f}")

        # Initialize sub-strategies
        self.pairs = PairsTradingStrategy(
            pairs_capital, PAIRS_WINDOW, PAIRS_ENTRY_THRESH,
            PAIRS_EXIT_THRESH, PAIRS_MAX_HOLDING, TRANSACTION_COST_BPS
        )
        self.stock_v = StockVMeanReversion(
            stock_v_capital, STOCK_V_LOOKBACK, STOCK_V_VOL_WIN,
            STOCK_V_Z_ENTRY, STOCK_V_Z_EXIT, TRANSACTION_COST_BPS
        )
        self.stock_q = StockQMeanReversion(
            stock_q_capital, STOCK_Q_MIN_HISTORY,
            STOCK_Q_ENTRY_THRESH, STOCK_Q_EXIT_THRESH, TRANSACTION_COST_BPS
        )
        self.hmm = HMMVolatilityStrategy(
            hmm_capital, HMM_N_STATES, HMM_N_ITER,
            HMM_N_RESTARTS, HMM_EXCLUDED, TRANSACTION_COST_BPS
        )
        self.seasonality = SeasonalityStrategy(
            seasonality_capital, SEASONAL_STOCKS, TRANSACTION_COST_BPS
        )

        self.cash_reserve = cash_reserve
        self.combined_equity = []

    def run(self, df):
        """
        Run all strategies on the dataframe.

        Args:
            df: DataFrame with columns for each stock (A-Y) and 'Date'
        """
        n_days = len(df)

        # Prepare data
        prices_L = df['L'].values
        prices_O = df['O'].values
        prices_V = df['V'].values
        prices_Q = df['Q'].values
        log_prices_V = np.log(prices_V)

        # Parse dates for seasonality
        years, months, days = zip(*df['Date'].apply(parse_date))
        quarters = [get_quarter(m) for m in months]

        # Calculate market returns for HMM (excluding F, G, J)
        stock_cols = [c for c in df.columns if len(c) == 1 and c.isalpha() and c not in HMM_EXCLUDED]
        prices_df = df[stock_cols]
        returns_df = prices_df.pct_change()
        mkt_returns = returns_df.mean(axis=1).fillna(0).values

        # Fit HMM on all data (in practice, would use expanding window)
        print("\nFitting HMM model...")
        self.hmm.fit_hmm(pd.Series(mkt_returns[1:]))  # Skip first NaN
        print(f"HMM fitted with states: {self.hmm.state_mapping}")

        print(f"\nRunning backtest over {n_days} days...")

        # Run day by day
        for day in range(n_days):
            # Pairs trading
            self.pairs.step(day, prices_L[day], prices_O[day], prices_L, prices_O)

            # Stock V mean reversion
            self.stock_v.step(day, prices_V[day], log_prices_V)

            # Stock Q mean reversion
            self.stock_q.step(day, prices_Q[day], prices_Q)

            # HMM strategy (needs at least 1 day of returns)
            if day > 0:
                prices_dict = {col: df[col].iloc[day] for col in stock_cols}
                self.hmm.step(day, prices_dict, mkt_returns[day])
            else:
                self.hmm.equity_history.append(self.hmm.initial_capital)
                self.hmm.regime_history.append('MID')

            # Seasonality
            seasonal_prices = {stock: df[stock].iloc[day] for stock in SEASONAL_STOCKS}
            self.seasonality.step(day, seasonal_prices, quarters[day])

            # Aggregate equity
            total_equity = (
                self.pairs.equity_history[-1] +
                self.stock_v.equity_history[-1] +
                self.stock_q.equity_history[-1] +
                self.hmm.equity_history[-1] +
                self.seasonality.equity_history[-1] +
                self.cash_reserve
            )
            self.combined_equity.append(total_equity)

        # Force close all positions
        self.pairs.force_close(n_days - 1, prices_L[-1], prices_O[-1])
        self.stock_v.force_close(n_days - 1, prices_V[-1])
        self.stock_q.force_close(n_days - 1, prices_Q[-1])
        self.hmm.force_close({col: df[col].iloc[-1] for col in stock_cols})
        self.seasonality.force_close({stock: df[stock].iloc[-1] for stock in SEASONAL_STOCKS})

        print("Backtest complete!")

        return self.combined_equity

    def get_results(self):
        """Get performance results for each sub-strategy and combined."""
        results = {}

        strategies = {
            'Pairs Trading': self.pairs,
            'Stock V Mean Reversion': self.stock_v,
            'Stock Q Mean Reversion': self.stock_q,
            'HMM Volatility': self.hmm,
            'Seasonality': self.seasonality
        }

        for name, strat in strategies.items():
            equity = np.array(strat.equity_history)
            initial = strat.initial_capital
            daily_returns = np.diff(equity) / equity[:-1]

            results[name] = {
                'initial_capital': initial,
                'final_equity': equity[-1],
                'total_return': (equity[-1] - initial) / initial,
                'sharpe': calc_sharpe(pd.Series(daily_returns)),
                'max_drawdown': calc_max_drawdown(equity),
                'equity_history': equity
            }

        # Combined results
        combined = np.array(self.combined_equity)
        combined_returns = np.diff(combined) / combined[:-1]
        results['Combined'] = {
            'initial_capital': self.initial_capital,
            'final_equity': combined[-1],
            'total_return': (combined[-1] - self.initial_capital) / self.initial_capital,
            'sharpe': calc_sharpe(pd.Series(combined_returns)),
            'max_drawdown': calc_max_drawdown(combined),
            'equity_history': combined
        }

        return results

    def print_results(self):
        """Print performance summary."""
        results = self.get_results()

        print("\n" + "="*85)
        print("COMBINED STRATEGY PERFORMANCE SUMMARY")
        print("="*85)
        print(f"\n{'Strategy':<25} {'Initial':>12} {'Final':>12} {'Return':>10} {'Sharpe':>8} {'Max DD':>10}")
        print("-"*85)

        for name in ['Pairs Trading', 'Stock V Mean Reversion', 'Stock Q Mean Reversion',
                     'HMM Volatility', 'Seasonality', 'Combined']:
            r = results[name]
            print(f"{name:<25} ${r['initial_capital']:>10,.0f} ${r['final_equity']:>10,.0f} "
                  f"{r['total_return']*100:>9.1f}% {r['sharpe']:>8.2f} {r['max_drawdown']*100:>9.1f}%")

        print("-"*85)
        print(f"\nCash Reserve: ${self.cash_reserve:,.0f} (held throughout)")

        return results

    def plot_results(self):
        """Plot equity curves for all strategies."""
        results = self.get_results()

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Individual strategies
        ax1 = axes[0]
        colors = ['blue', 'green', 'purple', 'orange', 'red']
        for i, name in enumerate(['Pairs Trading', 'Stock V Mean Reversion', 'Stock Q Mean Reversion',
                                  'HMM Volatility', 'Seasonality']):
            equity = results[name]['equity_history']
            # Normalize to 100 for comparison
            normalized = equity / equity[0] * 100
            ax1.plot(normalized, label=f"{name} ({results[name]['total_return']*100:.1f}%)",
                    color=colors[i], alpha=0.7, linewidth=1)

        ax1.axhline(100, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Trading Day')
        ax1.set_ylabel('Normalized Value (Base=100)')
        ax1.set_title('Individual Strategy Performance (Normalized)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Combined strategy
        ax2 = axes[1]
        combined = results['Combined']['equity_history']
        ax2.plot(combined, color='darkgreen', linewidth=2,
                label=f"Combined ({results['Combined']['total_return']*100:.1f}%)")
        ax2.axhline(self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')

        ax2.set_xlabel('Trading Day')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.set_title('Combined Strategy Performance')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('combined_strategy_results.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("\nPlot saved to combined_strategy_results.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print("COMBINED TRADING STRATEGY BACKTEST")
    print("="*70)

    # Load data
    print("\nLoading price data...")
    df = pd.read_csv('simulated_prices.csv')
    print(f"Loaded {len(df)} trading days, {len(df.columns)-1} stocks")
    print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")

    # Initialize and run combined strategy
    print(f"\nInitializing combined strategy with ${INITIAL_CAPITAL:,} capital...")
    strategy = CombinedStrategy(INITIAL_CAPITAL)

    # Run backtest
    strategy.run(df)

    # Print and plot results
    results = strategy.print_results()
    strategy.plot_results()

    # Export results
    export_data = pd.DataFrame({
        'Day': range(len(strategy.combined_equity)),
        'Combined_Equity': strategy.combined_equity,
        'Pairs_Equity': strategy.pairs.equity_history,
        'StockV_Equity': strategy.stock_v.equity_history,
        'StockQ_Equity': strategy.stock_q.equity_history,
        'HMM_Equity': strategy.hmm.equity_history,
        'Seasonality_Equity': strategy.seasonality.equity_history,
        'Regime': strategy.hmm.regime_history
    })
    export_data.to_csv('combined_strategy_backtest.csv', index=False)
    print("\nResults exported to combined_strategy_backtest.csv")

    return strategy, results


if __name__ == "__main__":
    strategy, results = main()
