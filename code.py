import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, data_path):
        df = pd.read_csv(data_path, parse_dates=['date'])
        
        # Pivot to wide format
        self.data = df.pivot(index='date', columns='ticker', values='close').sort_index()
        
        self.data = self.data.ffill().bfill()
        
        self.tickers = self.data.columns
        self.num_stocks = len(self.tickers)

    def run(self, strategy, initial_capital=1.0):
        """
        Executes the daily close trading simulation.
        """
        dates = self.data.index
        # Track Net Asset Value (NAV) and Weights
        nav_curve = pd.Series(index=dates, dtype=float)
        nav_curve.iloc[0] = initial_capital
        
        # DataFrame to log daily weights for the report [cite: 44]
        weights_history = pd.DataFrame(index=dates, columns=self.tickers, dtype=float)

        # Iterate day-by-day to avoid lookahead bias [cite: 21, 31]
        for i in range(1, len(dates)):
            current_date = dates[i]
            prev_date = dates[i-1]
            
            # Slice data: only info available up to "today's" close [cite: 20]
            historical_data = self.data.loc[:current_date]
            
            # 1. Get target weights from strategy
            target_weights = strategy.compute_weights(historical_data)
            
            # 2. Validate Constraints [cite: 22, 23, 27, 28]
            weights_vec = self._validate_weights(target_weights)
            weights_history.loc[current_date] = weights_vec
            
            # 3. Calculate Daily Return
            # Execution happens at closing price [cite: 25]
            daily_returns = self.data.loc[current_date] / self.data.loc[prev_date] - 1
            
            # Portfolio return is the weighted sum of stock returns
            # Remainder is held in cash (0% return) [cite: 23]
            port_return = np.dot(weights_vec, daily_returns)
            
            # 4. Update NAV
            nav_curve.iloc[i] = nav_curve.iloc[i-1] * (1 + port_return)

        return nav_curve, weights_history

    def _validate_weights(self, weights):
        """
        Ensures weights sum to <= 1 and are all >= 0.
        """
        weights_vec = np.array([weights.get(t, 0.0) for t in self.tickers])
        
        # Enforce no short selling
        weights_vec = np.maximum(weights_vec, 0)
        
        # Enforce no leverage (sum <= 1)
        total_weight = np.sum(weights_vec)
        if total_weight > 1.0:
            weights_vec = weights_vec / total_weight
            
        return weights_vec

class BaseStrategy:
    def compute_weights(self, lookback_data):
        # To be implemented by specific strategies
        raise NotImplementedError
    
class PerformanceAnalytics:
    @staticmethod
    def calculate_metrics(nav_curve, risk_free_rate=0.0):
        """
        Calculates key backtesting metrics for the final report.
        """
        # Daily returns from the NAV curve
        returns = nav_curve.pct_change(fill_method=None).dropna()
        if returns.empty:
            return {"Error": "No returns calculated. Check data alignment."}
                  
        # Cumulative Return
        total_return = (nav_curve.iloc[-1] / nav_curve.iloc[0]) - 1
        
        # Annualized Volatility (assuming 252 trading days)
        ann_vol = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        mean_return = returns.mean() * 252
        sharpe = (mean_return - risk_free_rate) / ann_vol if ann_vol != 0 else 0
        
        # Maximum Drawdown
        rolling_max = nav_curve.cummax()
        drawdown = (nav_curve - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        return {
            "Cumulative Return": total_return,
            "Annualized Vol": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd
        }

class BenchmarkSMA(BaseStrategy):
    def __init__(self, tickers, short_window=20, long_window=50):
        self.tickers = tickers
        self.short_window = short_window
        self.long_window = long_window

    def compute_weights(self, data_slice):
        """
        Logic for Benchmark 1: SMA(20) > SMA(50)
        """
        if len(data_slice) < self.long_window:
            return {} # Not enough data yet, hold cash

        # Calculate means for the last N rows
        short_sma = data_slice.tail(self.short_window).mean()
        long_sma = data_slice.tail(self.long_window).mean()
        
        # Identify stocks meeting the criteria
        selected_stocks = short_sma[short_sma > long_sma].index.tolist()
        
        if not selected_stocks:
            return {} # Hold everything in cash [cite: 81]
            
        # Uniform weighting [cite: 81]
        weight_per_stock = 1.0 / len(selected_stocks)
        return {ticker: weight_per_stock for ticker in selected_stocks}


class SingleStockStrategy(BaseStrategy):
    def __init__(self, ticker, method='momentum', window=20):
        self.ticker = ticker
        self.method = method
        self.window = window

    def compute_weights(self, data_slice):
        # Dynamic window: only require 200 days for the filtered strategy
        required_window = 200 if self.method == 'trend_filtered_mom' else 20
        
        if len(data_slice) < required_window:
            return {}

        prices = data_slice[self.ticker]
        current_price = prices.iloc[-1]
        
        # --- Classic Momentum ---
        if self.method == 'momentum':
            # Buy if price is higher than it was [window] days ago
            if current_price > prices.iloc[-self.window]:
                return {self.ticker: 1.0}

        # --- Classic Mean Reversion ---
        elif self.method == 'mean_reversion':
            # Buy if price is 5% below its 20-day average
            sma_20 = prices.tail(self.window).mean()
            if current_price < (sma_20 * 0.95):
                return {self.ticker: 1.0}
        
        # --- Variation 1: Buy & Hold ---
        elif self.method == 'buy_hold':
            return {self.ticker: 1.0}

        # --- Variation 2: Trend-Filtered Momentum ---
        elif self.method == 'trend_filtered_mom':
            sma_200 = prices.tail(200).mean()
            mom_signal = current_price > prices.iloc[-self.window]
            if mom_signal and current_price > sma_200:
                return {self.ticker: 1.0}

        # --- Variation 3: RSI Mean Reversion ---
        elif self.method == 'rsi_reversion':
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9) # Epsilon to avoid div by zero
            rsi = 100 - (100 / (1 + rs))
            
            if rsi.iloc[-1] < 30: # Oversold
                return {self.ticker: 1.0}
            elif rsi.iloc[-1] > 70: # Overbought
                return {} # Exit to cash

        return {}
    
class BenchmarkTopK(BaseStrategy):
    def __init__(self, tickers, lookback=30, top_k=10):
        self.tickers = tickers
        self.lookback = lookback
        self.top_k = top_k

    def compute_weights(self, data_slice):
        if len(data_slice) < self.lookback + 1:
            return {}
        
        # Calculate 30-day trailing return [cite: 82]
        returns_30d = (data_slice.iloc[-1] / data_slice.iloc[-self.lookback]) - 1
        
        # Rank and select top K [cite: 83]
        top_performers = returns_30d.nlargest(self.top_k).index.tolist()
        
        # Equal weighting [cite: 84]
        weight_per_stock = 1.0 / self.top_k
        return {ticker: weight_per_stock for ticker in top_performers}

class StrategyB_FilteredReversion(BaseStrategy):
    def __init__(self, tickers, lookback=20, top_k=10):
        self.tickers = tickers
        self.lookback = lookback
        self.top_k = top_k

    def compute_weights(self, data_slice):
        if len(data_slice) < self.lookback:
            return {}
        
        # 1. Selection: Pick stocks with consistent upward growth (low volatility momentum)
        returns = (data_slice.iloc[-1] / data_slice.iloc[-self.lookback]) - 1
        volatility = data_slice.pct_change().tail(self.lookback).std()
        
        # Risk-adjusted momentum score
        ra_score = returns / (volatility + 1e-6)
        selected = ra_score.nlargest(self.top_k).index
        
        # 2. Weighting: Risk-adjusted (Inverse Volatility) [cite: 71, 73]
        selected_vols = volatility[selected]
        inv_vols = 1.0 / (selected_vols + 1e-6)
        
        weights = inv_vols / inv_vols.sum()
        return weights.to_dict()
    
class RiskAdjustedStrategy(BaseStrategy):
    def __init__(self, tickers, window=20):
        self.tickers = tickers
        self.window = window

    def compute_weights(self, data_slice):
        if len(data_slice) < self.window + 1:
            return {}

        # 1. Calculate rolling daily returns
        returns = data_slice.pct_change().tail(self.window)
        
        # 2. Estimate volatility (rolling standard deviation) [cite: 70]
        volatility = returns.std()
        
        # 3. Compute raw weights (1 / sigma) [cite: 71, 73]
        # Avoid division by zero with a small epsilon
        raw_weights = 1.0 / (volatility + 1e-6)
        
        # 4. Normalize weights so sum is <= 1 [cite: 74]
        total_raw_weight = raw_weights.sum()
        normalized_weights = raw_weights / total_raw_weight
        
        return normalized_weights.to_dict()
    
class VolAdjustedMomentum(BaseStrategy):
    def __init__(self, tickers, lookback=30, top_k=10, vol_window=20):
        self.tickers = tickers
        self.lookback = lookback
        self.top_k = top_k
        self.vol_window = vol_window

    def compute_weights(self, data_slice):
        if len(data_slice) < max(self.lookback, self.vol_window) + 1:
            return {}

        # 1. Momentum: Calculate trailing returns over the lookback (30 days)
        returns_30d = data_slice.pct_change(self.lookback).iloc[-1]
        
        # 2. Selection: Pick Top K performers (Benchmark 2 style) [cite: 83]
        top_performers = returns_30d.nlargest(self.top_k).index
        
        # 3. Volatility Filter: Calculate 20-day rolling volatility for selected stocks
        recent_returns = data_slice[top_performers].pct_change().tail(self.vol_window)
        vols = recent_returns.std()
        
        # 4. Weighting: Inverse Volatility (1 / sigma) [cite: 71, 73]
        # This rewards "smooth" momentum and punishes "erratic" spikes
        inv_vols = 1.0 / (vols + 1e-6)
        weights = inv_vols / inv_vols.sum()
        
        return weights.to_dict()
    

if __name__ == "__main__":
    # Initialize Engine
    engine = BacktestEngine(data_path="projects/231_backtesting_simulation/nasdaq100_daily_5y.csv")
    results = {}

    # Define all portfolio strategies
    portfolio_strats = {
        "Benchmark 1 (SMA)": BenchmarkSMA(engine.tickers),
        "Benchmark 2 (Top K)": BenchmarkTopK(engine.tickers),
        "Strategy A (Vol-Mom)": VolAdjustedMomentum(engine.tickers),
        "Strategy B (MR-Risk)": StrategyB_FilteredReversion(engine.tickers)
    }

    print("--- Portfolio Level Results ---")
    for name, strat in portfolio_strats.items():
        nav, weights_log = engine.run(strat)
        metrics = PerformanceAnalytics.calculate_metrics(nav)
        results[name] = metrics
        
        # Save logs for Deliverable 1 
        weights_log.to_csv(f"{name.replace(' ', '_')}_weights.csv")
        
        print(f"\n{name}:")
        for m, v in metrics.items():
            print(f"  {m}: {v:.4f}")

    # Complete Deliverable 3: Run all 5 single-stock variations for WDC
    ticker_to_test = 'WDC'
    single_variations = [
        ('Buy Hold', 'buy_hold'),
        ('Momentum', 'momentum'),
        ('Mean Reversion', 'mean_reversion'),
        ('Trend Filtered Mom', 'trend_filtered_mom'),
        ('RSI Reversion', 'rsi_reversion')
    ]

    print(f"\n--- Single Stock Results: {ticker_to_test} ---")
    for name, method in single_variations:
        strat = SingleStockStrategy(ticker_to_test, method=method)
        nav, _ = engine.run(strat)
        stats = PerformanceAnalytics.calculate_metrics(nav)
        print(f"{name} Sharpe: {stats['Sharpe Ratio']:.4f}")