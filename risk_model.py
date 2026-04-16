import yfinance as yf
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

TICKER = 'AVGO'
S0 = 351.57
K  = 330.00
T  = 30 / 252
RF = 0.045
N_PATHS = 10_000
TRADING_DAYS_IN_SIM = 30
np.random.seed(42)

print('Fetching data...')
hist_5y = yf.download(TICKER, period='5y', auto_adjust=True, progress=False)
hist_2y = yf.download(TICKER, period='2y', auto_adjust=True, progress=False)
close_5y = hist_5y['Close'].dropna().squeeze()
close_2y = hist_2y['Close'].dropna().squeeze()
print(f'5y rows: {len(close_5y)}, 2y rows: {len(close_2y)}')
print(f'5y range: {close_5y.index[0].date()} to {close_5y.index[-1].date()}')
print(f'2y range: {close_2y.index[0].date()} to {close_2y.index[-1].date()}')

# MODEL 1
prices_5y = close_5y.values
rolling_30d_returns = (prices_5y[30:] - prices_5y[:-30]) / prices_5y[:-30]
drawdown_threshold = (K - S0) / S0
n_below = int(np.sum(rolling_30d_returns <= drawdown_threshold))
n_total = len(rolling_30d_returns)
hist_prob = n_below / n_total * 100
print(f'M1: threshold={drawdown_threshold*100:.4f}%, windows={n_total}, below={n_below}, prob={hist_prob:.4f}%')

# MODEL 2
log_rets_2y = np.log(close_2y.values[1:] / close_2y.values[:-1])
mu_daily = log_rets_2y.mean()
sigma_daily = log_rets_2y.std(ddof=1)
mu_annual = mu_daily * 252
sigma_annual = sigma_daily * np.sqrt(252)
print(f'mu_annual={mu_annual*100:.4f}%, sigma_annual={sigma_annual*100:.4f}%')

dt = 1
Z = np.random.standard_normal((N_PATHS, TRADING_DAYS_IN_SIM))
daily_factor = np.exp((mu_daily - 0.5 * sigma_daily**2) * dt + sigma_daily * np.sqrt(dt) * Z)
price_paths = np.empty((N_PATHS, TRADING_DAYS_IN_SIM + 1))
price_paths[:, 0] = S0
for t in range(TRADING_DAYS_IN_SIM):
    price_paths[:, t + 1] = price_paths[:, t] * daily_factor[:, t]
min_prices = price_paths[:, 1:].min(axis=1)
paths_below = int(np.sum(min_prices <= K))
gbm_prob = paths_below / N_PATHS * 100
print(f'M2: paths_below={paths_below}, prob={gbm_prob:.4f}%')

# MODEL 3
log_rets_5y = np.log(close_5y.values[1:] / close_5y.values[:-1])
sigma_30d = log_rets_5y[-30:].std(ddof=1) * np.sqrt(252)
d1 = (np.log(S0 / K) + (RF + 0.5 * sigma_30d**2) * T) / (sigma_30d * np.sqrt(T))
d2 = d1 - sigma_30d * np.sqrt(T)
bs_prob = norm.cdf(-d2) * 100
print(f'M3: sigma_30d={sigma_30d*100:.4f}%, d1={d1:.6f}, d2={d2:.6f}, prob={bs_prob:.4f}%')

avg = (hist_prob + gbm_prob + bs_prob) / 3
print(f'AVG: {avg:.4f}%')

# Store results for table
print(f'RESULTS|{hist_prob:.4f}|{gbm_prob:.4f}|{bs_prob:.4f}|{avg:.4f}')
print(f'PARAMS|{drawdown_threshold*100:.4f}|{n_total}|{n_below}|{mu_annual*100:.4f}|{sigma_annual*100:.4f}|{paths_below}|{sigma_30d*100:.4f}|{d1:.6f}|{d2:.6f}')