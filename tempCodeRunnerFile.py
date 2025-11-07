import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import backtrader as bt
import matplotlib
import warnings

# --- Suppress all warnings for a cleaner output ---
warnings.filterwarnings('ignore')
matplotlib.use('Agg') # Use Agg backend for saving file

# --- !!! SET YOUR BEST STOCK HERE !!! ---
# After running the other script, put your best stock's ticker here.
BEST_STOCK_TICKER = 'HAVELLS.NS' 
# ----------------------------------------

print(f"--- Generating Equity Curve for {BEST_STOCK_TICKER} ---")

# --- 1. GET DATA ---
try:
    data = yf.download(BEST_STOCK_TICKER, start='2010-01-01', end=None, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
except Exception as e:
    print(f"Error downloading {BEST_STOCK_TICKER}: {e}")
    exit()

# --- 2. FEATURE ENGINEERING ---
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))
data['Lag_Return_5'] = data['Close'].pct_change(5)
ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26

# --- 3. TARGET & CLEAN ---
threshold = 0.003
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'] * (1 + threshold), 1, 0)
data.dropna(inplace=True)

if len(data) < 252:
    print("Not enough data to analyze.")
    exit()

# --- 4. MODELING ---
features = ['SMA_50', 'SMA_200', 'RSI', 'Lag_Return_5', 'MACD']
X = data[features]
y = data['Target']

split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                              objective='binary:logistic', 
                              eval_metric='logloss', random_state=42)
model_xgb.fit(X_train, y_train)
preds_xgb = model_xgb.predict(X_test)

test_data_for_bt = data.iloc[split_index:].copy()
test_data_for_bt['prediction'] = preds_xgb

# --- 5. BACKTESTING ---
class MLStrategy(bt.Strategy):
    def __init__(self): self.prediction = self.data.prediction
    def next(self):
        if not self.position and self.prediction[0] == 1: self.buy()
        elif self.position and self.prediction[0] == 0: self.sell()

class PandasDataWithPred(bt.feeds.PandasData):
    lines = ('prediction',); params = (('prediction', -1),)

cerebro = bt.Cerebro()
data_feed = PandasDataWithPred(dataname=test_data_for_bt)
cerebro.adddata(data_feed)
cerebro.addstrategy(MLStrategy)
cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.001)

print("Running backtest to generate plot...")
# Run the backtest
cerebro.run()

# --- 6. SAVE PLOT ---
print("Saving graph_3_equity_curve.png ...")
plt.figure(figsize=(16, 10)) # Make the plot larger
figure = cerebro.plot(style='candlestick', barup='green', bardown='red', iplot=False)[0][0]
figure.savefig('graph_3_equity_curve.png')
plt.close()
print("... Done.")
print("\n--- Equity curve plot saved as 'graph_3_equity_curve.png' ---")