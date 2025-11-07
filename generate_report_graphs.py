import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import backtrader as bt
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# --- Suppress all warnings for a cleaner output ---
warnings.filterwarnings('ignore')

# ===================================================================
# --- CORE ANALYSIS FUNCTIONS (from app.py) ---
# ===================================================================

# @st.cache_data is removed, as this isn't a streamlit app
def analyze_stock(ticker, start_date='2010-01-01'):
    """
    Runs the full analysis (Features, Model) for a single stock.
    Returns the analysis results or None if it fails.
    """
    # --- 1. GET DATA ---
    try:
        data = yf.download(ticker, start=start_date, end=None, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
    except Exception: return None

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

    if len(data) < 252: return None

    # --- 4. MODELING (NO TUNING) ---
    features = ['SMA_50', 'SMA_200', 'RSI', 'Lag_Return_5', 'MACD']
    X = data[features]
    y = data['Target']

    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    if len(X_test) == 0: return None

    # Train Random Forest
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_rf.fit(X_train, y_train)
    preds_rf = model_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, preds_rf)
    precision_rf = precision_score(y_test, preds_rf, zero_division=0) # Added Precision

    # Train XGBoost
    model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                                  objective='binary:logistic', 
                                  eval_metric='logloss', random_state=42)
    model_xgb.fit(X_train, y_train)
    preds_xgb = model_xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, preds_xgb)
    precision_xgb = precision_score(y_test, preds_xgb, zero_division=0) # Added Precision
    
    test_data_for_bt = data.iloc[split_index:].copy()
    test_data_for_bt['prediction'] = preds_xgb

    return {
        'ticker': ticker,
        'test_data_for_bt': test_data_for_bt,
        'accuracy_xgb': acc_xgb,
        'precision_xgb': precision_xgb, # Return precision
        'accuracy_rf': acc_rf,
        'precision_rf': precision_rf # Return precision
    }

def run_backtest(ticker, test_data_for_bt, starting_cash):
    """
    Runs the backtest on the pre-analyzed data.
    """
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
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    try: results = cerebro.run()
    except Exception: return None
        
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    returns = strat.analyzers.returns.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()

    return {
        'Ticker': ticker,
        'Starting_Cash': starting_cash,
        'Final_Value': final_value,
        'Profit_Loss': final_value - starting_cash,
        'CAGR_%': returns.get('rnorm100'),
        'Sharpe_Ratio': sharpe.get('sharperatio'),
        'Max_Drawdown_%': drawdown.get('max', {}).get('drawdown')
    }

# ===================================================================
# --- MAIN SCRIPT: Run Full Analysis & Generate Graphs ---
# ===================================================================

# --- 1. Set Up Analysis ---
STARTING_CASH = 10000.0
SEGMENT_FILES = {
    'Large-Cap': 'nifty50.csv',
    'Mid-Cap': 'nifty_midcap150.csv',
    'Small-Cap': 'nifty_smallcap250.csv'
}
all_results = []
print("--- Starting Full Batch Analysis ---")
print("This will take several minutes...")

# --- 2. Run Full Analysis to Gather All Data ---
for segment, csv_filename in SEGMENT_FILES.items():
    print(f"Analyzing {segment}...")
    try:
        stock_df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Warning: {csv_filename} not found. Skipping.")
        continue
        
    for ticker in stock_df['symbol']:
        print(f"  ... processing {ticker}")
        model_output = analyze_stock(ticker)
        if model_output:
            backtest_result = run_backtest(
                ticker, 
                model_output['test_data_for_bt'], 
                STARTING_CASH
            )
            if backtest_result:
                # Combine results
                full_result = {
                    **backtest_result,
                    'Segment': segment,
                    'Accuracy_XGB': model_output['accuracy_xgb'],
                    'Precision_XGB': model_output['precision_xgb'],
                    'Accuracy_RF': model_output['accuracy_rf'],
                    'Precision_RF': model_output['precision_rf']
                }
                all_results.append(full_result)

print("\n--- Analysis Complete. Generating Graphs. ---")
final_df = pd.DataFrame(all_results)

# --- 3. Generate Graph 1: Model Precision by Segment ---
try:
    print("Generating graph_1_model_precision.png ...")
    # Calculate average precision
    avg_precision = final_df.groupby('Segment')[['Precision_RF', 'Precision_XGB']].mean().reset_index()
    # Melt the dataframe for Seaborn
    precision_melted = avg_precision.melt(id_vars='Segment', var_name='Model', value_name='Avg. Precision')
    precision_melted['Model'] = precision_melted['Model'].replace({
        'Precision_RF': 'Random Forest',
        'Precision_XGB': 'XGBoost'
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=precision_melted, x='Segment', y='Avg. Precision', hue='Model', palette='Blues')
    plt.title('Graph 1: Average Model Precision by Segment', fontsize=16)
    plt.ylabel('Average Precision (Higher is better)')
    plt.xlabel('Market Segment')
    plt.legend()
    plt.savefig('graph_1_model_precision.png')
    plt.close()
    print("... Done.")
except Exception as e:
    print(f"Error generating Graph 1: {e}")

# --- 4. Generate Graph 2: Top 5 Mid-Cap Performers ---
try:
    print("Generating graph_2_top_5_mid_cap.png ...")
    mid_cap_df = final_df[final_df['Segment'] == 'Mid-Cap']
    top_5_mid = mid_cap_df.nlargest(5, 'Profit_Loss')
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=top_5_mid, y='Ticker', x='Profit_Loss', color='green', orient='h')
    plt.title('Graph 2: Top 5 Mid-Cap Performers (by Profit/Loss)', fontsize=16)
    plt.xlabel('Total Profit/Loss (INR)')
    plt.ylabel('Stock Ticker')
    plt.savefig('graph_2_top_5_mid_cap.png')
    plt.close()
    print("... Done.")
except Exception as e:
    print(f"Error generating Graph 2: {e}")

# --- 5. Generate Graph 4: Top 10 "Best Buys" (All Segments) ---
try:
    print("Generating graph_4_top_10_all.png ...")
    # Sort by Sharpe Ratio for the "best buy"
    top_10_all = final_df.nlargest(10, 'Sharpe_Ratio')
    
    # Create a color palette for the segments
    palette = {'Large-Cap': 'blue', 'Mid-Cap': 'green', 'Small-Cap': 'red'}
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_10_all, y='Ticker', x='Sharpe_Ratio', hue='Segment', palette=palette, orient='h', dodge=False)
    plt.title('Graph 4: Top 10 "Best Buys" by Risk-Adjusted Return (Sharpe Ratio)', fontsize=16)
    plt.xlabel('Sharpe Ratio (Higher is better)')
    plt.ylabel('Stock Ticker')
    plt.legend(title='Segment')
    plt.savefig('graph_4_top_10_all.png')
    plt.close()
    print("... Done.")
except Exception as e:
    print(f"Error generating Graph 4: {e}")

print("\n--- All graphs saved as .png files in the current directory. ---")


# ===================================================================
# --- NEW: FINAL REPORT SUMMARY FOR YOUR TABLES ---
# ===================================================================

print("\n\n--- REPORT SUMMARY (COPY THIS TO YOUR DOCUMENT) ---")

# --- Data for Table 1: Model Performance ---
print("\n--- Data for Table 1: Average Model Performance ---")
try:
    # Calculate average performance
    avg_performance = final_df.groupby('Segment')[['Accuracy_RF', 'Precision_RF', 'Accuracy_XGB', 'Precision_XGB']].mean()
    
    # Re-order segments
    avg_performance = avg_performance.reindex(['Large-Cap', 'Mid-Cap', 'Small-Cap'])
    
    # Format for printing
    avg_performance['Accuracy_RF'] = (avg_performance['Accuracy_RF'] * 100).map('{:.1f}%'.format)
    avg_performance['Precision_RF'] = (avg_performance['Precision_RF'] * 100).map('{:.1f}%'.format)
    avg_performance['Accuracy_XGB'] = (avg_performance['Accuracy_XGB'] * 100).map('{:.1f}%'.format)
    avg_performance['Precision_XGB'] = (avg_performance['Precision_XGB'] * 100).map('{:.1f}%'.format)
    
    print(avg_performance)
    
    print("\nAction: Copy the percentages above into Table 1 of your report.")

except Exception as e:
    print(f"Error generating Table 1 data: {e}")

print("\n--- Data for Tables 2, 3, 4, 5: Run the Streamlit App ---")
print("To get the data for the other tables, run the Streamlit app:")
print("\n  streamlit run app.py\n")
print("1. Use the 'Best Stock by Segment' tab for Tables 2, 3, and 4.")
print("2. Use the 'Collective Analysis' tab for Table 5.")
print("-------------------------------------------------------")