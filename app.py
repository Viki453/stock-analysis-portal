import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import backtrader as bt
import warnings

# --- Suppress all warnings for a cleaner output ---
warnings.filterwarnings('ignore')

# ===================================================================
# --- CORE ANALYSIS FUNCTION (Cached for Speed) ---
# ===================================================================
# @st.cache_data tells Streamlit to not re-run this function
# if the inputs (ticker, start_date) haven't changed.
@st.cache_data
def analyze_stock(ticker, start_date='2010-01-01'):
    """
    Runs the full analysis (Features, Model) for a single stock.
    Returns the analysis results or None if it fails.
    """
    print(f"\n--- Caching Analysis for {ticker} ---")
    
    # --- 1. GET DATA ---
    try:
        data = yf.download(ticker, start=start_date, end=None, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
    except Exception:
        return None

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
        return None

    # --- 4. MODELING (NO TUNING) ---
    features = ['SMA_50', 'SMA_200', 'RSI', 'Lag_Return_5', 'MACD']
    X = data[features]
    y = data['Target']

    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    if len(X_test) == 0:
        return None

    # Train Random Forest
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_rf.fit(X_train, y_train)
    preds_rf = model_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, preds_rf)

    # Train XGBoost
    model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                                  objective='binary:logistic', 
                                  eval_metric='logloss', random_state=42)
    model_xgb.fit(X_train, y_train)
    preds_xgb = model_xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, preds_xgb)
    
    # We will use XGBoost for the backtest
    test_data_for_bt = data.iloc[split_index:].copy()
    test_data_for_bt['prediction'] = preds_xgb

    return {
        'ticker': ticker,
        'test_data_for_bt': test_data_for_bt,
        'accuracy_xgb': acc_xgb,
        'accuracy_rf': acc_rf
    }

# ===================================================================
# --- BACKTESTING FUNCTION ---
# ===================================================================
def run_backtest(ticker, test_data_for_bt, starting_cash):
    """
    Runs the backtest on the pre-analyzed data.
    """
    class MLStrategy(bt.Strategy):
        def __init__(self):
            self.prediction = self.data.prediction
        def next(self):
            if not self.position and self.prediction[0] == 1:
                self.buy()
            elif self.position and self.prediction[0] == 0:
                self.sell()
    
    class PandasDataWithPred(bt.feeds.PandasData):
        lines = ('prediction',)
        params = (('prediction', -1),)

    cerebro = bt.Cerebro()
    data_feed = PandasDataWithPred(dataname=test_data_for_bt)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(MLStrategy)
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    try:
        results = cerebro.run()
    except Exception:
        return None
        
    # Extract results
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    
    returns_analysis = strat.analyzers.returns.get_analysis()
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    drawdown_analysis = strat.analyzers.drawdown.get_analysis()

    return {
        'Ticker': ticker,
        'Starting_Cash': starting_cash,
        'Final_Value': final_value,
        'Profit_Loss': final_value - starting_cash,
        'CAGR_%': returns_analysis.get('rnorm100'),
        'Sharpe_Ratio': sharpe_analysis.get('sharperatio'),
        'Max_Drawdown_%': drawdown_analysis.get('max', {}).get('drawdown')
    }

# ===================================================================
# --- STREAMLIT WEB APP UI (NEW ATTRACTIVE VERSION) ---
# ===================================================================

# Set page config for a dark theme and wide layout
st.set_page_config(layout="wide", page_title="ML Stock Portal", page_icon="📈")

# Custom CSS for a more polished look
st.markdown("""
    <style>
    /* Main title */
    .stApp > header {
        background-color: transparent;
    }
    h1 {
        color: #00AEEF; /* Bright blue for the main title */
        font-weight: 700;
    }
    /* Sidebar */
    .css-18e3th9 {
        background-color: #0E1117 !important;
        border-right: 1px solid #262730;
    }
    .css-1d391kg {
        background-color: #0E1117;
        border: 1px solid #262730;
    }
    /* Metric cards */
    .stMetric {
        background-color: #1a1a2e; /* Dark blue/purple card */
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #4a4a6a;
    }
    .stMetric > div {
        color: #ffffff; /* White label */
    }
    .stMetric > div > div {
        color: #00AEEF; /* Blue value */
    }
    .stMetric > div > div:last-child { /* Help text */
        color: #a9a9a9;
    }
    /* Dataframe */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)


st.title("📈 ML Stock Analysis Portal")
st.caption("Exploring Small, Mid, and Large-Cap Stocks with Machine Learning | By Vikrant, Anshul, & Raghul")

# --- 1. SIDEBAR CONTROLS ---
st.sidebar.title("Controls")
investment_amount = st.sidebar.number_input(
    "Enter Initial Investment Amount (INR)", 
    min_value=100.0, 
    value=10000.0, 
    step=1000.0,
    help="This amount will be used for all backtest simulations."
)

# Dictionary to map segment names to filenames
SEGMENT_FILES = {
    'Large-Cap': 'nifty50.csv',
    'Mid-Cap': 'nifty_midcap150.csv',
    'Small-Cap': 'nifty_smallcap250.csv'
}

# --- 2. TABS FOR EACH REQUEST ---
tab1, tab2, tab3 = st.tabs([
    "🏆 Best Stock by Segment", 
    "🔍 Single Stock Stats", 
    "📊 Collective Analysis"
])

# --- TAB 1: Best Stock by Segment (Request #3) ---
with tab1:
    st.header("🏆 Best Performing Stock by Segment")
    st.markdown("Run the analysis on a full market segment to find the top-performing stock based on our model.")
    
    segment = st.radio(
        "Select Market Segment:", 
        list(SEGMENT_FILES.keys()), 
        horizontal=True,
        key="segment_radio"
    )
    
    if st.button(f"Analyze {segment}", type="primary"):
        csv_filename = SEGMENT_FILES[segment]
        
        try:
            stock_df = pd.read_csv(csv_filename)
        except FileNotFoundError:
            st.error(f"File not found: {csv_filename}. Please make sure it's in the same directory as app.py")
            st.stop()

        all_results = []
        progress_bar = st.progress(0, text=f"Analyzing {segment} stocks...")
        
        with st.spinner(f"Analyzing {len(stock_df['symbol'])} stocks... This may take a few minutes."):
            for i, ticker in enumerate(stock_df['symbol']):
                model_output = analyze_stock(ticker)
                if model_output:
                    backtest_result = run_backtest(
                        ticker, 
                        model_output['test_data_for_bt'], 
                        investment_amount
                    )
                    if backtest_result:
                        backtest_result['Accuracy_XGB'] = model_output['accuracy_xgb']
                        all_results.append(backtest_result)
                
                progress_bar.progress((i + 1) / len(stock_df['symbol']), text=f"Analyzing: {ticker}")
        
        progress_bar.empty()
        
        if all_results:
            final_df = pd.DataFrame(all_results)
            final_df.sort_values(by='Profit_Loss', ascending=False, inplace=True)
            final_df.set_index('Ticker', inplace=True)
            
            # Show the best one
            best_stock = final_df.iloc[0]
            st.subheader(f"🥇 Top Performer in {segment}")
            
            # Use color for profit/loss
            profit_loss_str = f"₹{best_stock['Profit_Loss']:,.2f}"
            if best_stock['Profit_Loss'] > 0:
                st.success(f"**{best_stock.name}** is the top-performing stock, with a simulated profit of **{profit_loss_str}**.")
            else:
                st.error(f"**{best_stock.name}** had the best performance, but still resulted in a loss of **{profit_loss_str}**.")

            # Metric Cards
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Value", f"₹{best_stock['Final_Value']:,.2f}", 
                        f"Initial: ₹{best_stock['Starting_Cash']:,.2f}")
            col2.metric("Sharpe Ratio", f"{best_stock['Sharpe_Ratio']:.3f}" if best_stock['Sharpe_Ratio'] is not None else "N/A")
            col3.metric("XGBoost Accuracy", f"{best_stock['Accuracy_XGB'] * 100:.1f}%")

            # Show the full table
            st.subheader(f"Full Results for {segment}")
            st.dataframe(final_df.style.format({
                'Starting_Cash': '₹{:,.2f}',
                'Final_Value': '₹{:,.2f}',
                'Profit_Loss': '₹{:,.2f}',
                'CAGR_%': '{:.2f}%',
                'Sharpe_Ratio': '{:.3f}',
                'Max_Drawdown_%': '{:.2f}%',
                'Accuracy_XGB': '{:.1%}'
            }).highlight_max(subset=['Profit_Loss', 'Sharpe_Ratio'], color='#004d00') # Dark green highlight
              .highlight_min(subset=['Profit_Loss', 'Sharpe_Ratio'], color='#660000') # Dark red highlight
            )
        else:
            st.warning("No stocks in this segment could be successfully analyzed.")

# --- TAB 2: Stats for a specific stock (Request #2) ---
with tab2:
    st.header("🔍 Stats for a Specific Stock")
    st.markdown("Look up the simulated performance of any single stock using our model.")
    
    ticker = st.text_input("Enter stock ticker (e.g., RELIANCE.NS, HAVELLS.NS, SUZLON.NS)", key="single_stock_input")
    
    if st.button("Analyze Stock", type="primary"):
        if ticker:
            with st.spinner(f"Analyzing {ticker}..."):
                model_output = analyze_stock(ticker)
                
                if model_output:
                    backtest_result = run_backtest(
                        ticker, 
                        model_output['test_data_for_bt'], 
                        investment_amount
                    )
                    
                    if backtest_result:
                        st.subheader(f"Results for {ticker}")
                        
                        # Use color for profit/loss
                        profit_loss_str = f"₹{backtest_result['Profit_Loss']:,.2f}"
                        if backtest_result['Profit_Loss'] > 0:
                            st.success(f"Simulated run for **{ticker}** resulted in a profit of **{profit_loss_str}**.")
                        else:
                            st.error(f"Simulated run for **{ticker}** resulted in a loss of **{profit_loss_str}**.")

                        # Metric Cards
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Final Value", f"₹{backtest_result['Final_Value']:,.2f}", 
                                    f"Initial: ₹{backtest_result['Starting_Cash']:,.2f}")
                        col2.metric("Sharpe Ratio", f"{backtest_result['Sharpe_Ratio']:.3f}" if backtest_result['Sharpe_Ratio'] is not None else "N/A")
                        col3.metric("XGBoost Accuracy", f"{model_output['accuracy_xgb'] * 100:.1f}%")
                        col4.metric("RF Accuracy", f"{model_output['accuracy_rf'] * 100:.1f}%")
                        
                        with st.expander("Show Raw Backtest Data"):
                            st.json(backtest_result)
                    else:
                        st.error(f"Backtest failed for {ticker}.")
                else:
                    st.error(f"Could not analyze {ticker}. The ticker may be invalid or have insufficient data.")
        else:
            st.warning("Please enter a ticker.")

# --- TAB 3: Collective Analysis (Request #1) ---
with tab3:
    st.header("📊 Collective Analysis (Best Overall)")
    st.markdown("Run the analysis across all segments to find the single best-performing stock.")
    
    if st.button("Find Best Buy Overall", type="primary"):
        all_results = []
        total_stocks = 0
        all_files = list(SEGMENT_FILES.items())
        
        # First, count total stocks for the progress bar
        for segment, csv_filename in all_files:
            try:
                total_stocks += len(pd.read_csv(csv_filename))
            except FileNotFoundError:
                st.error(f"File not found: {csv_filename}. Skipping this segment.")
                continue
        
        if total_stocks == 0:
            st.error("No stock files found. Please add large_cap_stocks.csv, mid_cap_stocks.csv, and small_cap_stocks.csv")
            st.stop()

        progress_bar = st.progress(0, text="Initializing...")
        current_stock_count = 0
        
        with st.spinner("Analyzing all segments... This will take time."):
            for segment, csv_filename in all_files:
                st.write(f"Analyzing {segment}...")
                try:
                    stock_df = pd.read_csv(csv_filename)
                except FileNotFoundError:
                    continue # Already warned
                    
                for ticker in stock_df['symbol']:
                    model_output = analyze_stock(ticker)
                    if model_output:
                        backtest_result = run_backtest(
                            ticker, 
                            model_output['test_data_for_bt'], 
                            investment_amount
                        )
                        if backtest_result:
                            backtest_result['Segment'] = segment
                            all_results.append(backtest_result)
                    
                    current_stock_count += 1
                    progress_bar.progress(current_stock_count / total_stocks, text=f"Analyzed: {ticker}")
        
        progress_bar.empty()
        
        if all_results:
            final_df = pd.DataFrame(all_results)
            final_df.sort_values(by='Profit_Loss', ascending=False, inplace=True)
            final_df.set_index('Ticker', inplace=True)
            
            # Show the best one
            best_overall = final_df.iloc[0]
            st.subheader(f"🥇 Best Buy Overall (All Segments)")
            
            profit_loss_str = f"₹{best_overall['Profit_Loss']:,.2f}"
            if best_overall['Profit_Loss'] > 0:
                st.success(f"**{best_overall.name}** ({best_overall['Segment']}) is the top-performing stock, with a simulated profit of **{profit_loss_str}**.")
            else:
                st.error(f"**{best_overall.name}** ({best_overall['Segment']}) had the best performance, but still resulted in a loss of **{profit_loss_str}**.")

            # Metric Cards
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Value", f"₹{best_overall['Final_Value']:,.2f}",
                        f"Initial: ₹{best_overall['Starting_Cash']:,.2f}")
            col2.metric("Sharpe Ratio", f"{best_overall['Sharpe_Ratio']:.3f}" if best_overall['Sharpe_Ratio'] is not None else "N/A")
            col3.metric("Segment", best_overall['Segment'])

            # Show the full table
            st.subheader("Full Combined Results (Top 50)")
            st.dataframe(final_df.head(50).style.format({
                'Starting_Cash': '₹{:,.2f}',
                'Final_Value': '₹{:,.2f}',
                'Profit_Loss': '₹{:,.2f}',
                'CAGR_%': '{:.2f}%',
                'Sharpe_Ratio': '{:.3f}',
                'Max_Drawdown_%': '{:.2f}%'
            }).highlight_max(subset=['Profit_Loss', 'Sharpe_Ratio'], color='#004d00')
              .highlight_min(subset=['Profit_Loss', 'Sharpe_Ratio'], color='#660000')
            )
        else:
            st.warning("No stocks could be successfully analyzed.")