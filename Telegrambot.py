# dragon_brain_qn1/main.py
import streamlit as st
import pandas as pd
import time
import threading
import queue
import MetaTrader5 as mt5 # Import MT5 for constants

from config import MT5_CONFIG, TELEGRAM_CONFIG, TRADING_SETTINGS, LANGUAGE_SETTINGS
from mt5_integration import connect_mt5, get_market_data, place_order, close_order, get_open_trades, get_account_info
from scalping_engine import calculate_indicators, generate_scalping_signal_dl
from news_sentiment import fetch_news, analyze_sentiment_dl
from prediction_engine import simulate_market_outcomes
from trade_logger import log_trade_to_csv, init_sqlite_db, save_trade_to_db, get_trade_history_from_db
from telegram_alerts import send_telegram_message
from utils import get_localized_text, calculate_position_size, calculate_sl_tp

# Initialize SQLite database
DB_PATH = 'trades.db'
init_sqlite_db(DB_PATH)

# Global flag for kill switch
kill_switch_active = False

# Queue for real-time data updates to Streamlit
data_queue = queue.Queue()

def trading_loop(symbol, timeframe, scalping_interval_seconds, news_interval_seconds, account_type, risk_level, risk_per_trade_percent):
    """
    Main trading loop that runs in a separate thread.
    Fetches data, generates signals, and executes trades.
    """
    global kill_switch_active

    st.session_state.trading_status = get_localized_text('running_trading_loop', st.session_state.language)
    st.session_state.last_scalping_check = 0
    st.session_state.last_news_check = 0

    while st.session_state.is_trading_active and not kill_switch_active:
        current_time = time.time()

        # Update account info and open trades periodically for position sizing and dashboard
        account_info = get_account_info()
        open_trades = get_open_trades()
        st.session_state.open_trades_count = len(open_trades)

        if account_info is None:
            st.error(f"🔴 {get_localized_text('failed_get_account_info', st.session_state.language)}")
            time.sleep(5) # Wait before retrying
            continue

        # Scalping Engine Check
        if st.session_state.strategy_mode in ['Scalping', 'Mixed'] and \
           (current_time - st.session_state.last_scalping_check) >= scalping_interval_seconds:
            st.session_state.last_scalping_check = current_time
            st.session_state.current_action = get_localized_text('checking_scalping', st.session_state.language)
            try:
                # Fetch market data for scalping
                bars = get_market_data(symbol, timeframe, 100) # Get last 100 bars for indicator calculation
                if bars is not None and not bars.empty:
                    df = pd.DataFrame(bars)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)

                    # Calculate indicators (still useful for context, even if DL model is used)
                    df_with_indicators = calculate_indicators(df.copy()) # Pass a copy

                    # Generate scalping signal using Deep Learning model (conceptual)
                    scalping_signal = generate_scalping_signal_dl(df_with_indicators)
                    st.session_state.scalping_signal = scalping_signal
                    st.session_state.current_action = f"{get_localized_text('scalping_signal', st.session_state.language)}: {scalping_signal}"

                    # Simulate trade execution based on signal
                    if st.session_state.strategy_mode != 'Manual':
                        if scalping_signal == 'BUY' or scalping_signal == 'SELL':
                            if st.session_state.open_trades_count < TRADING_SETTINGS['max_open_trades']:
                                current_price_tick = mt5.symbol_info_tick(symbol)
                                if current_price_tick is None:
                                    st.session_state.alert_message = f"🔴 {get_localized_text('failed_get_tick_data', st.session_state.language)}"
                                    continue

                                entry_price = current_price_tick.ask if scalping_signal == 'BUY' else current_price_tick.bid
                                # Calculate SL/TP based on ATR and risk level
                                # For a real system, ATR should be calculated from the df_with_indicators
                                # For now, let's use a simplified mock ATR or fixed pip values for SL/TP
                                # Placeholder for ATR calculation:
                                # atr_value = df_with_indicators['atr'].iloc[-1] if 'atr' in df_with_indicators.columns else 0.0002 # Example default
                                # For this demo, let's use a fixed pip value for SL/TP distance
                                sl_pips = TRADING_SETTINGS['default_sl_pips']
                                tp_pips = TRADING_SETTINGS['default_tp_pips']

                                sl, tp = calculate_sl_tp(symbol, entry_price, scalping_signal, sl_pips, tp_pips)

                                # Calculate position size
                                volume = calculate_position_size(
                                    account_info.balance,
                                    risk_per_trade_percent,
                                    sl,
                                    entry_price,
                                    symbol,
                                    scalping_signal
                                )
                                # Ensure volume is within MT5 limits and positive
                                if volume <= 0 or volume > mt5.symbol_info(symbol).volume_max:
                                    st.session_state.alert_message = f"🔴 {get_localized_text('invalid_volume', st.session_state.language)}: {volume}"
                                    continue
                                volume = round(volume, TRADING_SETTINGS['volume_precision']) # Round to appropriate precision

                                order_type = mt5.ORDER_TYPE_BUY if scalping_signal == 'BUY' else mt5.ORDER_TYPE_SELL
                                trade_result = place_order(symbol, order_type, volume, entry_price, sl, tp,
                                                            comment=f"Scalping {scalping_signal} | Risk: {risk_level}")
                                if trade_result:
                                    trade_info = {
                                        "timestamp": pd.Timestamp.now(),
                                        "symbol": symbol,
                                        "type": "BUY" if scalping_signal == 'BUY' else "SELL",
                                        "volume": volume,
                                        "price": trade_result.price,
                                        "reason": f"Scalping Signal: {scalping_signal}",
                                        "status": "OPEN",
                                        "ticket": trade_result.order,
                                        "sl": sl,
                                        "tp": tp
                                    }
                                    log_trade_to_csv(trade_info)
                                    save_trade_to_db(trade_info)
                                    send_telegram_message(f"DRAGON BRAIN QN-1: New {scalping_signal} trade opened on {symbol} (Vol: {volume}) via Scalping Engine! SL: {sl}, TP: {tp}")
                                    st.session_state.alert_message = f"🟢 {get_localized_text('trade_opened', st.session_state.language)}: {scalping_signal} {volume}!"
                                else:
                                    st.session_state.alert_message = f"🔴 {get_localized_text('trade_failed', st.session_state.language)}: {scalping_signal}."
                            else:
                                st.session_state.alert_message = get_localized_text('max_trades_open', st.session_state.language)
                        else:
                            st.session_state.alert_message = get_localized_text('no_scalping_signal', st.session_state.language)
                else:
                    st.session_state.alert_message = get_localized_text('no_market_data', st.session_state.language)

            except Exception as e:
                st.error(f"Error in scalping engine: {e}")
                st.session_state.alert_message = f"🔴 {get_localized_text('scalping_error', st.session_state.language)}: {e}"

        # News Sentiment Sniper Check
        if st.session_state.strategy_mode in ['News', 'Mixed'] and \
           (current_time - st.session_state.last_news_check) >= news_interval_seconds:
            st.session_state.last_news_check = current_time
            st.session_state.current_action = get_localized_text('checking_news_sentiment', st.session_state.language)
            try:
                news_articles = fetch_news()
                if news_articles:
                    sentiment = analyze_sentiment_dl(news_articles[0]['text']) # Analyze the first article for simplicity
                    st.session_state.news_sentiment = sentiment
                    st.session_state.current_action = f"{get_localized_text('news_sentiment', st.session_state.language)}: {sentiment}"

                    # Simulate trade execution based on sentiment
                    if st.session_state.strategy_mode != 'Manual':
                        if sentiment == 'BUY' or sentiment == 'SELL':
                            if st.session_state.open_trades_count < TRADING_SETTINGS['max_open_trades']:
                                current_price_tick = mt5.symbol_info_tick(symbol)
                                if current_price_tick is None:
                                    st.session_state.alert_message = f"🔴 {get_localized_text('failed_get_tick_data', st.session_state.language)}"
                                    continue

                                entry_price = current_price_tick.ask if sentiment == 'BUY' else current_price_tick.bid
                                sl_pips = TRADING_SETTINGS['default_sl_pips']
                                tp_pips = TRADING_SETTINGS['default_tp_pips']
                                sl, tp = calculate_sl_tp(symbol, entry_price, sentiment, sl_pips, tp_pips)

                                volume = calculate_position_size(
                                    account_info.balance,
                                    risk_per_trade_percent,
                                    sl,
                                    entry_price,
                                    symbol,
                                    sentiment
                                )
                                if volume <= 0 or volume > mt5.symbol_info(symbol).volume_max:
                                    st.session_state.alert_message = f"🔴 {get_localized_text('invalid_volume', st.session_state.language)}: {volume}"
                                    continue
                                volume = round(volume, TRADING_SETTINGS['volume_precision'])

                                order_type = mt5.ORDER_TYPE_BUY if sentiment == 'BUY' else mt5.ORDER_TYPE_SELL
                                trade_result = place_order(symbol, order_type, volume, entry_price, sl, tp,
                                                            comment=f"News {sentiment} | Sentiment: {sentiment}")
                                if trade_result:
                                    trade_info = {
                                        "timestamp": pd.Timestamp.now(),
                                        "symbol": symbol,
                                        "type": "BUY" if sentiment == 'BUY' else "SELL",
                                        "volume": volume,
                                        "price": trade_result.price,
                                        "reason": f"News Sentiment: {sentiment}",
                                        "status": "OPEN",
                                        "ticket": trade_result.order,
                                        "sl": sl,
                                        "tp": tp
                                    }
                                    log_trade_to_csv(trade_info)
                                    save_trade_to_db(trade_info)
                                    send_telegram_message(f"DRAGON BRAIN QN-1: New {sentiment} trade opened on {symbol} (Vol: {volume}) via News Engine! SL: {sl}, TP: {tp}")
                                    st.session_state.alert_message = f"🟢 {get_localized_text('trade_opened', st.session_state.language)}: {sentiment} {volume}!"
                                else:
                                    st.session_state.alert_message = f"🔴 {get_localized_text('trade_failed', st.session_state.language)}: {sentiment}."
                            else:
                                st.session_state.alert_message = get_localized_text('max_trades_open', st.session_state.language)
                        else:
                            st.session_state.alert_message = get_localized_text('no_news_signal', st.session_state.language)
                else:
                    st.session_state.alert_message = get_localized_text('no_news_fetched', st.session_state.language)
            except Exception as e:
                st.error(f"Error in news sentiment engine: {e}")
                st.session_state.alert_message = f"🔴 {get_localized_text('news_error', st.session_state.language)}: {e}"

        # Pseudo-Quantum Prediction Engine (conceptual)
        st.session_state.current_action = get_localized_text('simulating_quantum', st.session_state.language)
        try:
            current_signals = {
                "scalping": st.session_state.scalping_signal,
                "news": st.session_state.news_sentiment
            }
            prediction_recommendation = simulate_market_outcomes(current_signals)
            st.session_state.quantum_prediction = prediction_recommendation
            st.session_state.current_action = f"{get_localized_text('quantum_prediction', st.session_state.language)}: {prediction_recommendation}"
        except Exception as e:
            st.error(f"Error in pseudo-quantum prediction engine: {e}")
            st.session_state.alert_message = f"🔴 {get_localized_text('quantum_error', st.session_state.language)}: {e}"


        # Update live data for dashboard
        try:
            data_queue.put({
                "account_info": account_info,
                "open_trades": open_trades,
                "scalping_signal": st.session_state.scalping_signal,
                "news_sentiment": st.session_state.news_sentiment,
                "quantum_prediction": st.session_state.quantum_prediction,
                "current_action": st.session_state.current_action,
                "alert_message": st.session_state.alert_message
            })
            st.session_state.alert_message = "" # Clear alert after displaying
        except Exception as e:
            st.error(f"Error updating live data: {e}")

        time.sleep(1) # Small delay to prevent excessive CPU usage

    st.session_state.trading_status = get_localized_text('trading_stopped', st.session_state.language)
    st.session_state.alert_message = get_localized_text('trading_session_ended', st.session_state.language)
    st.experimental_rerun() # Rerun Streamlit to update UI state


def main_dashboard():
    """
    Streamlit UI for the Dragon Brain QN-1 dashboard.
    """
    st.set_page_config(layout="wide", page_title="Dragon Brain QN-1")

    # Initialize session state variables if not already present
    if 'is_mt5_connected' not in st.session_state:
        st.session_state.is_mt5_connected = False
    if 'mt5_connection_status' not in st.session_state:
        st.session_state.mt5_connection_status = get_localized_text('not_connected', 'en')
    if 'is_trading_active' not in st.session_state:
        st.session_state.is_trading_active = False
    if 'trading_thread' not in st.session_state:
        st.session_state.trading_thread = None
    if 'account_info' not in st.session_state:
        st.session_state.account_info = None
    if 'open_trades' not in st.session_state:
        st.session_state.open_trades = []
    if 'scalping_signal' not in st.session_state:
        st.session_state.scalping_signal = "N/A"
    if 'news_sentiment' not in st.session_state:
        st.session_state.news_sentiment = "N/A"
    if 'quantum_prediction' not in st.session_state:
        st.session_state.quantum_prediction = "N/A"
    if 'current_action' not in st.session_state:
        st.session_state.current_action = get_localized_text('idle', 'en')
    if 'alert_message' not in st.session_state:
        st.session_state.alert_message = ""
    if 'strategy_mode' not in st.session_state:
        st.session_state.strategy_mode = 'Mixed' # Default strategy
    if 'risk_level' not in st.session_state:
        st.session_state.risk_level = 'Medium' # Default risk
    if 'trade_frequency' not in st.session_state:
        st.session_state.trade_frequency = 'Normal' # Default frequency
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    if 'open_trades_count' not in st.session_state:
        st.session_state.open_trades_count = 0
    if 'risk_per_trade_percent' not in st.session_state:
        st.session_state.risk_per_trade_percent = TRADING_SETTINGS['risk_per_trade_percent']

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }
        .main-header {
            font-size: 2.5em;
            font-weight: 700;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            font-weight: 600;
            padding: 10px 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            border: none;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        .stButton>button:active {
            transform: translateY(0);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .stButton>button.st-emotion-cache-1x85k09 { /* Specific for primary button */
            background-color: #4CAF50;
            color: white;
        }
        .stButton>button.st-emotion-cache-1x85k09:hover {
            background-color: #45a049;
        }
        .stButton>button[kind="secondary"] {
            background-color: #f44336;
            color: white;
        }
        .stButton>button[kind="secondary"]:hover {
            background-color: #da190b;
        }
        .st-emotion-cache-1x85k09 { /* General button styling */
            border-radius: 12px !important;
        }
        .st-emotion-cache-1v0bb6z { /* Selectbox styling */
            border-radius: 12px !important;
        }
        .metric-card {
            background-color: #2e2e2e;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            border: 1px solid #444;
        }
        .metric-card h3 {
            color: #aaa;
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        .metric-card .value {
            font-size: 1.8em;
            font-weight: 700;
            color: #4CAF50;
        }
        .metric-card .value.red {
            color: #f44336;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-indicator.green { background-color: #4CAF50; }
        .status-indicator.red { background-color: #f44336; }
        .status-indicator.yellow { background-color: #FFC107; }

        .stAlert {
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .stAlert.success { background-color: #d4edda; color: #155724; border-color: #c3e6cb; }
        .stAlert.error { background-color: #f8d7da; color: #721c24; border-color: #f5c6cb; }
        .stAlert.info { background-color: #d1ecf1; color: #0c5460; border-color: #bee5eb; }

        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size:1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f'<div class="main-header">{get_localized_text("project_name", st.session_state.language)}</div>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="text-align: center; color: #bbb;">{get_localized_text("slogan", st.session_state.language)}</h2>', unsafe_allow_html=True)

    # Language selection
    lang_col1, lang_col2 = st.columns([0.8, 0.2])
    with lang_col2:
        selected_language = st.selectbox(
            get_localized_text('select_language', st.session_state.language),
            options=['en', 'ar'],
            format_func=lambda x: 'English' if x == 'en' else 'العربية',
            key='language_selector'
        )
        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            st.experimental_rerun() # Rerun to apply language changes

    # Display alerts
    if st.session_state.alert_message:
        if "🟢" in st.session_state.alert_message:
            st.success(st.session_state.alert_message)
        elif "🔴" in st.session_state.alert_message:
            st.error(st.session_state.alert_message)
        else:
            st.info(st.session_state.alert_message)
        # st.session_state.alert_message = "" # Clear alert after display

    # MT5 Connection Section
    st.sidebar.header(get_localized_text('mt5_connection', st.session_state.language))
    st.sidebar.markdown(f"**{get_localized_text('status', st.session_state.language)}:** <span class='status-indicator {'green' if st.session_state.is_mt5_connected else 'red'}'></span> {st.session_state.mt5_connection_status}", unsafe_allow_html=True)

    if not st.session_state.is_mt5_connected:
        mt5_login = st.sidebar.text_input(get_localized_text('mt5_login', st.session_state.language), value=MT5_CONFIG['MT5_LOGIN'])
        mt5_password = st.sidebar.text_input(get_localized_text('mt5_password', st.session_state.language), type="password", value=MT5_CONFIG['MT5_PASSWORD'])
        mt5_server = st.sidebar.text_input(get_localized_text('mt5_server', st.session_state.language), value=MT5_CONFIG['MT5_SERVER'])
        mt5_path = st.sidebar.text_input(get_localized_text('mt5_path', st.session_state.language), value=MT5_CONFIG['MT5_PATH'])
        account_type = st.sidebar.selectbox(get_localized_text('account_type', st.session_state.language), ['DEMO', 'REAL'], index=0 if MT5_CONFIG['ACCOUNT_TYPE'] == 'DEMO' else 1)

        if st.sidebar.button(get_localized_text('connect_mt5', st.session_state.language)):
            with st.spinner(get_localized_text('connecting', st.session_state.language)):
                try:
                    connected = connect_mt5(int(mt5_login), mt5_password, mt5_server, mt5_path)
                    if connected:
                        st.session_state.is_mt5_connected = True
                        st.session_state.mt5_connection_status = get_localized_text('connected_success', st.session_state.language)
                        st.session_state.alert_message = get_localized_text('mt5_connected_alert', st.session_state.language)
                        st.experimental_rerun()
                    else:
                        st.session_state.is_mt5_connected = False
                        st.session_state.mt5_connection_status = get_localized_text('connection_failed', st.session_state.language)
                        st.session_state.alert_message = get_localized_text('mt5_connection_failed_alert', st.session_state.language)
                except Exception as e:
                    st.session_state.is_mt5_connected = False
                    st.session_state.mt5_connection_status = f"{get_localized_text('connection_error', st.session_state.language)}: {e}"
                    st.session_state.alert_message = f"🔴 {get_localized_text('mt5_connection_error_alert', st.session_state.language)}: {e}"
    else:
        st.sidebar.success(get_localized_text('mt5_connected', st.session_state.language))
        if st.sidebar.button(get_localized_text('disconnect_mt5', st.session_state.language)):
            # In a real scenario, you'd call mt5.shutdown()
            st.session_state.is_mt5_connected = False
            st.session_state.mt5_connection_status = get_localized_text('disconnected', st.session_state.language)
            st.session_state.is_trading_active = False
            if st.session_state.trading_thread and st.session_state.trading_thread.is_alive():
                st.session_state.trading_thread.join(timeout=1) # Give it a moment to stop
            st.session_state.trading_thread = None
            st.session_state.alert_message = get_localized_text('mt5_disconnected_alert', st.session_state.language)
            st.experimental_rerun()

    # Trading Controls
    st.sidebar.header(get_localized_text('trading_controls', st.session_state.language))
    if st.session_state.is_mt5_connected:
        st.session_state.strategy_mode = st.sidebar.selectbox(
            get_localized_text('strategy_mode', st.session_state.language),
            ['Scalping', 'News', 'Mixed', 'Manual'],
            index=['Scalping', 'News', 'Mixed', 'Manual'].index(st.session_state.strategy_mode)
        )
        st.session_state.risk_level = st.sidebar.selectbox(
            get_localized_text('risk_level', st.session_state.language),
            ['Low', 'Medium', 'High'],
            index=['Low', 'Medium', 'High'].index(st.session_state.risk_level)
        )
        # Map risk level to percentage
        risk_percentage_map = {'Low': 0.005, 'Medium': 0.01, 'High': 0.02} # 0.5%, 1%, 2%
        st.session_state.risk_per_trade_percent = risk_percentage_map[st.session_state.risk_level]

        st.session_state.trade_frequency = st.sidebar.selectbox(
            get_localized_text('trade_frequency', st.session_state.language),
            ['Low', 'Normal', 'High'],
            index=['Low', 'Normal', 'High'].index(st.session_state.trade_frequency)
        )

        scalping_interval_map = {'Low': 60, 'Normal': 30, 'High': 10} # seconds
        news_interval_map = {'Low': 300, 'Normal': 60, 'High': 10} # seconds

        scalping_interval = scalping_interval_map[st.session_state.trade_frequency]
        news_interval = news_interval_map[st.session_state.trade_frequency]

        col_start, col_stop = st.sidebar.columns(2)
        with col_start:
            if st.button(get_localized_text('start_trading', st.session_state.language), disabled=st.session_state.is_trading_active):
                if st.session_state.is_mt5_connected:
                    st.session_state.is_trading_active = True
                    global kill_switch_active
                    kill_switch_active = False # Reset kill switch
                    # Start trading loop in a new thread
                    st.session_state.trading_thread = threading.Thread(
                        target=trading_loop,
                        args=(TRADING_SETTINGS['symbol'], TRADING_SETTINGS['timeframe'],
                              scalping_interval, news_interval, MT5_CONFIG['ACCOUNT_TYPE'],
                              st.session_state.risk_level, st.session_state.risk_per_trade_percent)
                    )
                    st.session_state.trading_thread.daemon = True # Allow thread to exit with main program
                    st.session_state.trading_thread.start()
                    st.session_state.alert_message = get_localized_text('trading_started_alert', st.session_state.language)
                    st.experimental_rerun()
                else:
                    st.session_state.alert_message = get_localized_text('connect_mt5_first', st.session_state.language)
        with col_stop:
            if st.button(get_localized_text('stop_trading', st.session_state.language), disabled=not st.session_state.is_trading_active, type="secondary"):
                st.session_state.is_trading_active = False
                st.session_state.alert_message = get_localized_text('stopping_trading_alert', st.session_state.language)
                st.experimental_rerun()

        # Emergency Kill Switch
        st.sidebar.markdown("---")
        if st.sidebar.button(get_localized_text('emergency_kill_switch', st.session_state.language), type="secondary"):
            global kill_switch_active
            kill_switch_active = True
            st.session_state.is_trading_active = False
            st.session_state.alert_message = get_localized_text('kill_switch_activated', st.session_state.language)
            st.experimental_rerun()

    else:
        st.sidebar.info(get_localized_text('connect_mt5_to_trade', st.session_state.language))


    # Main Dashboard Area
    st.markdown(f"### {get_localized_text('system_status', st.session_state.language)}: <span class='status-indicator {'green' if st.session_state.is_trading_active else 'red'}'></span> {st.session_state.mt5_connection_status} | {st.session_state.current_action}", unsafe_allow_html=True)

    # Real-time data update placeholder
    live_data_placeholder = st.empty()

    # Function to update live data
    def update_live_data():
        if not data_queue.empty():
            data = data_queue.get()
            st.session_state.account_info = data["account_info"]
            st.session_state.open_trades = data["open_trades"]
            st.session_state.scalping_signal = data["scalping_signal"]
            st.session_state.news_sentiment = data["news_sentiment"]
            st.session_state.quantum_prediction = data["quantum_prediction"]
            st.session_state.current_action = data["current_action"]
            st.session_state.alert_message = data["alert_message"]
            live_data_placeholder.empty() # Clear previous content
            with live_data_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card"><h3>{get_localized_text("balance", st.session_state.language)}</h3><div class="value">{st.session_state.account_info.balance if st.session_state.account_info else "N/A":.2f}</div></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><h3>{get_localized_text("equity", st.session_state.language)}</h3><div class="value">{st.session_state.account_info.equity if st.session_state.account_info else "N/A":.2f}</div></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-card"><h3>{get_localized_text("profit", st.session_state.language)}</h3><div class="value {'green' if (st.session_state.account_info and st.session_state.account_info.profit >= 0) else 'red'}">{st.session_state.account_info.profit if st.session_state.account_info else "N/A":.2f}</div></div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(f"### {get_localized_text('ai_insights', st.session_state.language)}")
                col_ai1, col_ai2, col_ai3 = st.columns(3)
                with col_ai1:
                    st.markdown(f'<div class="metric-card"><h3>{get_localized_text("scalping_signal", st.session_state.language)}</h3><div class="value">{st.session_state.scalping_signal}</div></div>', unsafe_allow_html=True)
                with col_ai2:
                    st.markdown(f'<div class="metric-card"><h3>{get_localized_text("news_sentiment", st.session_state.language)}</h3><div class="value">{st.session_state.news_sentiment}</div></div>', unsafe_allow_html=True)
                with col_ai3:
                    st.markdown(f'<div class="metric-card"><h3>{get_localized_text("quantum_prediction", st.session_state.language)}</h3><div class="value">{st.session_state.quantum_prediction}</div></div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(f"### {get_localized_text('open_trades', st.session_state.language)}")
                if st.session_state.open_trades:
                    trades_df = pd.DataFrame(st.session_state.open_trades)
                    trades_df['time'] = pd.to_datetime(trades_df['time'], unit='s')
                    trades_df = trades_df[['ticket', 'time', 'type', 'symbol', 'volume', 'price_open', 'price_current', 'profit', 'comment']]
                    st.dataframe(trades_df, use_container_width=True)

                    # Close trade functionality (manual mode or emergency)
                    st.markdown(f"#### {get_localized_text('close_trade', st.session_state.language)}")
                    trade_to_close = st.selectbox(get_localized_text('select_trade_to_close', st.session_state.language),
                                                  options=[t.ticket for t in st.session_state.open_trades])
                    if st.button(get_localized_text('close_selected_trade', st.session_state.language)):
                        if trade_to_close:
                            try:
                                result = close_order(trade_to_close)
                                if result:
                                    st.success(f"{get_localized_text('trade_closed_success', st.session_state.language)}: {trade_to_close}")
                                    send_telegram_message(f"DRAGON BRAIN QN-1: Trade {trade_to_close} closed manually.")
                                    # Update trade status in DB
                                    trade_info = {
                                        "timestamp": pd.Timestamp.now(),
                                        "ticket": trade_to_close,
                                        "status": "CLOSED",
                                        "close_price": result.price
                                    }
                                    save_trade_to_db(trade_info, update=True)
                                else:
                                    st.error(f"{get_localized_text('trade_close_failed', st.session_state.language)}: {trade_to_close}")
                            except Exception as e:
                                st.error(f"{get_localized_text('error_closing_trade', st.session_state.language)}: {e}")
                else:
                    st.info(get_localized_text('no_open_trades', st.session_state.language))
        else:
            # Display initial placeholders if no data yet
            with live_data_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card"><h3>{get_localized_text("balance", st.session_state.language)}</h3><div class="value">N/A</div></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><h3>{get_localized_text("equity", st.session_state.language)}</h3><div class="value">N/A</div></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-card"><h3>{get_localized_text("profit", st.session_state.language)}</h3><div class="value">N/A</div></div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(f"### {get_localized_text('ai_insights', st.session_state.language)}")
                col_ai1, col_ai2, col_ai3 = st.columns(3)
                with col_ai1:
                    st.markdown(f'<div class="metric-card"><h3>{get_localized_text("scalping_signal", st.session_state.language)}</h3><div class="value">N/A</div></div>', unsafe_allow_html=True)
                with col_ai2:
                    st.markdown(f'<div class="metric-card"><h3>{get_localized_text("news_sentiment", st.session_state.language)}</h3><div class="value">N/A</div></div>', unsafe_allow_html=True)
                with col_ai3:
                    st.markdown(f'<div class="metric-card"><h3>{get_localized_text("quantum_prediction", st.session_state.language)}</h3><div class="value">N/A</div></div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(f"### {get_localized_text('open_trades', st.session_state.language)}")
                st.info(get_localized_text('no_open_trades_yet', st.session_state.language))


    # Tabs for Trade Log and Real-time Visualization (Conceptual)
    tab1, tab2 = st.tabs([get_localized_text('trade_log', st.session_state.language), get_localized_text('realtime_visualization', st.session_state.language)])

    with tab1:
        st.header(get_localized_text('trade_history', st.session_state.language))
        trade_history = get_trade_history_from_db(DB_PATH)
        if trade_history:
            df_history = pd.DataFrame(trade_history)
            st.dataframe(df_history, use_container_width=True)
        else:
            st.info(get_localized_text('no_trade_history', st.session_state.language))

    with tab2:
        st.header(get_localized_text('realtime_charts', st.session_state.language))
        st.info(get_localized_text('chart_placeholder', st.session_state.language))
        # This section would typically embed interactive charting libraries like Plotly or Matplotlib
        # to visualize live price data and indicators.
        # Example: st.line_chart(df_with_indicators['close'])

    # Auto-refresh the dashboard every second to check for updates from the trading thread
    if st.session_state.is_trading_active:
        time.sleep(1)
        update_live_data()
        st.experimental_rerun() # Rerun to update the UI


if __name__ == "__main__":
    main_dashboard()

```

```python
# dragon_brain_qn1/config.py
# MT5 Connection Settings
MT5_CONFIG = {
    "MT5_LOGIN": "806850",  # Replace with your MT5 login
    "MT5_PASSWORD": "K&5Q@jOK",  # Replace with your MT5 password
    "MT5_SERVER": "PUPrime-Demo",  # e.g., "MetaQuotes-Demo" or your broker's server
    "MT5_PATH": "C:/Program Files/MetaTrader 5/terminal64.exe", # Path to your MT5 terminal
    "ACCOUNT_TYPE": "DEMO" # "DEMO" or "REAL"
}

# Telegram Bot Settings
# Go to BotFather on Telegram to create a bot and get your token.
# Then send a message to your bot and go to api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
# to find your chat_id.
TELEGRAM_CONFIG = {
    "BOT_TOKEN": "7928049984:AAGLMnMJbhSwydHt7Q_Twtvn1nS8E6GPNM4",
    "CHAT_ID": "6044605755" # Your personal chat ID or group chat ID
}

# Trading Settings
TRADING_SETTINGS = {
    "symbol": "EURUSD",  # Trading symbol
    "timeframe": "M1",   # Timeframe for scalping (e.g., M1, M5, M10, M30)
    "risk_per_trade_percent": 0.01, # 1% risk per trade (used for position sizing)
    "max_open_trades": 1, # Maximum number of open trades at any time
    "default_sl_pips": 10, # Default Stop Loss in pips (e.g., 10 pips)
    "default_tp_pips": 20, # Default Take Profit in pips (e.g., 20 pips, 1:2 R:R)
    "volume_precision": 2 # Number of decimal places for lot size (e.g., 0.01)
}

# Language Settings
LANGUAGE_SETTINGS = {
    'en': {
        'project_name': 'Dragon Brain QN-1',
        'slogan': 'AI-Powered Auto-Trading System for MT5',
        'select_language': 'Select Language',
        'mt5_connection': 'MT5 Connection',
        'status': 'Status',
        'not_connected': 'Not Connected',
        'connected_success': 'Connected Successfully',
        'connection_failed': 'Connection Failed',
        'connection_error': 'Connection Error',
        'mt5_login': 'MT5 Login',
        'mt5_password': 'MT5 Password',
        'mt5_server': 'MT5 Server',
        'mt5_path': 'MT5 Path',
        'account_type': 'Account Type',
        'connect_mt5': 'Connect to MT5',
        'connecting': 'Connecting...',
        'mt5_connected': 'MT5 Connected!',
        'disconnect_mt5': 'Disconnect MT5',
        'mt5_connected_alert': '🟢 Successfully connected to MT5!',
        'mt5_connection_failed_alert': '🔴 Failed to connect to MT5. Check credentials and path.',
        'mt5_connection_error_alert': '🔴 An error occurred during MT5 connection',
        'mt5_disconnected_alert': '🔴 Disconnected from MT5.',
        'trading_controls': 'Trading Controls',
        'strategy_mode': 'Strategy Mode',
        'risk_level': 'Risk Level',
        'trade_frequency': 'Trade Frequency',
        'start_trading': 'Start Trading',
        'stopping_trading_alert': '🟡 Stopping trading...',
        'stop_trading': 'Stop Trading',
        'emergency_kill_switch': 'Emergency Kill Switch',
        'kill_switch_activated': '🔴 EMERGENCY KILL SWITCH ACTIVATED! All trading stopped.',
        'connect_mt5_to_trade': 'Connect to MT5 to enable trading controls.',
        'trading_started_alert': '🟢 Trading session started!',
        'trading_stopped': 'Trading Stopped',
        'trading_session_ended': 'Trading session ended.',
        'system_status': 'System Status',
        'idle': 'Idle',
        'balance': 'Balance',
        'equity': 'Equity',
        'profit': 'Profit',
        'ai_insights': 'AI Insights',
        'scalping_signal': 'Scalping Signal',
        'news_sentiment': 'News Sentiment',
        'quantum_prediction': 'Quantum Prediction',
        'open_trades': 'Open Trades',
        'no_open_trades': 'No open trades.',
        'no_open_trades_yet': 'No open trades yet.',
        'close_trade': 'Close Trade',
        'select_trade_to_close': 'Select Trade to Close',
        'close_selected_trade': 'Close Selected Trade',
        'trade_closed_success': 'Trade closed successfully',
        'trade_close_failed': 'Failed to close trade',
        'error_closing_trade': 'Error closing trade',
        'trade_log': 'Trade Log',
        'realtime_visualization': 'Real-time Visualization',
        'trade_history': 'Trade History',
        'no_trade_history': 'No trade history available.',
        'realtime_charts': 'Real-time Charts & Indicators',
        'chart_placeholder': 'Live charts and indicator visualizations will appear here.',
        'running_trading_loop': 'Running trading loop...',
        'checking_scalping': 'Checking scalping signals...',
        'checking_news_sentiment': 'Checking news sentiment...',
        'simulating_quantum': 'Simulating quantum predictions...',
        'buy_signal': 'BUY Signal detected',
        'buy_failed': 'Failed to place BUY order',
        'sell_signal': 'SELL Signal detected',
        'sell_failed': 'Failed to place SELL order',
        'no_scalping_signal': 'No strong scalping signal.',
        'no_market_data': 'No market data fetched for scalping analysis.',
        'scalping_error': 'Scalping Engine Error',
        'news_buy_signal': 'News: BUY Signal detected',
        'news_sell_signal': 'News: SELL Signal detected',
        'no_news_signal': 'No strong news sentiment signal.',
        'no_news_fetched': 'No news articles fetched.',
        'news_error': 'News Sentiment Engine Error',
        'quantum_error': 'Pseudo-Quantum Prediction Error',
        'trade_already_open': 'Trade already open. Waiting for current trade to close.',
        'failed_get_account_info': 'Failed to get account information.',
        'failed_get_tick_data': 'Failed to get current tick data for symbol.',
        'invalid_volume': 'Calculated volume is invalid or too small. Check risk settings.',
        'trade_opened': 'Trade opened',
        'trade_failed': 'Trade failed',
        'max_trades_open': 'Maximum number of open trades reached. Waiting for current trades to close.',
    },
    'ar': {
        'project_name': 'دماغ التنين QN-1',
        'slogan': 'نظام تداول آلي مدعوم بالذكاء الاصطناعي لمنصة MT5',
        'select_language': 'اختر اللغة',
        'mt5_connection': 'اتصال MT5',
        'status': 'الحالة',
        'not_connected': 'غير متصل',
        'connected_success': 'تم الاتصال بنجاح',
        'connection_failed': 'فشل الاتصال',
        'connection_error': 'خطأ في الاتصال',
        'mt5_login': 'تسجيل دخول MT5',
        'mt5_password': 'كلمة مرور MT5',
        'mt5_server': 'خادم MT5',
        'mt5_path': 'مسار MT5',
        'account_type': 'نوع الحساب',
        'connect_mt5': 'الاتصال بـ MT5',
        'connecting': 'جاري الاتصال...',
        'mt5_connected': 'تم الاتصال بـ MT5!',
        'disconnect_mt5': 'قطع الاتصال بـ MT5',
        'mt5_connected_alert': '🟢 تم الاتصال بـ MT5 بنجاح!',
        'mt5_connection_failed_alert': '🔴 فشل الاتصال بـ MT5. تحقق من بيانات الاعتماد والمسار.',
        'mt5_connection_error_alert': '🔴 حدث خطأ أثناء الاتصال بـ MT5',
        'mt5_disconnected_alert': '🔴 تم قطع الاتصال بـ MT5.',
        'trading_controls': 'ضوابط التداول',
        'strategy_mode': 'وضع الإستراتيجية',
        'risk_level': 'مستوى المخاطرة',
        'trade_frequency': 'تكرار التداول',
        'start_trading': 'بدء التداول',
        'stopping_trading_alert': '🟡 جاري إيقاف التداول...',
        'stop_trading': 'إيقاف التداول',
        'emergency_kill_switch': 'مفتاح الإيقاف الطارئ',
        'kill_switch_activated': '🔴 تم تفعيل مفتاح الإيقاف الطارئ! تم إيقاف جميع عمليات التداول.',
        'connect_mt5_to_trade': 'اتصل بـ MT5 لتمكين ضوابط التداول.',
        'trading_started_alert': '🟢 بدأت جلسة التداول!',
        'trading_stopped': 'تم إيقاف التداول',
        'trading_session_ended': 'انتهت جلسة التداول.',
        'system_status': 'حالة النظام',
        'idle': 'خامل',
        'balance': 'الرصيد',
        'equity': 'حقوق الملكية',
        'profit': 'الربح',
        'ai_insights': 'رؤى الذكاء الاصطناعي',
        'scalping_signal': 'إشارة السكالبينج',
        'news_sentiment': 'تحليل الأخبار',
        'quantum_prediction': 'التنبؤ الكمي',
        'open_trades': 'الصفقات المفتوحة',
        'no_open_trades': 'لا توجد صفقات مفتوحة.',
        'no_open_trades_yet': 'لا توجد صفقات مفتوحة بعد.',
        'close_trade': 'إغلاق صفقة',
        'select_trade_to_close': 'اختر الصفقة للإغلاق',
        'close_selected_trade': 'إغلاق الصفقة المحددة',
        'trade_closed_success': 'تم إغلاق الصفقة بنجاح',
        'trade_close_failed': 'فشل إغلاق الصفقة',
        'error_closing_trade': 'خطأ في إغلاق الصفقة',
        'trade_log': 'سجل التداول',
        'realtime_visualization': 'التصور في الوقت الفعلي',
        'trade_history': 'سجل التداول',
        'no_trade_history': 'لا يوجد سجل تداول متاح.',
        'realtime_charts': 'الرسوم البيانية والمؤشرات في الوقت الفعلي',
        'chart_placeholder': 'ستظهر الرسوم البيانية الحية وتصورات المؤشرات هنا.',
        'running_trading_loop': 'جاري تشغيل حلقة التداول...',
        'checking_scalping': 'جاري التحقق من إشارات السكالبينج...',
        'checking_news_sentiment': 'جاري التحقق من تحليل الأخبار...',
        'simulating_quantum': 'جاري محاكاة التنبؤات الكمية...',
        'buy_signal': 'تم اكتشاف إشارة شراء',
        'buy_failed': 'فشل وضع أمر الشراء',
        'sell_signal': 'تم اكتشاف إشارة بيع',
        'sell_failed': 'فشل وضع أمر البيع',
        'no_scalping_signal': 'لا توجد إشارة سكالبينج قوية.',
        'no_market_data': 'لم يتم جلب بيانات السوق لتحليل السكالبينج.',
        'scalping_error': 'خطأ في محرك السكالبينج',
        'news_buy_signal': 'أخبار: تم اكتشاف إشارة شراء',
        'news_sell_signal': 'أخبار: تم اكتشاف إشارة بيع',
        'no_news_signal': 'لا توجد إشارة قوية لتحليل الأخبار.',
        'no_news_fetched': 'لم يتم جلب أي مقالات إخبارية.',
        'news_error': 'خطأ في محرك تحليل الأخبار',
        'quantum_error': 'خطأ في محرك التنبؤ شبه الكمي',
        'trade_already_open': 'الصفقة مفتوحة بالفعل. في انتظار إغلاق الصفقة الحالية.',
        'failed_get_account_info': 'فشل الحصول على معلومات الحساب.',
        'failed_get_tick_data': 'فشل الحصول على بيانات التجزئة الحالية للرمز.',
        'invalid_volume': 'الحجم المحسوب غير صالح أو صغير جدًا. تحقق من إعدادات المخاطر.',
        'trade_opened': 'تم فتح الصفقة',
        'trade_failed': 'فشلت الصفقة',
        'max_trades_open': 'تم الوصول إلى الحد الأقصى لعدد الصفقات المفتوحة. في انتظار إغلاق الصفقات الحالية.',
    }
}

```

```python
# dragon_brain_qn1/scalping_engine.py
import pandas as pd
import ta # Technical Analysis library for Python
import logging
# import tensorflow as tf # For deep learning models (conceptual)
# from tensorflow.keras.models import load_model # For loading pre-trained models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Deep Learning Model Placeholder ---
# In a real scenario, you would load your pre-trained deep learning model here.
# Example (uncomment and replace 'path/to/your/model.h5' with actual path):
# try:
#     technical_dl_model = load_model('path/to/your/technical_pattern_model.h5')
#     logging.info("Technical Deep Learning model loaded successfully.")
# except Exception as e:
#     logging.warning(f"Could not load technical deep learning model: {e}. Falling back to rule-based.")
#     technical_dl_model = None

technical_dl_model = None # Placeholder: set to None for now, as no model is provided


def calculate_indicators(df):
    """
    Calculates various technical indicators and adds them to the DataFrame.
    Requires 'open', 'high', 'low', 'close', 'real_volume' columns.
    """
    if df.empty:
        logging.warning("DataFrame is empty, cannot calculate indicators.")
        return df

    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'real_volume']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"Missing required columns for indicator calculation. Needs: {required_cols}")
        return df

    try:
        # RSI (Relative Strength Index)
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff() # Histogram

        # EMA (Exponential Moving Average)
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()

        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()

        # Volume filter (simple moving average of volume)
        df['volume_ma'] = df['real_volume'].rolling(window=20).mean()

        # ATR (Average True Range) - useful for SL/TP calculation
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()


        logging.info("Technical indicators calculated successfully.")
        return df
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        return df


def generate_scalping_signal_dl(df):
    """
    Generates a scalping signal (BUY, SELL, or HOLD) using a Deep Learning model
    if available, otherwise falls back to the rule-based logic.
    """
    if df.empty or len(df) < 50: # Need enough data for indicators/DL input
        logging.warning("Insufficient data for scalping signal generation.")
        return "HOLD"

    if technical_dl_model:
        try:
            # --- Deep Learning Prediction Logic ---
            # This is a conceptual placeholder.
            # In a real scenario, you would preprocess the DataFrame (e.g., normalize,
            # create sequences of features) into the format expected by your DL model.
            # Example:
            # features = df[['open', 'high', 'low', 'close', 'rsi', 'macd', 'ema_9', 'bb_upper', 'adx', 'volume_ma']].values
            # # Assuming your model expects a time-series input (batch_size, sequence_length, num_features)
            # sequence_length = 30 # Example, based on your model's training
            # if len(features) < sequence_length:
            #     return "HOLD"
            # input_sequence = features[-sequence_length:].reshape(1, sequence_length, -1)
            # prediction = technical_dl_model.predict(input_sequence)

            # # Assuming your model outputs a probability for BUY/SELL/HOLD
            # # You would convert this probability into a signal
            # if prediction[0][0] > 0.7: # Example threshold for BUY
            #     return "BUY"
            # elif prediction[0][1] > 0.7: # Example threshold for SELL
            #     return "SELL"
            # else:
            #     return "HOLD"

            logging.info("Using conceptual Deep Learning model for scalping signal.")
            # For demonstration, we'll still use the rule-based logic if DL model is not truly implemented
            return _generate_scalping_signal_rule_based(df)

        except Exception as e:
            logging.error(f"Error during Deep Learning model prediction for scalping: {e}. Falling back to rule-based.")
            return _generate_scalping_signal_rule_based(df)
    else:
        logging.info("No Deep Learning model loaded for scalping. Using rule-based logic.")
        return _generate_scalping_signal_rule_based(df)


def _generate_scalping_signal_rule_based(df):
    """
    Generates a scalping signal (BUY, SELL, or HOLD) based on a hybrid logic
    of multiple indicator confirmations. This is the fallback/initial logic.
    """
    if df.empty or len(df) < 2: # Need at least current and previous for crossovers
        logging.warning("Insufficient data for rule-based scalping signal generation.")
        return "HOLD"

    # Get the latest candle's indicator values
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    buy_confirmations = 0
    sell_confirmations = 0

    # 1. RSI
    if latest['rsi'] < 30: # Oversold
        buy_confirmations += 1
    elif latest['rsi'] > 70: # Overbought
        sell_confirmations += 1

    # 2. MACD Crossover
    if latest['macd'] > latest['macd_signal'] and previous['macd'] <= previous['macd_signal']:
        buy_confirmations += 1
    elif latest['macd'] < latest['macd_signal'] and previous['macd'] >= previous['macd_signal']:
        sell_confirmations += 1

    # 3. EMA (9/21/50) - Order and Crossover
    # 9 EMA crosses above 21 EMA AND 21 EMA is above 50 EMA (strong bullish)
    if latest['ema_9'] > latest['ema_21'] and previous['ema_9'] <= previous['ema_21'] and latest['ema_21'] > latest['ema_50']:
        buy_confirmations += 1
    # 9 EMA crosses below 21 EMA AND 21 EMA is below 50 EMA (strong bearish)
    elif latest['ema_9'] < latest['ema_21'] and previous['ema_9'] >= previous['ema_21'] and latest['ema_21'] < latest['ema_50']:
        sell_confirmations += 1

    # 4. Bollinger Bands
    # Price closes below lower band (oversold, potential reversal up)
    if latest['close'] < latest['bb_lower'] and previous['close'] >= previous['bb_lower']:
        buy_confirmations += 1
    # Price closes above upper band (overbought, potential reversal down)
    elif latest['close'] > latest['bb_upper'] and previous['close'] <= previous['bb_upper']:
        sell_confirmations += 1

    # 5. ADX (>25 for strong trend, combined with +DI/-DI)
    if latest['adx'] > 25:
        if latest['adx_pos'] > latest['adx_neg'] and latest['adx_pos'] > previous['adx_pos']: # +DI increasing
            buy_confirmations += 1
        elif latest['adx_neg'] > latest['adx_pos'] and latest['adx_neg'] > previous['adx_neg']: # -DI increasing
            sell_confirmations += 1

    # 6. Volume filter
    # Check if current volume is above its moving average (confirming strength)
    if latest['real_volume'] > latest['volume_ma']:
        if buy_confirmations > sell_confirmations: # Add to the stronger signal
            buy_confirmations += 1
        elif sell_confirmations > buy_confirmations:
            sell_confirmations += 1

    logging.info(f"Scalping confirmations (rule-based): BUY={buy_confirmations}, SELL={sell_confirmations}")

    # Decision based on 3+ indicator confirmations
    if buy_confirmations >= 3:
        return "BUY"
    elif sell_confirmations >= 3:
        return "SELL"
    else:
        return "HOLD"

```

```python
# dragon_brain_qn1/news_sentiment.py
import random
import logging
import time
# import tensorflow as tf # For deep learning models (conceptual)
# from transformers import pipeline # For NLP sentiment analysis (conceptual)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Deep Learning NLP Model Placeholder ---
# In a real scenario, you would load your pre-trained NLP model here.
# Example (uncomment and replace 'sentiment-model-name' with actual model):
# try:
#     sentiment_pipeline = pipeline("sentiment-analysis", model="sentiment-model-name")
#     logging.info("NLP Sentiment Analysis model loaded successfully.")
# except Exception as e:
#     logging.warning(f"Could not load NLP sentiment model: {e}. Falling back to keyword-based.")
#     sentiment_pipeline = None

sentiment_pipeline = None # Placeholder: set to None for now, as no model is provided


def fetch_news():
    """
    Mocks fetching live news articles.
    In a real scenario, this would connect to news APIs (e.g., Google News API, Reuters, Twitter API).
    Returns a list of dictionaries, each with 'title' and 'text'.
    """
    logging.info("Mocking news fetching...")
    # Simulate API call delay
    time.sleep(2)

    mock_news_headlines = [
        {"title": "Central Bank announces interest rate cut", "text": "The Central Bank has decided to cut interest rates by 25 basis points, aiming to stimulate economic growth. This move is expected to boost consumer spending and investment."},
        {"title": "Inflation data higher than expected", "text": "Latest inflation figures show a significant increase, surpassing market expectations. This could lead to tighter monetary policies in the near future."},
        {"title": "Major tech company reports record earnings", "text": "A leading technology firm announced unprecedented quarterly earnings, driven by strong demand for its new products and services. Stock prices are expected to surge."},
        {"title": "Geopolitical tensions rise in key region", "text": "Escalating geopolitical tensions in a critical oil-producing region have raised concerns about global supply chains and commodity prices."},
        {"title": "New trade agreement signed by major economies", "text": "Several prominent economies have finalized a new trade agreement, aiming to reduce tariffs and foster greater international commerce."},
        {"title": "Company X faces legal challenges over product safety", "text": "Shares of Company X plummeted today after news broke of a major lawsuit regarding product safety concerns. Regulatory bodies are investigating."},
        {"title": "Market remains calm ahead of economic data release", "text": "Investors are holding steady, awaiting the release of crucial economic data later this week, which is expected to provide clarity on the market's direction."},
    ]

    # Return a random subset or all mock headlines
    return random.sample(mock_news_headlines, k=random.randint(1, len(mock_news_headlines)))


def analyze_sentiment_dl(news_text):
    """
    Analyzes sentiment of news text using a Deep Learning NLP model if available,
    otherwise falls back to keyword-based analysis.
    Returns 'BUY', 'SELL', or 'NEUTRAL'.
    """
    logging.info(f"Analyzing sentiment for news: '{news_text[:50]}...'")

    if sentiment_pipeline:
        try:
            # --- Deep Learning NLP Prediction Logic ---
            # This is a conceptual placeholder.
            # The actual output format depends on your NLP model.
            # Example for a Hugging Face pipeline:
            # result = sentiment_pipeline(news_text)[0]
            # label = result['label']
            # score = result['score']

            # if label == 'POSITIVE' and score > 0.8: # Example threshold
            #     return "BUY"
            # elif label == 'NEGATIVE' and score > 0.8:
            #     return "SELL"
            # else:
            #     return "NEUTRAL"

            logging.info("Using conceptual Deep Learning NLP model for sentiment analysis.")
            # For demonstration, we'll still use the rule-based logic if DL model is not truly implemented
            return _analyze_sentiment_keyword_based(news_text)

        except Exception as e:
            logging.error(f"Error during Deep Learning NLP sentiment analysis: {e}. Falling back to keyword-based.")
            return _analyze_sentiment_keyword_based(news_text)
    else:
        logging.info("No Deep Learning NLP model loaded. Using keyword-based sentiment analysis.")
        return _analyze_sentiment_keyword_based(news_text)


def _analyze_sentiment_keyword_based(news_text):
    """
    Mocks the NLP model for sentiment analysis using a simple keyword-based approach.
    Returns 'BUY', 'SELL', or 'NEUTRAL'.
    """
    news_text_lower = news_text.lower()

    positive_keywords = ['cut interest rates', 'stimulate economic growth', 'boost', 'record earnings', 'surge', 'trade agreement', 'reduce tariffs', 'foster greater international commerce']
    negative_keywords = ['inflation', 'tighter monetary policies', 'geopolitical tensions', 'legal challenges', 'plummeted', 'lawsuit', 'concerns']

    is_positive = any(keyword in news_text_lower for keyword in positive_keywords)
    is_negative = any(keyword in news_text_lower for keyword in negative_keywords)

    if is_positive and not is_negative:
        logging.info("Sentiment: BUY (keyword-based)")
        return "BUY"
    elif is_negative and not is_positive:
        logging.info("Sentiment: SELL (keyword-based)")
        return "SELL"
    else:
        logging.info("Sentiment: NEUTRAL (keyword-based)")
        return "NEUTRAL"

```

```python
# dragon_brain_qn1/utils.py
import MetaTrader5 as mt5
from config import LANGUAGE_SETTINGS, TRADING_SETTINGS
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_localized_text(key, lang):
    """
    Retrieves localized text based on the key and selected language.
    Falls back to English if the key is not found in the selected language.
    """
    if lang not in LANGUAGE_SETTINGS:
        lang = 'en' # Default to English if language not found

    if key in LANGUAGE_SETTINGS[lang]:
        return LANGUAGE_SETTINGS[lang][key]
    else:
        # Fallback to English if key not found in selected language
        return LANGUAGE_SETTINGS['en'].get(key, f"MISSING_TEXT[{key}]")


def calculate_position_size(account_balance, risk_percent, stop_loss_price, entry_price, symbol, trade_type):
    """
    Calculates the appropriate position size (volume/lots) based on account balance,
    risk percentage per trade, and stop loss distance.

    Args:
        account_balance (float): Current account balance or equity.
        risk_percent (float): Percentage of account balance to risk per trade (e.g., 0.01 for 1%).
        stop_loss_price (float): The calculated stop loss price.
        entry_price (float): The price at which the trade is entered.
        symbol (str): The trading symbol (e.g., "EURUSD").
        trade_type (str): "BUY" or "SELL".

    Returns:
        float: The calculated lot size (volume) for the trade. Returns 0 if calculation fails.
    """
    if not mt5.is_connected():
        logging.error("MT5 not connected. Cannot calculate position size.")
        return 0.0

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}.")
        return 0.0

    # Get tick size and point value
    # For Forex, point is usually 0.00001 (5 decimal places) or 0.001 (3 decimal places for JPY pairs)
    # For indices/commodities, it varies.
    # We use symbol_info.point which is the minimum price change.
    # We need to consider the contract size for the symbol.
    contract_size = symbol_info.trade_contract_size # e.g., 100000 for standard forex lot

    # Calculate risk amount in currency
    risk_amount = account_balance * risk_percent

    # Calculate stop loss distance in points (absolute difference)
    if trade_type == "BUY":
        sl_distance_points = abs(entry_price - stop_loss_price) / symbol_info.point
    elif trade_type == "SELL":
        sl_distance_points = abs(stop_loss_price - entry_price) / symbol_info.point
    else:
        logging.error("Invalid trade_type for position size calculation. Must be 'BUY' or 'SELL'.")
        return 0.0

    if sl_distance_points == 0:
        logging.warning("Stop loss distance is zero, cannot calculate position size. Returning 0 volume.")
        return 0.0

    # Calculate value per pip/point for 1 standard lot (contract_size * point)
    # For Forex, this is typically $10 per standard lot for a 5-digit broker (0.0001 point)
    # If the quote currency is not USD, conversion is needed. For simplicity, assuming USD account.
    # This is a simplified calculation. A truly accurate one would involve conversion rates if base currency != account currency.
    value_per_point_per_lot = symbol_info.trade_tick_value / symbol_info.trade_tick_size # Value of one point move per lot

    # Calculate lot size
    # Risk per lot = SL_distance_points * Value_per_point_per_lot
    # Lots = Risk_Amount / Risk_per_lot
    # Or, simpler: Lots = Risk_Amount / (SL_distance_points * value_per_point_per_lot)
    # For MT5, volume is in lots.
    try:
        # Calculate the monetary risk per unit of volume (e.g., per 0.01 lot)
        # This is a crucial part and depends on the symbol's tick value and tick size.
        # trade_tick_value is the value of a tick in deposit currency.
        # trade_tick_size is the minimum price change.
        # So, (sl_distance_points * symbol_info.trade_tick_value) gives risk in deposit currency for 1 standard lot.
        # To get risk per unit of volume, we need to divide by contract size or adjust based on minimum volume.

        # A more direct way:
        # Calculate the value of one pip/point for the given symbol.
        # For EURUSD, if point is 0.00001, pip is 0.0001. A 10 pip move is 0.00100.
        # The value of 1 pip for a 1 standard lot (100,000 units) of EURUSD is $10.
        # So, risk per pip = risk_amount / (sl_pips * pip_value_per_lot)
        # This is simplified. The correct way is to use the actual stop loss distance in price.

        # Risk per unit of volume (e.g., 1 lot)
        # For a standard lot (100,000 units), if price moves 1 point, profit/loss is trade_tick_value.
        # So, total risk per lot = sl_distance_points * symbol_info.trade_tick_value
        risk_per_lot = sl_distance_points * symbol_info.trade_tick_value

        if risk_per_lot <= 0:
            logging.warning(f"Calculated risk per lot is zero or negative ({risk_per_lot}). Cannot determine volume.")
            return 0.0

        calculated_volume = risk_amount / risk_per_lot

        # Adjust for minimum and maximum volume allowed by broker
        min_volume = symbol_info.volume_min
        max_volume = symbol_info.volume_max
        step_volume = symbol_info.volume_step

        # Ensure volume is a multiple of step_volume and within min/max
        calculated_volume = max(min_volume, calculated_volume)
        calculated_volume = min(max_volume, calculated_volume)
        calculated_volume = round(calculated_volume / step_volume) * step_volume

        logging.info(f"Calculated volume: {calculated_volume:.{TRADING_SETTINGS['volume_precision']}f} lots for {symbol} with risk {risk_percent*100}%")
        return calculated_volume
    except Exception as e:
        logging.error(f"Error calculating position size: {e}")
        return 0.0


def calculate_sl_tp(symbol, entry_price, trade_type, sl_pips, tp_pips):
    """
    Calculates Stop Loss (SL) and Take Profit (TP) prices based on entry price
    and desired pip distances.

    Args:
        symbol (str): The trading symbol.
        entry_price (float): The entry price of the trade.
        trade_type (str): "BUY" or "SELL".
        sl_pips (int): Desired Stop Loss distance in pips.
        tp_pips (int): Desired Take Profit distance in pips.

    Returns:
        tuple: (stop_loss_price, take_profit_price)
    """
    if not mt5.is_connected():
        logging.error("MT5 not connected. Cannot calculate SL/TP.")
        return 0.0, 0.0

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}.")
        return 0.0, 0.0

    # Calculate pip value in terms of price (e.g., 0.0001 for EURUSD)
    # A pip is usually 10 points for 5-digit brokers (point is 0.00001)
    # Or 1 point for 3-digit brokers (point is 0.001)
    # So, pip_value_in_price = symbol_info.point * 10 (for 5-digit)
    # Or, more generally, it's the value of a single pip.
    # For most forex pairs, a pip is 0.0001. For JPY pairs, 0.01.
    # Let's assume 5-digit for simplicity, where 1 pip = 10 * symbol_info.point
    # A more robust way is to define pip size based on the symbol's digits.
    # For now, let's use a fixed pip size for common forex pairs.
    # For example, if symbol_info.digits == 5, pip_size = 0.0001. If digits == 3, pip_size = 0.01.
    pip_size = 0.0001 # Default for 5-digit forex. Adjust if handling JPY pairs or other assets.
    if symbol_info.digits == 3: # For JPY pairs like USDJPY
        pip_size = 0.01
    elif symbol_info.digits == 2: # For some indices
        pip_size = 0.1

    sl_distance = sl_pips * pip_size
    tp_distance = tp_pips * pip_size

    stop_loss_price = 0.0
    take_profit_price = 0.0

    if trade_type == "BUY":
        stop_loss_price = entry_price - sl_distance
        take_profit_price = entry_price + tp_distance
    elif trade_type == "SELL":
        stop_loss_price = entry_price + sl_distance
        take_profit_price = entry_price - tp_distance
    else:
        logging.error("Invalid trade_type for SL/TP calculation. Must be 'BUY' or 'SELL'.")
        return 0.0, 0.0

    # Round SL/TP to appropriate digits for the symbol
    stop_loss_price = round(stop_loss_price, symbol_info.digits)
    take_profit_price = round(take_profit_price, symbol_info.digits)

    logging.info(f"Calculated SL: {stop_loss_price}, TP: {take_profit_price} for {trade_type} at {entry_price}")
    return stop_loss_price, take_profit_price

