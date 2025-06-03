import logging
from logging.handlers import RotatingFileHandler
import os
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import okx.Trade as Trade
import okx.MarketData as MarketData
import okx.Account as Account
import okx.PublicData as PublicData
import traceback

# 设置日志
logging.basicConfig(
    handlers=[RotatingFileHandler('trading_bot.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# API 密钥存储文件
CONFIG_FILE = 'okx_config.json'

# 全局变量
trade_client = None
market_client = None
account_client = None
public_client = None
current_price = 0.0
latest_rsi = None
latest_macd = None
latest_signal = None
latest_histogram = None
running = True
last_rsi_buy_time = None
last_rsi_sell_time = None
last_macd_buy_time = None
last_macd_sell_time = None

# 固定交易参数
TIMEFRAME = '1H'  # 1小时周期
RSI_BUY_VALUE = 25.0  # RSI 低于25开多
RSI_SELL_VALUE = 75.0  # RSI 高于75开空
BUY_RATIO = 0.05  # 每次开仓使用5% USDT
LEVERAGE = 10.0  # 10倍杠杆
MARGIN_MODE = 'cross'  # 全仓模式
TAKE_PROFIT = 12.0  # 12%止盈
STOP_LOSS = 8.0  # 8%止损
COOLDOWN = timedelta(hours=1)  # 1小时冷却时间

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        except Exception as e:
            logging.error(f"加载配置文件失败: {str(e)}")
    return {}

def save_config(api_key, api_secret, passphrase):
    config = {'api_key': api_key, 'api_secret': api_secret, 'passphrase': passphrase}
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        logging.info("配置文件已保存")
    except Exception as e:
        logging.error(f"保存配置文件失败: {str(e)}")

def init_okx(api_key, api_secret, passphrase, flag='0'):
    global trade_client, market_client, account_client, public_client
    try:
        trade_client = Trade.TradeAPI(api_key, api_secret, passphrase, use_server_time=False, flag=flag)
        market_client = MarketData.MarketAPI(flag=flag)
        account_client = Account.AccountAPI(api_key, api_secret, passphrase, use_server_time=False, flag=flag)
        public_client = PublicData.PublicAPI(flag=flag)
        response = account_client.get_account_balance()
        if response.get('code') != '0':
            raise Exception(f"账户余额检查失败: {response.get('msg', '未知错误')}")
        logging.info(f"OKX API 初始化成功，flag={flag}")
        return True
    except Exception as e:
        logging.error(f"OKX API 初始化失败: {str(e)}\n{traceback.format_exc()}")
        return False

def set_leverage(symbol):
    try:
        params = {'instId': symbol, 'lever': str(LEVERAGE), 'mgnMode': MARGIN_MODE}
        response = account_client.set_leverage(**params)
        if response.get('code') == '0':
            logging.info(f"设置杠杆成功: {symbol}, 杠杆={LEVERAGE}, 模式={MARGIN_MODE}")
            return True
        else:
            raise Exception(f"设置杠杆失败: {response.get('msg', '未知错误')}")
    except Exception as e:
        logging.error(f"设置杠杆失败: {str(e)}")
        return False

def get_btc_price():
    global current_price
    try:
        ticker = market_client.get_ticker(instId='BTC-USDT-SWAP')
        if ticker.get('code') != '0':
            raise Exception(f"获取行情失败: {ticker.get('msg', '未知错误')}")
        current_price = float(ticker['data'][0]['last'])
        return current_price
    except Exception as e:
        logging.error(f"获取价格失败: {str(e)}")
        return None

def get_klines(symbol, interval, limit=100):
    try:
        klines = market_client.get_candlesticks(instId=symbol, bar=interval, limit=str(limit))
        if not klines.get('data'):
            logging.warning(f"无K线数据: {symbol}, {interval}")
            return None
        df = pd.DataFrame(klines['data'], columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
        df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        df = df.sort_values('ts').reset_index(drop=True)
        if df.empty:
            logging.warning(f"无有效K线数据: {symbol}, {interval}")
            return None
        logging.info(f"获取K线数据: {len(df)} 条, 最新时间: {df['ts'].iloc[-1]}")
        return df
    except Exception as e:
        logging.error(f"获取K线错误: {str(e)}\n{traceback.format_exc()}")
        return None

def calculate_rsi(df, period=14):
    try:
        if len(df) < period + 1:
            logging.warning(f"RSI 数据不足: {len(df)} 条")
            return None
        close = df['close'].values
        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        if avg_loss == 0:
            rs = np.inf
        else:
            rs = avg_gain / avg_loss
        rsi = np.zeros(len(close))
        rsi[:period] = np.nan
        rsi[period] = 100 - (100 / (1 + rs))
        for i in range(period + 1, len(close)):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
            if avg_loss == 0:
                rs = np.inf
            else:
                rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        if np.isnan(rsi[-1]):
            logging.warning("RSI 计算结果无效")
            return None
        logging.info(f"RSI 计算: 周期={period}, 最新RSI={rsi[-1]:.2f}")
        return pd.Series(rsi, index=df.index)
    except Exception as e:
        logging.error(f"RSI 计算错误: {str(e)}\n{traceback.format_exc()}")
        return None

def calculate_macd(df, fast=12, slow=26, signal=9):
    try:
        if len(df) < slow + signal - 1:
            logging.warning(f"MACD 数据不足: {len(df)} 条")
            return None, None, None
        close = df['close'].values
        ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
        ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
        macd = ema_fast - ema_slow
        signal_line = pd.Series(macd).ewm(span=signal, adjust=False).mean().values
        histogram = macd - signal_line
        logging.info(f"MACD 计算: MACD={macd[-1]:.2f}, 信号线={signal_line[-1]:.2f}, 柱状图={histogram[-1]:.2f}")
        return pd.Series(macd, index=df.index), pd.Series(signal_line, index=df.index), pd.Series(histogram, index=df.index)
    except Exception as e:
        logging.error(f"MACD 计算错误: {str(e)}\n{traceback.format_exc()}")
        return None, None, None

def get_symbol_info(symbol='BTC-USDT-SWAP'):
    try:
        info = public_client.get_instruments(instType='SWAP', instId=symbol)
        if info.get('code') != '0':
            raise Exception(f"获取交易对信息失败: {info.get('msg', '未知错误')}")
        ct_val = float(info['data'][0]['ctVal'])
        min_qty = float(info['data'][0]['minSz'])
        return ct_val, min_qty
    except Exception as e:
        logging.error(f"获取交易对信息失败: {str(e)}")
        return 0.0001, 1

def place_order(side, pos_side, quantity):
    try:
        ct_val, min_qty = get_symbol_info('BTC-USDT-SWAP')
        quantity = max(round(quantity / ct_val), 1)
        if quantity < min_qty:
            logging.warning(f"下单失败: 数量 {quantity} 张小于最小值 {min_qty} 张")
            return None
        set_leverage('BTC-USDT-SWAP')
        params = {
            'instId': 'BTC-USDT-SWAP',
            'tdMode': MARGIN_MODE,
            'side': side.lower(),
            'posSide': pos_side.lower(),
            'ordType': 'market',
            'sz': str(quantity),
            'clOrdId': f"order_{int(time.time())}",
            'tpTriggerPx': str(round(current_price * (1 + TAKE_PROFIT / 100 if pos_side == 'long' else 1 - TAKE_PROFIT / 100), 2)),
            'tpOrdPx': '-1',
            'slTriggerPx': str(round(current_price * (1 - STOP_LOSS / 100 if pos_side == 'long' else 1 + STOP_LOSS / 100), 2)),
            'slOrdPx': '-1'
        }
        order = trade_client.place_order(**params)
        if order['code'] == '0':
            action = '开多' if side == 'buy' and pos_side == 'long' else '开空' if side == 'sell' and pos_side == 'short' else '平仓'
            logging.info(f"{action} 订单已下: 数量 {quantity} 张, 止盈 {TAKE_PROFIT}%, 止损 {STOP_LOSS}%")
            return order['data'][0]['ordId']
        else:
            logging.error(f"下单失败: {order['msg']}")
            return None
    except Exception as e:
        logging.error(f"下单失败: {str(e)}\n{traceback.format_exc()}")
        return None

def get_balance():
    try:
        balance = account_client.get_account_balance()
        if balance.get('code') != '0':
            raise Exception(f"获取余额失败: {balance.get('msg', '未知错误')}")
        usdt = float(next((asset for asset in balance['data'][0]['details'] if asset['ccy'] == 'USDT'), {'availEq': '0'})['availEq'])
        total_equity = float(balance['data'][0]['totalEq'])
        positions = account_client.get_positions(instType='SWAP', instId='BTC-USDT-SWAP')
        long_qty = 0
        short_qty = 0
        long_avg_price = 0.0
        short_avg_price = 0.0
        if positions.get('code') == '0' and positions.get('data'):
            for pos in positions['data']:
                if pos['instId'] == 'BTC-USDT-SWAP':
                    qty = float(pos['pos'])
                    avg_price = float(pos['avgPx']) if pos['avgPx'] else 0.0
                    if pos['posSide'] == 'long':
                        long_qty = qty
                        long_avg_price = avg_price
                    elif pos['posSide'] == 'short':
                        short_qty = qty
                        short_avg_price = avg_price
        return usdt, long_qty, short_qty, long_avg_price, short_avg_price, total_equity
    except Exception as e:
        logging.error(f"获取余额失败: {str(e)}")
        return None, None, None, None, None, None

def check_take_profit_stop_loss(long_qty, short_qty, long_avg_price, short_avg_price, current_price, rsi, macd, signal):
    try:
        pos_side = None
        qty = 0
        reason = None
        if long_qty > 0 and long_avg_price > 0:
            profit_percentage = (current_price - long_avg_price) / long_avg_price * 100
            if profit_percentage >= TAKE_PROFIT:
                pos_side = 'long'
                qty = long_qty
                reason = '止盈'
            elif profit_percentage <= -STOP_LOSS:
                pos_side = 'long'
                qty = long_qty
                reason = '止损'
            elif rsi is not None and rsi >= RSI_SELL_VALUE:
                pos_side = 'long'
                qty = long_qty
                reason = 'RSI 高于75'
            elif macd is not None and signal is not None and macd > 0 and macd < signal and macd[-2] >= signal[-2]:
                pos_side = 'long'
                qty = long_qty
                reason = 'MACD 死叉'
        if short_qty > 0 and short_avg_price > 0:
            profit_percentage = (short_avg_price - current_price) / short_avg_price * 100
            if profit_percentage >= TAKE_PROFIT:
                pos_side = 'short'
                qty = short_qty
                reason = '止盈'
            elif profit_percentage <= -STOP_LOSS:
                pos_side = 'short'
                qty = short_qty
                reason = '止损'
            elif rsi is not None and rsi <= RSI_BUY_VALUE:
                pos_side = 'short'
                qty = short_qty
                reason = 'RSI 低于25'
            elif macd is not None and signal is not None and macd < 0 and macd > signal and macd[-2] <= signal[-2]:
                pos_side = 'short'
                qty = short_qty
                reason = 'MACD 金叉'
        return pos_side, qty, reason
    except Exception as e:
        logging.error(f"检查止盈止损错误: {str(e)}\n{traceback.format_exc()}")
        return None, 0, None

def trading_cycle():
    global last_rsi_buy_time, last_rsi_sell_time, last_macd_buy_time, last_macd_sell_time, current_price, latest_rsi, latest_macd, latest_signal, latest_histogram
    try:
        logging.info("开始交易周期")
        df = get_klines('BTC-USDT-SWAP', TIMEFRAME)
        if df is None:
            logging.warning("无K线数据，跳过交易")
            return

        rsi = calculate_rsi(df)
        macd, signal, histogram = calculate_macd(df)
        if rsi is None or macd is None or signal is None or histogram is None:
            logging.warning("指标计算失败，跳过交易")
            return
        latest_rsi = rsi.iloc[-1]
        latest_macd = macd.iloc[-1]
        latest_signal = signal.iloc[-1]
        latest_histogram = histogram.iloc[-1]

        usdt_balance, long_qty, short_qty, long_avg_price, short_avg_price, total_equity = get_balance()
        if usdt_balance is None:
            logging.warning("获取余额失败，跳过交易")
            return

        price = get_btc_price()
        if price:
            current_price = price
            logging.info(f"当前 BTC 价格: ${price:.2f}")
        else:
            logging.warning("无法获取价格，跳过交易")
            return

        now = datetime.now()
        pos_side, qty, reason = check_take_profit_stop_loss(long_qty, short_qty, long_avg_price, short_avg_price, current_price, latest_rsi, macd, signal)
        if pos_side and qty > 0:
            order = place_order('sell' if pos_side == 'long' else 'buy', pos_side, qty)
            if order:
                logging.info(f"平仓: {reason}, 数量 {qty:.0f} 张")
                if pos_side == 'long':
                    last_rsi_buy_time = last_macd_buy_time = None
                else:
                    last_rsi_sell_time = last_macd_sell_time = None
            return

        max_quantity = (total_equity * BUY_RATIO) / current_price * LEVERAGE
        if latest_rsi <= RSI_BUY_VALUE and (last_rsi_buy_time is None or (now - last_rsi_buy_time) >= COOLDOWN):
            quantity = min((usdt_balance * BUY_RATIO) / current_price * LEVERAGE, max_quantity)
            if quantity > 0:
                order = place_order('buy', 'long', quantity)
                if order:
                    last_rsi_buy_time = now
                    logging.info(f"RSI 开多: 数量 {quantity:.0f} 张")
        elif latest_rsi >= RSI_SELL_VALUE and (last_rsi_sell_time is None or (now - last_rsi_sell_time) >= COOLDOWN):
            if long_qty > 0:
                order = place_order('sell', 'long', long_qty)
                if order:
                    last_rsi_sell_time = now
                    logging.info(f"RSI 平多: 数量 {long_qty:.0f} 张")
            elif short_qty == 0:
                quantity = min((usdt_balance * BUY_RATIO) / current_price * LEVERAGE, max_quantity)
                order = place_order('sell', 'short', quantity)
                if order:
                    last_rsi_sell_time = now
                    logging.info(f"RSI 开空: 数量 {quantity:.0f} 张")

        if (latest_macd < 0 and latest_macd > latest_signal and macd.iloc[-2] <= signal.iloc[-2] and latest_histogram > 0 and
                (last_macd_buy_time is None or (now - last_macd_buy_time) >= COOLDOWN)):
            quantity = min((usdt_balance * BUY_RATIO) / current_price * LEVERAGE, max_quantity)
            if quantity > 0:
                order = place_order('buy', 'long', quantity)
                if order:
                    last_macd_buy_time = now
                    logging.info(f"MACD 开多: 数量 {quantity:.0f} 张")
        elif (latest_macd > 0 and latest_macd < latest_signal and macd.iloc[-2] >= signal.iloc[-2] and latest_histogram < 0 and
              (last_macd_sell_time is None or (now - last_macd_sell_time) >= COOLDOWN)):
            if long_qty > 0:
                order = place_order('sell', 'long', long_qty)
                if order:
                    last_macd_sell_time = now
                    logging.info(f"MACD 平多: 数量 {long_qty:.0f} 张")
            elif short_qty == 0:
                quantity = min((usdt_balance * BUY_RATIO) / current_price * LEVERAGE, max_quantity)
                order = place_order('sell', 'short', quantity)
                if order:
                    last_macd_sell_time = now
                    logging.info(f"MACD 开空: 数量 {quantity:.0f} 张")
    except Exception as e:
        logging.error(f"交易周期错误: {str(e)}\n{traceback.format_exc()}")

def main():
    try:
        config = load_config()
        api_key = config.get('api_key') or os.getenv('OKX_API_KEY')
        api_secret = config.get('api_secret') or os.getenv('OKX_API_SECRET')
        passphrase = config.get('passphrase') or os.getenv('OKX_PASSPHRASE')
        if not all([api_key, api_secret, passphrase]):
            logging.error("API 密钥未配置")
            return

        for flag in ['0', '1']:
            if init_okx(api_key, api_secret, passphrase, flag):
                save_config(api_key, api_secret, passphrase)
                break
        else:
            logging.error("API 初始化失败")
            return

        set_leverage('BTC-USDT-SWAP')

        while running:
            trading_cycle()
            logging.info("交易周期完成，等待下一小时")
            time.sleep(3600)
    except Exception as e:
        logging.error(f"主程序错误: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()