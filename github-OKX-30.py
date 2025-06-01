import os
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import okx.Trade as Trade
import okx.MarketData as MarketData
import okx.Account as Account
import okx.PublicData as PublicData

# 设置日志
log_handler = RotatingFileHandler('trading.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.basicConfig(handlers=[log_handler], level=logging.INFO)

# 全局变量
trade_client = None
market_client = None
account_client = None
public_client = None
current_price = 0.0
rsi_enabled = True
macd_enabled = True
rsi_buy_ratio = 10.0
macd_buy_ratio = 10.0
rsi_timeframe = '30m'  # 修改为 30 分钟
macd_timeframe = '30m'  # 修改为 30 分钟
rsi_buy_value = 30.0  # 修改为 30
rsi_sell_value = 60.0  # 修改为 60
rsi_full_sell_value = 70.0  # 新增：RSI 全卖阈值
last_rsi_buy_time = None
last_rsi_sell_time = None
last_macd_buy_time = None
last_macd_sell_time = None
latest_rsi = None
latest_macd = None
latest_signal = None
latest_histogram = None
running = False
lock = threading.Lock()
request_timestamps = []
REQUEST_LIMIT = 10  # 放宽到 10，符合 OKX API 限制
REQUEST_WINDOW = 2.0

def rate_limit():
    with lock:
        current_time = time.time()
        request_timestamps[:] = [t for t in request_timestamps if current_time - t < REQUEST_WINDOW]
        if len(request_timestamps) >= REQUEST_LIMIT:
            sleep_time = REQUEST_WINDOW - (current_time - request_timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            request_timestamps[:] = [t for t in request_timestamps if current_time - t < REQUEST_WINDOW]
        request_timestamps.append(current_time)

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
        logging.error(f"OKX API 初始化失败: {str(e)}")
        return False

def get_server_time():
    try:
        response = public_client.get_system_time()
        if response.get('code') == '0':
            server_time = int(response['data'][0]['ts'])
            return datetime.fromtimestamp(server_time / 1000.0)
        else:
            logging.error(f"获取服务器时间失败: {response.get('msg', '未知错误')}")
            return datetime.now()
    except Exception as e:
        logging.error(f"获取服务器时间失败: {str(e)}")
        return datetime.now()

def get_btc_price():
    global current_price
    rate_limit()
    try:
        ticker = market_client.get_ticker(instId='BTC-USDT')
        if ticker.get('code') != '0':
            raise Exception(f"获取行情失败: {ticker.get('msg', '未知错误')}")
        with lock:
            current_price = float(ticker['data'][0]['last'])
        return current_price
    except Exception as e:
        logging.error(f"获取价格失败: {str(e)}")
        return None

def get_klines(symbol, interval, limit='100'):
    rate_limit()
    try:
        timeframe_map = {'30m': '30min', '1h': '1H'}  # 添加 30m 分钟映射
        okx_interval = timeframe_map.get(interval, '30m')  # 默认 30m
        klines = market_client.get_candlesticks(instId=symbol, bar=okx_interval, limit=str(limit))
        if not klines.get('data'):
            logging.warning(f"未收到实时 K线数据（{symbol}, {okx_interval}），尝试获取历史 K 线数据")
            klines = market_client.get_history_candlesticks(instId=symbol, bar=okx_interval, limit=str(limit))
        if not klines.get('data'):
            logging.warning(f"未收到 K线数据（{symbol}, {okx_interval}）")
            return None
        df = pd.DataFrame(klines['data'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm'
        ])
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp'], errors='coerce'), unit='ms')
        df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
        df = df[df['confirm'] == '1']
        if df.empty:
            logging.warning(f"无已确认的 K线数据（{symbol}, {okx_interval}）")
            return None
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        if df['close'].isna().any():
            logging.warning(f"K线数据中存在空值（{symbol}, {okx_interval}），已尝试填补")
            df['close'] = df['close'].fillna(method='ffill')
        server_time = get_server_time()
        latest_kline_time = df['timestamp'].iloc[-1]
        if (server_time - latest_kline_time).total_seconds() > 172800:
            logging.warning(f"K线数据过旧，最新时间: {latest_kline_time}, 服务器时间: {server_time}")
            return None
        logging.info(f"获取 K线数据: {len(df)} 条, 最新 K线时间: {latest_kline_time}, 收盘价: {df['close'].iloc[-1]:.2f}")
        return df
    except Exception as e:
        logging.error(f"获取 K线失败（{symbol}}, {okx_interval}）: {str(e)}")
        return None

def calculate_rsi(df, period=14):
    try:
        if len(df) < period + 1:
            logging.warning(f"RSI 数据不足: {len(df)} 条")
            return None
        close = df['close'].values
        if np.any(np.isnan(close)) or np.all(close == close[0]):
            logging.warning("收盘价无效（含空值或无变化），RSI 计算跳过")
            return None
        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gains = np.zeros(len(close))
        avg_losses = np.zeros(len(close))
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        for i in range(period + 1, len(close)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        epsilon = 1e-10
        rs = avg_gains / (avg_losses + epsilon)
        rsi = 100 - (100 / (1 + rs))
        rsi = np.where(np.isfinite(rsi), rsi, np.nan)
        latest_rsi = rsi[-1]
        if np.isnan(latest_rsi):
            logging.warning("RSI 计算结果无效")
            return None
        logging.info(f"RSI 计算: 周期={period}, 最新 RSI={latest_rsi:.2f}")
        return pd.Series(rsi, index=df.index)
    except Exception as e:
        logging.error(f"RSI 计算错误: {str(e)}")
        return None

def calculate_macd(df, fast=12, slow=26, signal=9):
    try:
        if len(df) < slow + signal - 1:
            logging.warning(f"MACD 数据不足: {len(df)} 条")
            return None, None, None
        close = df['close'].values
        def ema(data, period):
            alpha = 2 / (period + 1)
            ema = np.zeros(len(data))
            ema[0] = data[0]
            for i in range(1, len(data)):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            return ema
        ema_fast = ema(close, fast)
        ema_slow = ema(close, slow)
        macd = ema_fast - ema_slow
        signal_line = ema(macd, signal)
        histogram = macd - signal_line
        logging.info(f"MACD 计算: MACD={macd[-1]:.2f}, 信号线={signal_line[-1]:.2f}, 柱状图={histogram[-1]:.2f}")
        return pd.Series(macd, index=df.index), pd.Series(signal_line, index=df.index), pd.Series(histogram, index=df.index)
    except Exception as e:
        logging.error(f"MACD 计算错误: {str(e)}")
        return None, None, None

def get_symbol_info(symbol='BTC-USDT'):
    rate_limit()
    try:
        info = public_client.get_instruments(instType='SPOT', instId=symbol)
        if info.get('code') != '0':
            raise Exception(f"获取交易对信息失败: {info.get('msg', '未知错误')}")
        quantity_precision = int(-np.log10(float(info['data'][0]['lotSz'])))
        min_qty = float(info['data'][0]['minSz'])
        return quantity_precision, min_qty
    except Exception as e:
        logging.error(f"获取交易对信息失败: {str(e)}")
        return 8, 0.0001

def place_order(side, quantity):
    try:
        quantity_precision, min_qty = get_symbol_info('BTC-USDT')
        quantity = round(quantity, quantity_precision)
        if quantity < min_qty:
            logging.warning(f"下单失败: 数量 {quantity} 小于最小值 {min_qty}")
            return None
        params = {
            'instId': 'BTC-USDT',
            'tdMode': 'cash',
            'side': side.lower(),
            'ordType': 'market',
            'sz': str(quantity),
            'clOrdId': f"order_{int(time.time())}"
        }
        order = trade_client.place_order(**params)
        if order['code'] == '0':
            logging.info(f"{side.upper()} 订单已下: 数量 {quantity:.6f} BTC")
            return order['data'][0]['ordId']
        else:
            logging.error(f"下单失败: {order['msg']}")
            return None
    except Exception as e:
        logging.error(f"下单失败: {str(e)}")
        return None

def get_balance():
    rate_limit()
    try:
        balance = account_client.get_account_balance()
        if balance.get('code') != '0':
            raise Exception(f"获取余额失败: {balance.get('msg', '未知错误')}")
        usdt = float(
            next((asset for asset in balance['data'][0]['details'] if asset['ccy'] == 'USDT'), {'availBal': '0'})['availBal'])
        btc = float(
            next((asset for asset in balance['data'][0]['details'] if asset['ccy'] == 'BTC'), {'availBal': '0'})['availBal'])
        logging.info(f"获取余额: {usdt:.2f} USDT, {btc:.6f} BTC")
        return usdt, btc
    except Exception as e:
        logging.error(f"获取余额失败: {str(e)}")
        return None, None

def rsi_trading(df=None):
    global last_rsi_buy_time, last_rsi_sell_time, latest_rsi
    if not rsi_enabled:
        logging.info("RSI 交易跳过: RSI 交易未启用")
        return
    try:
        logging.info("进入 RSI 交易逻辑")
        usdt_balance, btc_balance = get_balance()
        if usdt_balance is None or btc_balance is None:
            logging.warning("RSI 交易跳过: 获取余额失败")
            return
        logging.info(f"RSI 交易检查余额: {usdt_balance:.2f} USDT, {btc_balance:.6f} BTC")
        
        if usdt_balance == 0 and btc_balance == 0:
            logging.warning("RSI 交易跳过: 余额为 0")
            logging.info(f"RSI 买入未触发: RSI=未计算, 冷却时间={last_rsi_buy_time}")
            logging.info(f"RSI 卖出未触发: RSI=未计算, BTC余额={btc_balance:.6f}, 冷却时间={last_rsi_sell_time}")
            return
        
        if df is None:
            df = get_klines('BTC-USDT', rsi_timeframe)
        if df is None:
            logging.warning("RSI 交易跳过: 无 K 线数据")
            return
        rsi = calculate_rsi(df)
        if rsi is None:
            logging.warning("RSI 交易跳过: RSI 计算失败")
            return
        latest_rsi_value = rsi.iloc[-1]
        if pd.isna(latest_rsi_value):
            logging.warning("RSI 交易跳过: 无效 RSI 值")
            return
        with lock:
            latest_rsi = latest_rsi_value
        
        with lock:
            if current_price <= 0:
                logging.warning("RSI 交易跳过: 无效价格")
                return
            cooldown = timedelta(hours=1)
            now = datetime.now()
            
            # 买入逻辑：RSI <= 30，10% USDT 买入
            if latest_rsi <= rsi_buy_value and (last_rsi_buy_time is None or (now - last_rsi_buy_time) >= cooldown):
                quantity = (usdt_balance * rsi_buy_ratio / 100) / current_price
                if quantity <= 0:
                    logging.warning("RSI 买入跳过: 余额不足")
                    return
                logging.info(f"RSI 触发买入条件: RSI={latest_rsi:.2f}, 买入数量={quantity:.6f} BTC")
                order = place_order('buy', quantity)
                if order:
                    with lock:
                        last_rsi_buy_time = now
                    logging.info(f"RSI 买入成功: 数量={quantity:.6f} BTC, 订单 ID={order}")
                else:
                    logging.warning("RSI 买入失败: 订单未创建")
            else:
                logging.info(f"RSI 买入未触发: RSI={latest_rsi:.2f}, 冷却时间={last_rsi_buy_time}")
            
            # 卖出逻辑
            if btc_balance > 0 and (last_rsi_sell_time is None or (now - last_rsi_sell_time) >= cooldown):
                if latest_rsi >= rsi_full_sell_value:  # RSI >= 70，全卖
                    quantity = btc_balance
                    logging.info(f"RSI 触发全量卖出条件: RSI={latest_rsi:.2f}, 卖出数量={quantity:.6f} BTC")
                    order = place_order('sell', quantity)
                    if order:
                        with lock:
                            last_rsi_sell_time = now
                        logging.info(f"RSI 全量卖出成功: 数量={quantity:.6f} BTC, 订单 ID={order}")
                    else:
                        logging.warning("RSI 全量卖出失败: 订单未创建")
                elif latest_rsi >= rsi_sell_value:  # RSI >= 60，卖 50%
                    quantity = btc_balance * 0.5
                    logging.info(f"RSI 触发部分卖出条件: RSI={latest_rsi:.2f}, 卖出数量={quantity:.6f} BTC")
                    order = place_order('sell', quantity)
                    if order:
                        with lock:
                            last_rsi_sell_time = now
                        logging.info(f"RSI 部分卖出成功: 数量={quantity:.6f} BTC, 订单 ID={order}")
                    else:
                        logging.warning("RSI 部分卖出失败: 订单未创建")
                else:
                    logging.info(f"RSI 卖出未触发: RSI={latest_rsi:.2f}, BTC 余额={btc_balance:.6f}, 冷却时间={last_rsi_sell_time}")
            else:
                logging.info(f"RSI 卖出未触发: RSI={latest_rsi:.2f}, BTC 余额={btc_balance:.6f}, 冷却时间={last_rsi_sell_time}")
    except Exception as e:
        logging.error(f"RSI 交易错误: {str(e)}")

def macd_trading(df=None):
    global last_macd_buy_time, last_macd_sell_time, latest_macd, latest_signal, latest_histogram
    if not macd_enabled:
        logging.info("MACD 交易跳过: MACD 交易未启用")
        return
    try:
        logging.info("进入 MACD 交易逻辑")
        usdt_balance, btc_balance = get_balance()
        if usdt_balance is None or btc_balance is None:
            logging.warning("MACD 交易跳过: 获取余额失败")
            return
        logging.info(f"MACD 交易检查余额: {usdt_balance:.2f} USDT, {btc_balance:.6f} BTC")
        
        if usdt_balance == 0 and btc_balance == 0:
            logging.warning("MACD 交易跳过: 余额为 0")
            logging.info(f"MACD 买入未触发: MACD=未计算, 信号线=未计算, 柱状图=未计算, 冷却时间={last_macd_buy_time}")
            logging.info(f"MACD 卖出未触发: MACD=未计算, 信号线=未计算, 柱状图=未计算, BTC 余额={btc_balance:.6f}, 冷却时间={last_macd_sell_time}")
            return
        
        if df is None:
            df = get_klines('BTC-USDT', macd_timeframe)
        if df is None:
            logging.warning("MACD 交易跳过: 无 K 线数据")
            return
        macd, signal_line, histogram = calculate_macd(df)
        if macd is None or signal_line is None or histogram is None:
            logging.warning("MACD 交易跳过: MACD 计算失败")
            return
        latest_macd_value = macd.iloc[-1]
        latest_signal_value = signal_line.iloc[-1]
        latest_histogram_value = histogram.iloc[-1]
        prev_macd = macd.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        if any(pd.isna(x) for x in [latest_macd_value, prev_macd, latest_signal_value, prev_signal]):
            logging.warning("MACD 交易跳过: 无效 MACD 值")
            return
        with lock:
            latest_macd = latest_macd_value
            latest_signal = latest_signal_value
            latest_histogram = latest_histogram_value
        
        with lock:
            if current_price <= 0:
                logging.warning("MACD 交易跳过: 无效价格")
                return
            cooldown = timedelta(hours=1)
            now = datetime.now()
            
            # 买入逻辑：负数区间金叉
            if (latest_macd < 0 and prev_macd < prev_signal and 
                latest_macd > latest_signal and latest_histogram > 0 and 
                (last_macd_buy_time is None or (now - last_macd_buy_time) >= cooldown)):
                quantity = (usdt_balance * macd_buy_ratio / 100) / current_price
                if quantity <= 0:
                    logging.warning("MACD 买入跳过: 余额不足")
                    return
                logging.info(f"MACD 触发买入条件: MACD={latest_macd:.2f}, 信号线={latest_signal:.2f}, 柱状图={latest_histogram:.2f}, 买入数量={quantity:.6f} BTC")
                order = place_order('buy', quantity)
                if order:
                    with lock:
                        last_macd_buy_time = now
                    logging.info(f"MACD 买入成功: 数量={quantity:.6f} BTC, 订单 ID={order}")
                else:
                    logging.warning("MACD 买入失败: 订单未创建")
            else:
                logging.info(f"MACD 买入未触发: MACD={latest_macd:.2f}, 信号线={latest_signal:.2f}, 柱状图={latest_histogram:.2f}, 冷却时间={last_macd_buy_time}")
            
            # 卖出逻辑：正数区间死叉
            if (latest_macd > 0 and prev_macd > prev_signal and 
                latest_macd < latest_signal and latest_histogram < 0 and 
                btc_balance > 0 and 
                (last_macd_sell_time is None or (now - last_macd_sell_time) >= cooldown)):
                quantity = btc_balance * 0.2
                logging.info(f"MACD 触发卖出条件: MACD={latest_macd:.2f}, 信号线={latest_signal:.2f}, 柱状图={latest_histogram:.2f}, 卖出数量={quantity:.6f} BTC")
                order = place_order('sell', quantity)
                if order:
                    with lock:
                        last_macd_sell_time = now
                    logging.info(f"MACD 卖出成功: 数量={quantity:.6f} BTC, 订单 ID={order}")
                else:
                    logging.warning("MACD 卖出失败: 订单未创建")
            else:
                logging.info(f"MACD 卖出未触发: MACD={latest_macd:.2f}, 信号线={latest_signal:.2f}, 柱状图={latest_histogram:.2f}, BTC 余额={btc_balance:.6f}, 冷却时间={last_macd_sell_time}")
    except Exception as e:
        logging.error(f"MACD 交易错误: {str(e)}")

def trading_loop():
    start_time = time.time()
    logging.info(f"交易循环开始: {start_time}")
    global running, current_price, latest_rsi, latest_macd, latest_signal, latest_histogram
    running = True
    try:
        price = get_btc_price()
        if not price:
            logging.warning(f"价格获取失败，使用缓存价格: ${current_price:.2f}")
        else:
            with lock:
                current_price = price
            logging.info(f"价格更新: ${price:.2f}")
        
        usdt_balance, btc_balance = get_balance()
        if usdt_balance is None or btc_balance is None:
            logging.warning("余额获取失败，跳过交易")
            return
        logging.info(f"余额更新: {usdt_balance:.2f} USDT, {btc_balance:.6f} BTC")
        
        # 获取一次 K 线数据，复用
        df = get_klines('BTC-USDT', rsi_timeframe, limit=100)
        if df is None:
            logging.warning("K 线数据获取失败，跳过 RSI 和 MACD")
            return
        
        if rsi_enabled:
            rsi = calculate_rsi(df)
            if rsi is not None and not pd.isna(rsi.iloc[-1]):
                with lock:
                    latest_rsi = rsi.iloc[-1]
                logging.info(f"RSI 更新: {latest_rsi:.2f}")
            else:
                logging.warning("RSI 计算失败")
        
        if macd_enabled:
            macd, signal_line, histogram = calculate_macd(df)
            if macd is not None and not pd.isna(macd.iloc[-1]):
                with lock:
                    latest_macd = macd.iloc[-1]
                    latest_signal = signal_line.iloc[-1]
                    latest_histogram = histogram.iloc[-1]
                logging.info(f"MACD 更新: MACD={latest_macd:.2f}, 信号线={latest_signal:.2f}, 柱状图={latest_histogram:.2f}")
            else:
                logging.warning("MACD 计算失败")
        
        logging.info(f"开始 RSI 交易: {time.time() - start_time:.2f} 秒")
        rsi_trading(df)  # 传递 K 线数据
        logging.info(f"完成 RSI 交易: {time.time() - start_time:.2f} 秒")
        logging.info(f"开始 MACD 交易: {time.time() - start_time:.2f} 秒")
        macd_trading(df)  # 传递 K 线数据
        logging.info(f"完成 MACD 交易: {time.time() - start_time:.2f} 秒")
    except Exception as e:
        logging.error(f"交易循环错误: {str(e)}")
    finally:
        running = False
        logging.info(f"交易循环结束: 耗时 {time.time() - start_time:.2f} 秒")

def main():
    start_time = time.time()
    logging.info("启动交易机器人")
    api_key = os.getenv('OKX_API_KEY')
    api_secret = os.getenv('OKX_API_SECRET')
    passphrase = os.getenv('OKX_PASSPHRASE')
    flag = os.getenv('OKX_FLAG', '0')  # 默认实盘
    if not all([api_key, api_secret, passphrase]):
        logging.error("环境变量缺少 API 密钥信息")
        return
    if init_okx(api_key, api_secret, passphrase, flag):
        try:
            trading_loop()  # 单次运行
            logging.info("交易循环完成")
        except Exception as e:
            logging.error(f"交易循环错误: {str(e)}")
    else:
        logging.error("初始化 OKX API 失败")
    logging.info(f"主程序结束: 总耗时 {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"主程序错误: {str(e)}")
