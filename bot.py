from binance.spot import Spot
import numpy as np
import json
import os
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from queue import Queue
import traceback
import requests

# 设置日志（带轮转）
log_handler = RotatingFileHandler(
    'bot.log',
    maxBytes=1_000_000,  # 每个日志文件最大1MB
    backupCount=5  # 保留5个备份文件
)
logging.basicConfig(
    handlers=[log_handler],
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

# API 密钥存储文件（仅用于本地测试，GitHub Actions 使用环境变量）
CONFIG_FILE = 'binance_config.json'

# 全局变量
client = None
running = True
update_queue = Queue(maxsize=100)
ALL_COINS = ['USDT', 'USDC', 'FDUSD', 'DAI', 'XUSD', 'TUSD', 'USDP']
DEFAULT_PAIRS = ['DAI/USDT', 'FDUSD/USDT', 'USDC/USDT', 'XUSD/USDT', 'TUSD/USDT', 'USDP/USDT']
selected_pairs = DEFAULT_PAIRS
current_prices = {pair: 0.0 for pair in selected_pairs}
ma_values = {pair: 0.0 for pair in selected_pairs}
balances = {coin: 0.0 for coin in ALL_COINS}
last_trade_time = None
trade_speed = 0.1  # 默认10%
ma_threshold = 0.0001  # 默认0.01%
ma_period = 30  # 默认MA30
trade_cooldown = 3600  # 每小时交易一次（秒）
kline_interval = '4h'  # 默认4小时K线

# 加载 API 密钥（仅用于本地测试）
def load_config():
    if os.path.exists(CONFIG_FILE):
        if os.path.getsize(CONFIG_FILE) == 0:
            logging.error("Config file is empty")
            return {}
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logging.error("Config file is empty")
                    return {}
                return json.loads(content)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse config file: {str(e)}")
            return {}
        except Exception as e:
            logging.error(f"Error loading config: {str(e)}")
            return {}
    logging.error("Config file does not exist")
    return {}

# 初始化 Binance API
def init_binance(api_key, api_secret):
    global client
    try:
        proxies = {
            'https': 'https://sub-1.smjcdh.top/smjc/api/v1/client/subscribe?token=fcc6d43a359c6b90f9cffbc1cd82705d'
        }
        client = Spot(api_key=api_key, api_secret=api_secret, base_url='https://api.binance.com')
        client.time()
        logging.info("Binance API initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Binance API initialization failed: {str(e)}")
        return False

# 验证交易对
def validate_pairs(pairs):
    try:
        response = client.exchange_info()
        valid_pairs = []
        for pair in pairs:
            symbol = pair.replace('/', '')
            base_coin = pair.split('/')[0]
            if any(s['symbol'] == symbol for s in response['symbols']) and base_coin in ALL_COINS:
                valid_pairs.append(pair)
                logging.info(f"Pair {pair} validated successfully")
            else:
                logging.warning(f"Pair {pair} is not available or base coin {base_coin} is not supported")
        if not valid_pairs:
            valid_pairs = ['USDC/USDT']
            logging.warning("No valid pairs, defaulting to USDC/USDT")
        return valid_pairs
    except Exception as e:
        logging.error(f"Failed to validate pairs: {str(e)} - Stack: {traceback.format_exc()}")
        return ['USDC/USDT']

# 获取交易对当前价格
def get_pair_price(symbol):
    try:
        ticker = client.ticker_price(symbol=symbol.replace('/', ''))
        return float(ticker['price'])
    except Exception as e:
        logging.error(f"Failed to get price for {symbol}: {str(e)}")
        return None

# 获取 K 线数据并计算 MA
def get_klines(symbol, interval='4h', limit=31, ma_period=30):
    try:
        klines = client.klines(symbol=symbol.replace('/', ''), interval=interval, limit=limit)
        closes = np.array([float(k[4]) for k in klines[:-1]], dtype=np.float64)
        if np.isnan(closes).any():
            closes = np.nan_to_num(closes, nan=closes[~np.isnan(closes)][-1])
        ma = np.mean(closes[-ma_period:]) if len(closes) >= ma_period else None
        current_price = float(klines[-1][4])
        return current_price, ma
    except Exception as e:
        logging.error(f"Failed to get klines for {symbol}: {str(e)}")
        return None, None

# 获取交易对信息
def get_symbol_info(symbol):
    try:
        info = client.get_symbol_info(symbol)
        quantity_precision = info['quantityPrecision']
        min_qty = float(next(filter(lambda x: x['filterType'] == 'LOT_SIZE', info['filters']))['minQty'])
        return quantity_precision, min_qty
    except Exception as e:
        logging.error(f"Failed to get symbol info for {symbol}: {str(e)}")
        return 8, 0.0001

# 下单函数
def place_order(symbol, side, quantity):
    try:
        quantity_precision, min_qty = get_symbol_info(symbol.replace('/', ''))
        quantity = round(quantity, quantity_precision)
        if quantity < min_qty:
            logging.warning(f"Order failed: Quantity {quantity} below min {min_qty}")
            return None
        params = {
            'symbol': symbol.replace('/', ''),
            'side': side.upper(),
            'type': 'MARKET',
            'quantity': f"{quantity:.{quantity_precision}f}"
        }
        order = client.new_order(**params)
        logging.info(f"Order placed: {side} {quantity} {symbol}")
        return order
    except Exception as e:
        logging.error(f"Order failed for {symbol}: {str(e)}")
        return None

# 更新账户余额
def update_balances():
    global balances
    try:
        account = client.account()
        for coin in ALL_COINS:
            balance = float(next((asset['free'] for asset in account['balances'] if asset['asset'] == coin), 0.0))
            balances[coin] = balance
        logging.info(f"Balances updated: {balances}")
    except Exception as e:
        logging.error(f"Failed to update balances: {str(e)}")

# 执行交易
def execute_trade(from_coin, to_coin, amount, prices):
    global balances
    if from_coin == to_coin:
        logging.warning(f"Invalid trade: {from_coin} -> {to_coin}")
        return False, f"Invalid trade: {from_coin} -> {to_coin}"
    if from_coin not in ALL_COINS or to_coin not in ALL_COINS:
        logging.warning(f"Unsupported coin: {from_coin} or {to_coin}")
        return False, f"Unsupported coin: {from_coin} or {to_coin}"

    amount = balances[from_coin] * trade_speed
    amount = int(amount)
    if amount < 5:
        amount = 5 if balances[from_coin] >= 5 else int(balances[from_coin])
    if amount < 5:
        logging.warning(f"Insufficient amount: {from_coin} (balance: {balances[from_coin]:.2f})")
        return False, f"Insufficient amount: {from_coin} (balance: {balances[from_coin]:.2f})"

    try:
        if from_coin != 'USDT' and to_coin == 'USDT':
            pair = f"{from_coin}/USDT"
            if pair not in prices or prices[pair] is None:
                logging.warning(f"No price for pair: {pair}")
                return False, f"No price for pair: {pair}"
            order = place_order(pair, 'sell', amount)
            if order:
                to_amount = float(order['cummulativeQuoteQty'])
                logging.info(f"Sell order successful: {amount:.0f} {from_coin} -> USDT, Order ID: {order['orderId']}")
                update_balances()
                return True, f"Trade successful: {amount:.0f} {from_coin} -> {to_amount:.0f} USDT"
        elif from_coin == 'USDT' and to_coin != 'USDT':
            pair = f"{to_coin}/USDT"
            if pair not in prices or prices[pair] is None:
                logging.warning(f"No price for pair: {pair}")
                return False, f"No price for pair: {pair}"
            to_amount = amount / prices[pair]
            to_amount = int(to_amount)
            if to_amount < 5:
                to_amount = 5 if balances[from_coin] >= 5 * prices[pair] else int(balances[from_coin] / prices[pair])
            if to_amount < 5:
                logging.warning(f"Target amount too low: {to_coin} (available: {to_amount})")
                return False, f"Target amount too low: {to_coin} (available: {to_amount})"
            order = place_order(pair, 'buy', to_amount)
            if order:
                logging.info(f"Buy order successful: USDT -> {to_amount:.0f} {to_coin}, Order ID: {order['orderId']}")
                update_balances()
                return True, f"Trade successful: {amount:.0f} USDT -> {to_amount:.0f} {to_coin}"
        else:
            usdt_pair = f"{from_coin}/USDT"
            target_pair = f"{to_coin}/USDT"
            if usdt_pair not in prices or prices[usdt_pair] is None or target_pair not in prices or prices[target_pair] is None:
                logging.warning(f"No price for pair: {usdt_pair} or {target_pair}")
                return False, f"No price for pair: {usdt_pair} or {target_pair}"
            sell_order = place_order(usdt_pair, 'sell', amount)
            if not sell_order:
                logging.warning(f"Sell order failed: {from_coin} -> USDT")
                return False, f"Sell order failed: {from_coin} -> USDT"
            usdt_amount = float(sell_order['cummulativeQuoteQty'])
            logging.info(f"Sell order successful: {amount:.0f} {from_coin} -> USDT, Order ID: {sell_order['orderId']}")
            to_amount = usdt_amount / prices[target_pair]
            to_amount = int(to_amount)
            if to_amount < 5:
                to_amount = 5
                usdt_amount = to_amount * prices[target_pair]
                amount = usdt_amount / prices[usdt_pair]
                amount = int(amount)
                if amount < 5:
                    amount = 5 if balances[from_coin] >= 5 else int(balances[from_coin])
                if amount < 5:
                    logging.warning(f"Insufficient amount after adjustment: {from_coin} (need: {amount})")
                    return False, f"Insufficient amount after adjustment: {from_coin} (need: {amount})"
                sell_order = place_order(usdt_pair, 'sell', amount)
                if not sell_order:
                    logging.warning(f"Adjusted sell order failed: {from_coin} -> USDT")
                    return False, f"Adjusted sell order failed: {from_coin} -> USDT"
                usdt_amount = float(sell_order['cummulativeQuoteQty'])
                logging.info(f"Adjusted sell order successful: {amount:.0f} {from_coin} -> USDT, Order ID: {sell_order['orderId']}")
            buy_order = place_order(target_pair, 'buy', to_amount)
            if buy_order:
                logging.info(f"Buy order successful: USDT -> {to_amount:.0f} {to_coin}, Order ID: {buy_order['orderId']}")
                update_balances()
                return True, f"Trade successful: {amount:.0f} {from_coin} -> {to_amount:.0f} {to_coin}"
        return False, "Trade failed"
    except Exception as e:
        logging.error(f"Trade failed: {str(e)} - Stack: {traceback.format_exc()}")
        return False, f"Trade failed: {str(e)}"

# 主交易循环
def trading_loop():
    global running, last_trade_time
    while running:
        try:
            # 更新价格和 MA
            for pair in selected_pairs:
                base_coin = pair.split('/')[0]
                if base_coin not in ALL_COINS:
                    logging.warning(f"Skipping {pair}: Base coin {base_coin} not in supported list")
                    continue
                price, ma = get_klines(pair, interval=kline_interval, ma_period=ma_period)
                if price and ma:
                    current_prices[pair] = price
                    ma_values[pair] = ma
                    logging.info(f"Updated {pair}: Price={price:.4f}, MA={ma:.4f}")
                else:
                    logging.warning(f"Failed to get data for {pair}")

            # 更新余额
            update_balances()

            # 交易逻辑（所有币种共享每小时一次检查）
            now = datetime.now()
            if last_trade_time is None or (now - last_trade_time).total_seconds() >= trade_cooldown:
                last_trade_time = now
                above_ma_coins = []
                below_ma_coins = []
                trade_speeds = {}
                for pair in selected_pairs:
                    base_coin = pair.split('/')[0]
                    if base_coin not in ALL_COINS:
                        logging.warning(f"Skipping {pair}: Base coin {base_coin} not in supported list")
                        continue
                    price = current_prices.get(pair)
                    ma = ma_values.get(pair)
                    if price and ma:
                        diff_percent = abs(price - ma) / ma
                        if diff_percent > ma_threshold:
                            trade_speeds[base_coin] = 0.5 if diff_percent > 0.0005 else trade_speed
                            if price > ma:
                                above_ma_coins.append(base_coin)
                            elif price < ma:
                                below_ma_coins.append(base_coin)
                        else:
                            logging.info(f"{pair} deviation from MA ({diff_percent*100:.2f}%) below threshold ({ma_threshold*100:.2f}%)")
                logging.info(f"Above MA: {above_ma_coins}, Below MA: {below_ma_coins}")
                for from_coin in above_ma_coins:
                    if from_coin == 'USDT':
                        continue
                    for to_coin in below_ma_coins + ['USDT']:
                        if to_coin == from_coin:
                            continue
                        success, msg = execute_trade(from_coin, to_coin, balances[from_coin], current_prices)
                        logging.info(msg)
                        if success:
                            update_balances()
                        break  # 每个币种在一次循环中最多交易一次
                if 'USDT' not in above_ma_coins:
                    for to_coin in below_ma_coins:
                        if to_coin == 'USDT':
                            continue
                        success, msg = execute_trade('USDT', to_coin, balances['USDT'], current_prices)
                        logging.info(msg)
                        if success:
                            update_balances()
            time.sleep(trade_cooldown)  # 每小时检查一次
        except Exception as e:
            logging.error(f"Trading loop error: {str(e)} - Stack: {traceback.format_exc()}")
            time.sleep(60)  # 出错后等待1分钟再重试

# 主函数
def main():
    global selected_pairs
    # 从环境变量加载 API 密钥
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    if not api_key or not api_secret:
        logging.error("API key or secret not found in environment variables")
        return

    # 初始化 Binance API
    if not init_binance(api_key, api_secret):
        logging.error("Exiting due to Binance API initialization failure")
        return

    # 验证交易对
    selected_pairs = validate_pairs(DEFAULT_PAIRS)
    if not selected_pairs:
        logging.error("No valid trading pairs available, exiting")
        return

    # 更新初始余额
    update_balances()

    # 启动交易循环
    trading_loop()

if __name__ == "__main__":
    main()
