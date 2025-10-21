import logging
from logging.handlers import RotatingFileHandler
import os
import json
import time
import pandas as pd
import numpy as np
import okx.Trade as Trade
import okx.MarketData as MarketData
import okx.Account as Account
import okx.PublicData as PublicData
import traceback
import uuid

# 设置日志
logging.basicConfig(
    handlers=[RotatingFileHandler('trading_bot.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置文件
CONFIG_FILE = 'okx_config.json'
PARAMS_FILE = '参数.json'

# 全局变量
trade_client = None
market_client = None
account_client = None
public_client = None
symbols = ['BTC-USDT-SWAP']
state = {symbol: {'current_price': 0.0, 'latest_rsi': None, 'latest_macd': None, 'latest_signal': None, 'latest_histogram': None} for symbol in symbols}
SYMBOL_PARAMS = {}  # 动态加载的参数


def load_params():
    """加载交易参数"""
    global SYMBOL_PARAMS
    if not os.path.exists(PARAMS_FILE):
        logging.error(f"交易参数文件 {PARAMS_FILE} 不存在")
        raise FileNotFoundError(f"交易参数文件 {PARAMS_FILE} 不存在")

    try:
        with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                raise ValueError("交易参数文件为空")
            raw_params = json.loads(content)

        # 转换为实际参数格式，忽略description字段
        SYMBOL_PARAMS = {}
        required_keys = [
            'RSI_TIMEFRAME', 'MACD_TIMEFRAME', 'RSI_BUY_VALUE', 'RSI_SELL_VALUE',
            'BUY_RATIO', 'LEVERAGE', 'MARGIN_MODE', 'TAKE_PROFIT', 'STOP_LOSS'
        ]

        for symbol in symbols:
            if symbol not in raw_params:
                raise ValueError(f"缺少 {symbol} 的参数配置")
            SYMBOL_PARAMS[symbol] = {}
            for key in required_keys:
                if key not in raw_params[symbol]:
                    raise ValueError(f"{symbol} 缺少参数 {key}")
                if 'value' not in raw_params[symbol][key]:
                    raise ValueError(f"{symbol} 的 {key} 缺少 'value' 字段")
                SYMBOL_PARAMS[symbol][key] = raw_params[symbol][key]['value']

            # 验证参数类型
            for key in ['RSI_BUY_VALUE', 'RSI_SELL_VALUE', 'BUY_RATIO', 'LEVERAGE', 'TAKE_PROFIT', 'STOP_LOSS']:
                if not isinstance(SYMBOL_PARAMS[symbol][key], (int, float)):
                    raise ValueError(f"{symbol} 的 {key} 必须是数值类型")
            if SYMBOL_PARAMS[symbol]['MARGIN_MODE'] not in ['cross', 'isolated']:
                raise ValueError(f"{symbol} 的 MARGIN_MODE 必须是 'cross' 或 'isolated'")

        logging.info("交易参数加载成功")
        for symbol in symbols:
            logging.info(f"{symbol} 参数: {SYMBOL_PARAMS[symbol]}")
        return True
    except Exception as e:
        logging.error(f"加载交易参数失败: {str(e)}")
        raise ValueError(f"加载交易参数失败: {str(e)}")


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
    params = SYMBOL_PARAMS[symbol]
    try:
        response = account_client.set_leverage(instId=symbol, lever=str(params['LEVERAGE']), mgnMode=params['MARGIN_MODE'])
        if response.get('code') == '0':
            logging.info(f"设置杠杆成功: {symbol}, 杠杆={params['LEVERAGE']}, 模式={params['MARGIN_MODE']}")
            return True
        else:
            raise Exception(f"设置杠杆失败: {response.get('msg', '未知错误')}")
    except Exception as e:
        logging.error(f"设置杠杆失败: {symbol}, {str(e)}")
        return False

def get_price(symbol):
    for attempt in range(3):
        try:
            ticker = market_client.get_ticker(instId=symbol)
            if ticker.get('code') != '0':
                raise Exception(f"获取行情失败: {ticker.get('msg', '未知错误')}")
            price = float(ticker['data'][0]['last'])
            state[symbol]['current_price'] = price
            return price
        except Exception as e:
            logging.error(f"获取价格失败: {symbol} (尝试 {attempt + 1}/3): {str(e)}")
            time.sleep(2)
    logging.error(f"获取价格失败: {symbol}, 重试次数耗尽")
    return None

def get_klines(symbol, interval, limit=100):
    for attempt in range(5):
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
            df = df.dropna(subset=['close'])
            df = df[df['close'] > 0]
            if len(df) < 34:
                logging.warning(f"有效K线数据不足: {symbol}, {len(df)} 条，需至少 34 条")
                return None
            timeframe_map = {
                '1m': pd.Timedelta(minutes=1), '5m': pd.Timedelta(minutes=5), '15m': pd.Timedelta(minutes=15),
                '30m': pd.Timedelta(minutes=30), '1H': pd.Timedelta(hours=1), '4H': pd.Timedelta(hours=4),
                '1D': pd.Timedelta(days=1)
            }
            expected_diff = timeframe_map.get(interval)
            if expected_diff is None:
                logging.error(f"不支持的K线周期: {interval}")
                return None
            if len(df) > 1 and not df['ts'].diff().iloc[1:].eq(expected_diff).all():
                logging.warning(f"K线时间戳不连续: {symbol}, 预期时间差 {expected_diff}, 周期 {interval}")
                return None
            logging.info(f"K线数据统计: {symbol}, 长度={len(df)}, 收盘价最小={df['close'].min():.2f}, 最大={df['close'].max():.2f}, NaN={df['close'].isna().sum()}")
            logging.info(f"获取K线数据: {symbol}, {len(df)} 条, 最新时间: {df['ts'].iloc[-1]}")
            return df
        except Exception as e:
            logging.error(f"获取K线错误: {symbol} (尝试 {attempt + 1}/5): {str(e)}\n{traceback.format_exc()}")
            time.sleep(2)
    logging.error(f"获取K线失败: {symbol}, 重试次数耗尽")
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
            logging.warning(f"MACD 数据不足: {len(df)} 条，需至少 {slow + signal - 1} 条")
            return None, None, None
        close = df['close'].values
        if np.any(np.isnan(close)) or np.any(close <= 0):
            logging.warning("K线数据包含无效收盘价（NaN 或 <= 0）")
            return None, None, None
        ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
        ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
        macd = ema_fast - ema_slow
        signal_line = pd.Series(macd).ewm(span=signal, adjust=False).mean().values
        histogram = macd - signal_line
        macd_series = pd.Series(macd, index=df.index)
        signal_series = pd.Series(signal_line, index=df.index)
        histogram_series = pd.Series(histogram, index=df.index)
        if len(macd_series) < 2 or len(signal_series) < 2:
            logging.warning(f"MACD 输出长度不足: macd={len(macd_series)}, signal={len(signal_series)}")
            return None, None, None
        logging.info(f"MACD 计算: MACD={macd[-1]:.2f}, 信号线={signal_line[-1]:.2f}, 柱状图={histogram[-1]:.2f}, 数据长度={len(macd_series)}")
        return macd_series, signal_series, histogram_series
    except Exception as e:
        logging.error(f"MACD 计算错误: {str(e)}\n{traceback.format_exc()}")
        return None, None, None

def get_symbol_info(symbol):
    try:
        info = public_client.get_instruments(instType='SWAP', instId=symbol)
        if info.get('code') != '0':
            raise Exception(f"获取交易对信息失败: {info.get('msg', '未知错误')}")
        ct_val = float(info['data'][0]['ctVal'])
        min_qty = float(info['data'][0]['minSz'])
        tick_sz = float(info['data'][0]['tickSz'])
        lot_sz = float(info['data'][0].get('lotSz', min_qty))  # 获取lotSz，默认为min_qty
        logging.info(f"{symbol} 合约信息: ct_val={ct_val}, min_qty={min_qty}, tick_sz={tick_sz}, lot_sz={lot_sz}")
        return ct_val, min_qty, tick_sz, lot_sz
    except Exception as e:
        logging.error(f"获取交易对信息失败: {symbol}, {str(e)}\n{traceback.format_exc()}")
        return 0.01, 0.01, 0.01, 0.01

def place_order(symbol, side, pos_side, quantity):
    params = SYMBOL_PARAMS[symbol]
    try:
        ct_val, min_qty, tick_sz, lot_sz = get_symbol_info(symbol)
        quantity_in_contracts = quantity / ct_val
        quantity_in_contracts = max(round(quantity_in_contracts / lot_sz) * lot_sz, min_qty)  # 确保数量是lot_sz的倍数
        if quantity_in_contracts < min_qty:
            logging.warning(f"下单失败: {symbol}, 数量 {quantity_in_contracts:.2f} 张小于最小值 {min_qty:.2f} 张")
            return None

        current_price = state[symbol]['current_price']
        # 使用固定百分比止盈止损
        tp_price = round(current_price * (1 + params['TAKE_PROFIT'] / 100 if pos_side == 'long' else 1 - params['TAKE_PROFIT'] / 100), -int(np.log10(tick_sz)))
        sl_price = round(current_price * (1 - params['STOP_LOSS'] / 100 if pos_side == 'long' else 1 + params['STOP_LOSS'] / 100), -int(np.log10(tick_sz)))

        algo_order = {
            'tpTriggerPx': str(tp_price),
            'tpOrdPx': '-1',
            'slTriggerPx': str(sl_price),
            'slOrdPx': '-1',
            'tpOrdKind': 'condition',
            'slTriggerPxType': 'last',
            'tpTriggerPxType': 'last'
        }
        order_params = {
            'instId': symbol,
            'tdMode': params['MARGIN_MODE'],
            'side': side.lower(),
            'posSide': pos_side.lower(),
            'ordType': 'market',
            'sz': str(round(quantity_in_contracts, 2)),
            'clOrdId': str(uuid.uuid4()).replace('-', '')[:32],
            'attachAlgoOrds': [algo_order]
        }
        logging.info(f"{symbol} 准备下单: 方向={side}, 持仓方向={pos_side}, 数量={quantity_in_contracts:.2f} 张 (约 {quantity_in_contracts * ct_val:.6f} {symbol.split('-')[0]})")
        order = trade_client.place_order(**order_params)
        if order['code'] == '0':
            action = '开多' if side == 'buy' and pos_side == 'long' else '开空' if side == 'sell' and pos_side == 'short' else '平仓'
            logging.info(f"{symbol} {action} 订单已下: 数量 {quantity_in_contracts:.2f} 张, 止盈价格={tp_price:.2f}, 止损价格={sl_price:.2f}")
            return order['data'][0]['ordId']
        else:
            logging.error(f"下单失败: {symbol}, 错误码={order['code']}, 错误信息={order['msg']}, 详情={order.get('data', '无详细信息')}")
            return None
    except Exception as e:
        logging.error(f"下单失败: {symbol}, {str(e)}\n{traceback.format_exc()}")
        return None

def get_balance(symbol):
    for attempt in range(3):
        try:
            balance = account_client.get_account_balance()
            if balance.get('code') != '0':
                raise Exception(f"获取余额失败: {balance.get('msg', '未知错误')}")
            # 记录原始余额数据
            logging.debug(f"{symbol} 账户余额原始数据: {balance.get('data', '无数据')}")
            if not balance.get('data') or not balance['data'][0].get('details'):
                logging.warning(f"{symbol} 账户余额数据为空或无 details 字段")
                usdt = 0.0  # 默认余额为0
            else:
                usdt_asset = next((asset for asset in balance['data'][0]['details'] if asset['ccy'] == 'USDT'), {'availEq': '0'})
                usdt = float(usdt_asset['availEq']) if usdt_asset['availEq'] else 0.0
            total_equity = float(balance['data'][0]['totalEq']) if balance['data'][0].get('totalEq') else 0.0
            positions = account_client.get_positions(instType='SWAP', instId=symbol)
            if positions.get('code') != '0':
                raise Exception(f"获取持仓失败: {positions.get('msg', '未知错误')}")
            long_qty = 0
            short_qty = 0
            long_avg_price = 0.0
            short_avg_price = 0.0
            if positions.get('data'):
                for pos in positions['data']:
                    if pos['instId'] == symbol:
                        qty = float(pos['pos']) if pos['pos'] else 0.0
                        avg_price = float(pos['avgPx']) if pos['avgPx'] else 0.0
                        if pos['posSide'] == 'long':
                            long_qty = qty
                            long_avg_price = avg_price
                        elif pos['posSide'] == 'short':
                            short_qty = qty
                            short_avg_price = avg_price
            logging.info(f"{symbol} 余额获取成功: USDT={usdt:.2f}, 总权益={total_equity:.2f}")
            return usdt, long_qty, short_qty, long_avg_price, short_avg_price, total_equity
        except Exception as e:
            logging.error(f"获取余额失败: {symbol} (尝试 {attempt + 1}/3): {str(e)}\n{traceback.format_exc()}")
            time.sleep(2)
    logging.error(f"获取余额失败: {symbol}, 重试次数耗尽")
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 默认返回 0

def check_take_profit_stop_loss(symbol, long_qty, short_qty, long_avg_price, short_avg_price, current_price, rsi, macd, signal):
    params = SYMBOL_PARAMS[symbol]
    try:
        pos_side = None
        qty = 0.0
        reason = None
        if long_qty > 0 and long_avg_price > 0:
            profit_diff = (current_price - long_avg_price) / long_avg_price * 100
            if profit_diff >= params['TAKE_PROFIT']:
                pos_side = 'long'
                qty = long_qty
                reason = '固定百分比止盈'
            elif profit_diff <= -params['STOP_LOSS']:
                pos_side = 'long'
                qty = long_qty
                reason = '固定百分比止损'
            elif rsi is not None and rsi >= params['RSI_SELL_VALUE']:
                pos_side = 'long'
                qty = long_qty
                reason = f"RSI 高于{params['RSI_SELL_VALUE']}"
            elif macd is not None and signal is not None and len(macd) >= 2 and len(signal) >= 2 and macd.iloc[-1] > 0 and macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
                pos_side = 'long'
                qty = long_qty
                reason = 'MACD 死叉'
        if short_qty > 0 and short_avg_price > 0:
            profit_diff = (short_avg_price - current_price) / short_avg_price * 100
            if profit_diff >= params['TAKE_PROFIT']:
                pos_side = 'short'
                qty = short_qty
                reason = '固定百分比止盈'
            elif profit_diff <= -params['STOP_LOSS']:
                pos_side = 'short'
                qty = short_qty
                reason = '固定百分比止损'
            elif rsi is not None and rsi <= params['RSI_BUY_VALUE']:
                pos_side = 'short'
                qty = short_qty
                reason = f"RSI 低于{params['RSI_BUY_VALUE']}"
            elif macd is not None and signal is not None and len(macd) >= 2 and len(signal) >= 2 and macd.iloc[-1] < 0 and macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
                pos_side = 'short'
                qty = short_qty
                reason = 'MACD 金叉'
        return pos_side, qty, reason
    except Exception as e:
        logging.error(f"检查止盈止损错误: {symbol}, {str(e)}\n{traceback.format_exc()}")
        return None, 0.0, None

def execute_trading_logic(symbol):
    """执行数据获取、计算和交易逻辑"""
    params = SYMBOL_PARAMS[symbol]
    try:
        # 获取RSI的K线数据
        df_rsi = get_klines(symbol, params['RSI_TIMEFRAME'])
        if df_rsi is None:
            logging.warning(f"{symbol} 无RSI K线数据，跳过交易")
            return False

        # 获取MACD的K线数据
        df_macd = get_klines(symbol, params['MACD_TIMEFRAME'])
        if df_macd is None:
            logging.warning(f"{symbol} 无MACD K线数据，跳过交易")
            return False

        # 计算RSI
        rsi = calculate_rsi(df_rsi)
        if rsi is None:
            logging.warning(f"{symbol} RSI计算失败，跳过交易")
            return False
        state[symbol]['latest_rsi'] = rsi.iloc[-1]

        # 计算MACD
        macd, signal, histogram = calculate_macd(df_macd)
        if macd is None or signal is None or histogram is None:
            logging.warning(f"{symbol} MACD计算失败，跳过交易")
            return False
        state[symbol]['latest_macd'] = macd.iloc[-1]
        state[symbol]['latest_signal'] = signal.iloc[-1]
        state[symbol]['latest_histogram'] = histogram.iloc[-1]

        # 使用固定仓位比例
        dynamic_buy_ratio = params['BUY_RATIO']
        logging.info(f"{symbol} 固定仓位比例: {dynamic_buy_ratio:.2f}")

        # 获取账户余额和持仓
        usdt_balance, long_qty, short_qty, long_avg_price, short_avg_price, total_equity = get_balance(symbol)
        if usdt_balance < 0:  # 检查负余额
            logging.warning(f"{symbol} USDT余额无效: {usdt_balance:.2f}")
            return False
        if usdt_balance == 0 and total_equity == 0:  # 检查账户是否完全无资金
            logging.warning(f"{symbol} 账户无可用资金: USDT={usdt_balance:.2f}, 总权益={total_equity:.2f}")
            return False
        # 获取当前价格
        price = get_price(symbol)
        if price:
            state[symbol]['current_price'] = price
            logging.info(f"{symbol} 当前价格: ${price:.2f}")
        else:
            logging.warning(f"{symbol} 无法获取价格，跳过交易")
            return False

        logging.info(f"{symbol} 账户状态: USDT余额={usdt_balance:.2f}, 总权益={total_equity:.2f}, 多仓={long_qty:.2f}, 空仓={short_qty:.2f}")

        # 检查止盈止损或反向信号
        pos_side, qty, reason = check_take_profit_stop_loss(symbol, long_qty, short_qty, long_avg_price, short_avg_price, state[symbol]['current_price'], state[symbol]['latest_rsi'], macd, signal)
        if pos_side and qty > 0:
            order = place_order(symbol, 'sell' if pos_side == 'long' else 'buy', pos_side, qty)
            if order:
                logging.info(f"{symbol} 平仓: {reason}, 数量 {qty:.2f} 张")
                usdt_balance, long_qty, short_qty, long_avg_price, short_avg_price, total_equity = get_balance(symbol)
                if usdt_balance is None:
                    logging.warning(f"{symbol} 平仓后获取余额失败，跳过开仓")
                    return True

        # 仅当无仓位时开仓
        if long_qty == 0 and short_qty == 0:
            max_quantity = (total_equity * dynamic_buy_ratio) / state[symbol]['current_price'] * params['LEVERAGE']
            ct_val, min_qty, tick_sz, lot_sz = get_symbol_info(symbol)
            min_quantity = min_qty * ct_val  # 转换为币本位
            if state[symbol]['latest_rsi'] <= params['RSI_BUY_VALUE']:
                quantity = min((usdt_balance * dynamic_buy_ratio) / state[symbol]['current_price'] * params['LEVERAGE'], max_quantity)
                if quantity >= min_quantity:
                    order = place_order(symbol, 'buy', 'long', quantity)
                    if order:
                        logging.info(f"{symbol} RSI 开多: 数量 {quantity:.6f} {symbol.split('-')[0]} (约 {(quantity / ct_val):.2f} 张)")
                    else:
                        logging.warning(f"{symbol} RSI 未开多: 下单失败")
                    return True
                else:
                    logging.info(f"{symbol} RSI 未开多: 数量 {quantity:.6f} 小于最小下单单位 {min_quantity:.6f}")
            elif state[symbol]['latest_rsi'] >= params['RSI_SELL_VALUE']:
                quantity = min((usdt_balance * dynamic_buy_ratio) / state[symbol]['current_price'] * params['LEVERAGE'], max_quantity)
                if quantity >= min_quantity:
                    order = place_order(symbol, 'sell', 'short', quantity)
                    if order:
                        logging.info(f"{symbol} RSI 开空: 数量 {quantity:.6f} {symbol.split('-')[0]} (约 {(quantity / ct_val):.2f} 张)")
                    else:
                        logging.warning(f"{symbol} RSI 未开空: 下单失败")
                    return True
                else:
                    logging.info(f"{symbol} RSI 未开空: 数量 {quantity:.6f} 小于最小下单单位 {min_quantity:.6f}")
            elif len(macd) >= 2 and len(signal) >= 2:
                if state[symbol]['latest_macd'] < 0 and state[symbol]['latest_macd'] > state[symbol]['latest_signal'] and macd.iloc[-2] <= signal.iloc[-2] and state[symbol]['latest_histogram'] > 0:
                    logging.info(f"{symbol} MACD 检测到负区金叉")
                    quantity = min((usdt_balance * dynamic_buy_ratio) / state[symbol]['current_price'] * params['LEVERAGE'], max_quantity)
                    if quantity >= min_quantity:
                        order = place_order(symbol, 'buy', 'long', quantity)
                        if order:
                            logging.info(f"{symbol} MACD 开多: 数量 {quantity:.6f} {symbol.split('-')[0]} (约 {(quantity / ct_val):.2f} 张)")
                        else:
                            logging.warning(f"{symbol} MACD 未开多: 下单失败")
                        return True
                    else:
                        logging.info(f"{symbol} MACD 未开多: 数量 {quantity:.6f} 小于最小下单单位 {min_quantity:.6f}")
                elif state[symbol]['latest_macd'] > 0 and state[symbol]['latest_macd'] < state[symbol]['latest_signal'] and macd.iloc[-2] >= signal.iloc[-2] and state[symbol]['latest_histogram'] < 0:
                    logging.info(f"{symbol} MACD 检测到正区死叉")
                    quantity = min((usdt_balance * dynamic_buy_ratio) / state[symbol]['current_price'] * params['LEVERAGE'], max_quantity)
                    if quantity >= min_quantity:
                        order = place_order(symbol, 'sell', 'short', quantity)
                        if order:
                            logging.info(f"{symbol} MACD 开空: 数量 {quantity:.6f} {symbol.split('-')[0]} (约 {(quantity / ct_val):.2f} 张)")
                        else:
                            logging.warning(f"{symbol} MACD 未开空: 下单失败")
                        return True
                    else:
                        logging.info(f"{symbol} MACD 未开空: 数量 {quantity:.6f} 小于最小下单单位 {min_quantity:.6f}")
                else:
                    logging.info(f"{symbol} MACD 未形成金叉或死叉")
            else:
                logging.warning(f"{symbol} MACD 数据长度不足: macd={len(macd)}, signal={len(signal)}")
                return False
        else:
            logging.info(f"{symbol} 未开新仓: 当前持仓 多仓={long_qty:.2f}, 空仓={short_qty:.2f}")
        return True
    except Exception as e:
        logging.error(f"{symbol} 交易逻辑错误: {str(e)}\n{traceback.format_exc()}")
        return False

def trading_cycle():
    try:
        logging.info("开始交易周期")
        for symbol in symbols:
            logging.info(f"处理交易对: {symbol}")
            if execute_trading_logic(symbol):
                continue
            logging.info(f"{symbol} 首次交易逻辑失败，尝试重新执行")
            if execute_trading_logic(symbol):
                continue
            logging.warning(f"{symbol} 第二次交易逻辑失败，跳过本次交易")
    except Exception as e:
        logging.error(f"交易周期错误: {str(e)}\n{traceback.format_exc()}")

def main():
    try:
        # 加载交易参数
        load_params()

        # 加载API配置
        config = load_config()
        api_key = config.get('api_key') or os.getenv('OKX_API_KEY')
        api_secret = config.get('api_secret') or os.getenv('OKX_API_SECRET')
        passphrase = config.get('passphrase') or os.getenv('OKX_PASSPHRASE')
        if not all([api_key, api_secret, passphrase]):
            logging.error("API 密钥未配置")
            raise ValueError("API 密钥未配置")

        # 初始化OKX API
        if init_okx(api_key, api_secret, passphrase, flag='0'):
            save_config(api_key, api_secret, passphrase)
        else:
            logging.error("API 初始化失败")
            raise Exception("API 初始化失败")

        # 为每个交易对设置杠杆
        for symbol in symbols:
            set_leverage(symbol)

        # 执行交易循环
        trading_cycle()
        logging.info("交易周期完成，程序退出")
    except Exception as e:
        logging.error(f"主程序错误: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
