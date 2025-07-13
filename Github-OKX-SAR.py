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
import math
import uuid

# 设置日志
logging.basicConfig(
    handlers=[RotatingFileHandler('trading_bot.log', maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置文件
CONFIG_FILE = 'okx_config.json'
PARAMS_FILE = 'trading_params.json'

# 全局变量
trade_client = None
market_client = None
account_client = None
public_client = None
symbols = ['BTC-USDT-SWAP', 'ETH-USDT-SWAP' , 'SOL-USDT-SWAP', 'ADA-USDT-SWAP', 'XRP-USDT-SWAP']
state = {symbol: {'current_price': 0.0, 'latest_supertrend_1d': None, 'latest_trend_1d': None,
                  'latest_supertrend_1h': None, 'latest_trend_1h': None} for symbol in symbols}
SYMBOL_PARAMS = {}


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

        SYMBOL_PARAMS = {}
        required_keys = [
            'SAT_TIMEFRAME_1D', 'SAT_TIMEFRAME_1H', 'SAT_PERIOD', 'SAT_MULTIPLIER',
            'BUY_RATIO', 'LEVERAGE', 'MARGIN_MODE', 'TAKE_PROFIT', 'STOP_LOSS',
            'ATR_PERIOD', 'ATR_TARGET', 'ATR_BASE', 'BUY_RATIO_MIN', 'BUY_RATIO_MAX',
            'ATR_TP_MULTIPLIER', 'ATR_SL_MULTIPLIER'
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
            for key in ['SAT_MULTIPLIER', 'BUY_RATIO', 'LEVERAGE', 'TAKE_PROFIT', 'STOP_LOSS',
                        'ATR_BASE', 'BUY_RATIO_MIN', 'BUY_RATIO_MAX', 'ATR_TP_MULTIPLIER', 'ATR_SL_MULTIPLIER']:
                if not isinstance(SYMBOL_PARAMS[symbol][key], (int, float)):
                    raise ValueError(f"{symbol} 的 {key} 必须是数值类型")
            for key in ['SAT_PERIOD', 'ATR_PERIOD']:
                if not isinstance(SYMBOL_PARAMS[symbol][key], int):
                    raise ValueError(f"{symbol} 的 {key} 必须是整数")
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
        response = account_client.set_leverage(instId=symbol, lever=str(params['LEVERAGE']),
                                               mgnMode=params['MARGIN_MODE'])
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
            df = pd.DataFrame(klines['data'],
                              columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
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
            logging.info(
                f"K线数据统计: {symbol}, 长度={len(df)}, 收盘价最小={df['close'].min():.2f}, 最大={df['close'].max():.2f}, NaN={df['close'].isna().sum()}")
            logging.info(f"获取K线数据: {symbol}, {len(df)} 条, 最新时间: {df['ts'].iloc[-1]}")
            return df
        except Exception as e:
            logging.error(f"获取K线错误: {symbol} (尝试 {attempt + 1}/5): {str(e)}\n{traceback.format_exc()}")
            time.sleep(2)
    logging.error(f"获取K线失败: {symbol}, 重试次数耗尽")
    return None


def calculate_atr(df, period=7):
    try:
        if len(df) < period + 1:
            logging.warning(f"ATR 数据不足: {len(df)} 条，需至少 {period + 1} 条")
            return None
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        if atr.iloc[-1] is None or np.isnan(atr.iloc[-1]):
            logging.warning("ATR 计算结果无效")
            return None
        logging.info(f"ATR 计算: 周期={period}, 最新ATR={atr.iloc[-1]:.2f}")
        return atr.iloc[-1]
    except Exception as e:
        logging.error(f"ATR 计算错误: {str(e)}\n{traceback.format_exc()}")
        return None


def calculate_supertrend(df, period=7, multiplier=3.0):
    """计算Supertrend指标"""
    try:
        if len(df) < period + 1:
            logging.warning(f"SAT 数据不足: {len(df)} 条，需至少 {period + 1} 条")
            return None, None

        # 计算ATR
        atr = calculate_atr(df, period)
        if atr is None:
            logging.warning("ATR 计算失败，无法计算 Supertrend")
            return None, None

        # 计算中点价格 (HL2)
        hl2 = (df['high'] + df['low']) / 2

        # 初始化Supertrend
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        supertrend = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)  # 1 for uptrend, -1 for downtrend

        # 初始趋势假设为上涨
        supertrend.iloc[0] = lower_band.iloc[0]
        trend.iloc[0] = 1

        for i in range(1, len(df)):
            prev_close = df['close'].iloc[i - 1]
            curr_high = df['high'].iloc[i]
            curr_low = df['low'].iloc[i]
            prev_supertrend = supertrend.iloc[i - 1]
            prev_trend = trend.iloc[i - 1]

            # 更新上下轨
            curr_upper = hl2.iloc[i] + multiplier * atr
            curr_lower = hl2.iloc[i] - multiplier * atr

            if prev_trend == 1:
                # 上涨趋势
                if prev_close > prev_supertrend:
                    supertrend.iloc[i] = curr_lower
                    trend.iloc[i] = 1
                else:
                    supertrend.iloc[i] = curr_upper
                    trend.iloc[i] = -1
            else:
                # 下跌趋势
                if prev_close < prev_supertrend:
                    supertrend.iloc[i] = curr_upper
                    trend.iloc[i] = -1
                else:
                    supertrend.iloc[i] = curr_lower
                    trend.iloc[i] = 1

        # 确保数据有效
        if supertrend.isna().any() or trend.isna().any():
            logging.warning("Supertrend 计算结果包含 NaN")
            return None, None

        logging.info(f"Supertrend 计算: 最新值={supertrend.iloc[-1]:.2f}, 趋势={trend.iloc[-1]}")
        return supertrend, trend
    except Exception as e:
        logging.error(f"Supertrend 计算错误: {str(e)}\n{traceback.format_exc()}")
        return None, None


def get_symbol_info(symbol):
    try:
        info = public_client.get_instruments(instType='SWAP', instId=symbol)
        if info.get('code') != '0':
            raise Exception(f"获取交易对信息失败: {info.get('msg', '未知错误')}")
        ct_val = float(info['data'][0]['ctVal'])
        min_qty = float(info['data'][0]['minSz'])
        tick_sz = float(info['data'][0]['tickSz'])
        lot_sz = float(info['data'][0].get('lotSz', min_qty))
        logging.info(f"{symbol} 合约信息: ct_val={ct_val}, min_qty={min_qty}, tick_sz={tick_sz}, lot_sz={lot_sz}")
        return ct_val, min_qty, tick_sz, lot_sz
    except Exception as e:
        logging.error(f"获取交易对信息失败: {symbol}, {str(e)}\n{traceback.format_exc()}")
        return 0.01, 0.01, 0.01, 0.01


def place_order(symbol, side, pos_side, quantity, atr=None):
    params = SYMBOL_PARAMS[symbol]
    try:
        ct_val, min_qty, tick_sz, lot_sz = get_symbol_info(symbol)
        quantity_in_contracts = quantity / ct_val
        quantity_in_contracts = max(round(quantity_in_contracts / lot_sz) * lot_sz, min_qty)
        if quantity_in_contracts < min_qty:
            logging.warning(f"下单失败: {symbol}, 数量 {quantity_in_contracts:.2f} 张小于最小值 {min_qty:.2f} 张")
            return None

        current_price = state[symbol]['current_price']
        if atr is None:
            logging.warning(f"{symbol} 未提供 ATR 值，使用固定百分比止盈止损")
            tp_price = round(current_price * (
                1 + params['TAKE_PROFIT'] / 100 if pos_side == 'long' else 1 - params['TAKE_PROFIT'] / 100),
                             -int(np.log10(tick_sz)))
            sl_price = round(current_price * (
                1 - params['STOP_LOSS'] / 100 if pos_side == 'long' else 1 + params['STOP_LOSS'] / 100),
                             -int(np.log10(tick_sz)))
        else:
            tp_price = round(
                current_price + atr * params['ATR_TP_MULTIPLIER'] if pos_side == 'long' else current_price - atr *
                                                                                             params[
                                                                                                 'ATR_TP_MULTIPLIER'],
                -int(np.log10(tick_sz)))
            sl_price = round(
                current_price - atr * params['ATR_SL_MULTIPLIER'] if pos_side == 'long' else current_price + atr *
                                                                                             params[
                                                                                                 'ATR_SL_MULTIPLIER'],
                -int(np.log10(tick_sz)))
            logging.info(
                f"{symbol} ATR 计算: 当前价格={current_price:.2f}, ATR={atr:.2f}, 止盈倍数={params['ATR_TP_MULTIPLIER']}, 止损倍数={params['ATR_SL_MULTIPLIER']}, 止盈价格={tp_price:.2f}, 止损价格={sl_price:.2f}")

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
        logging.info(
            f"{symbol} 准备下单: 方向={side}, 持仓方向={pos_side}, 数量={quantity_in_contracts:.2f} 张 (约 {quantity_in_contracts * ct_val:.6f} {symbol.split('-')[0]})")
        order = trade_client.place_order(**order_params)
        if order['code'] == '0':
            action = '开多' if side == 'buy' and pos_side == 'long' else '开空' if side == 'sell' and pos_side == 'short' else '平仓'
            logging.info(
                f"{symbol} {action} 订单已下: 数量 {quantity_in_contracts:.2f} 张, 止盈价格={tp_price:.2f}, 止损价格={sl_price:.2f}")
            return order['data'][0]['ordId']
        else:
            logging.error(
                f"下单失败: {symbol}, 错误码={order['code']}, 错误信息={order['msg']}, 详情={order.get('data', '无详细信息')}")
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
            logging.debug(f"{symbol} 账户余额原始数据: {balance.get('data', '无数据')}")
            if not balance.get('data') or not balance['data'][0].get('details'):
                logging.warning(f"{symbol} 账户余额数据为空或无 details 字段")
                usdt = 0.0
            else:
                usdt_asset = next((asset for asset in balance['data'][0]['details'] if asset['ccy'] == 'USDT'),
                                  {'availEq': '0'})
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
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def check_take_profit_stop_loss(symbol, long_qty, short_qty, long_avg_price, short_avg_price, current_price, atr=None):
    params = SYMBOL_PARAMS[symbol]
    try:
        pos_side = None
        qty = 0.0
        reason = None

        # 获取1H周期Supertrend用于平仓信号
        df_1h = get_klines(symbol, params['SAT_TIMEFRAME_1H'])
        if df_1h is None:
            logging.warning(f"{symbol} 无1H K线数据，跳过平仓检查")
            return None, 0.0, None

        supertrend_1h, trend_1h = calculate_supertrend(df_1h, params['SAT_PERIOD'], params['SAT_MULTIPLIER'])
        if supertrend_1h is None or trend_1h is None:
            logging.warning(f"{symbol} 1H Supertrend 计算失败，跳过平仓检查")
            return None, 0.0, None

        # 检查持仓并判断平仓条件
        if long_qty > 0 and long_avg_price > 0:
            profit_diff = current_price - long_avg_price
            if atr is not None and profit_diff >= atr * params['ATR_TP_MULTIPLIER']:
                pos_side = 'long'
                qty = long_qty
                reason = f"ATR 止盈（{params['ATR_TP_MULTIPLIER']} 倍 ATR）"
            elif atr is None and (profit_diff / long_avg_price * 100) >= params['TAKE_PROFIT']:
                pos_side = 'long'
                qty = long_qty
                reason = '固定百分比止盈'
            elif atr is not None and (long_avg_price - current_price) >= atr * params['ATR_SL_MULTIPLIER']:
                pos_side = 'long'
                qty = long_qty
                reason = f"ATR 止损（{params['ATR_SL_MULTIPLIER']} 倍 ATR）"
            elif atr is None and (profit_diff / long_avg_price * 100) <= -params['STOP_LOSS']:
                pos_side = 'long'
                qty = long_qty
                reason = '固定百分比止损'
            elif len(trend_1h) >= 2 and all(trend_1h.iloc[-2:] == -1):
                pos_side = 'long'
                qty = long_qty
                reason = '1H Supertrend 连续下跌信号'
        elif short_qty > 0 and short_avg_price > 0:
            profit_diff = short_avg_price - current_price
            if atr is not None and profit_diff >= atr * params['ATR_TP_MULTIPLIER']:
                pos_side = 'short'
                qty = short_qty
                reason = f"ATR 止盈（{params['ATR_TP_MULTIPLIER']} 倍 ATR）"
            elif atr is None and (profit_diff / short_avg_price * 100) >= params['TAKE_PROFIT']:
                pos_side = 'short'
                qty = short_qty
                reason = '固定百分比止盈'
            elif atr is not None and (current_price - short_avg_price) >= atr * params['ATR_SL_MULTIPLIER']:
                pos_side = 'short'
                qty = short_qty
                reason = f"ATR 止损（{params['ATR_SL_MULTIPLIER']} 倍 ATR）"
            elif atr is None and (profit_diff / short_avg_price * 100) <= -params['STOP_LOSS']:
                pos_side = 'short'
                qty = short_qty
                reason = '固定百分比止损'
            elif len(trend_1h) >= 2 and all(trend_1h.iloc[-2:] == 1):
                pos_side = 'short'
                qty = short_qty
                reason = '1H Supertrend 连续上涨信号'

        return pos_side, qty, reason
    except Exception as e:
        logging.error(f"检查止盈止损错误: {symbol}, {str(e)}\n{traceback.format_exc()}")
        return None, 0.0, None


def execute_trading_logic(symbol):
    """执行数据获取、计算和交易逻辑"""
    params = SYMBOL_PARAMS[symbol]
    try:
        # 获取日线和1H周期的K线数据
        df_1d = get_klines(symbol, params['SAT_TIMEFRAME_1D'])
        if df_1d is None:
            logging.warning(f"{symbol} 无日线 K线数据，跳过交易")
            return False

        df_1h = get_klines(symbol, params['SAT_TIMEFRAME_1H'])
        if df_1h is None:
            logging.warning(f"{symbol} 无1H K线数据，跳过交易")
            return False

        # 计算日线和1H的Supertrend
        supertrend_1d, trend_1d = calculate_supertrend(df_1d, params['SAT_PERIOD'], params['SAT_MULTIPLIER'])
        if supertrend_1d is None or trend_1d is None:
            logging.warning(f"{symbol} 日线 Supertrend 计算失败，跳过交易")
            return False

        supertrend_1h, trend_1h = calculate_supertrend(df_1h, params['SAT_PERIOD'], params['SAT_MULTIPLIER'])
        if supertrend_1h is None or trend_1h is None:
            logging.warning(f"{symbol} 1H Supertrend 计算失败，跳过交易")
            return False

        # 更新状态
        state[symbol]['latest_supertrend_1d'] = supertrend_1d.iloc[-1]
        state[symbol]['latest_trend_1d'] = trend_1d.iloc[-1]
        state[symbol]['latest_supertrend_1h'] = supertrend_1h.iloc[-1]
        state[symbol]['latest_trend_1h'] = trend_1h.iloc[-1]

        # 计算ATR并调整仓位比例
        atr = calculate_atr(df_1h, period=params['ATR_PERIOD'])
        if atr is None:
            logging.warning(f"{symbol} ATR计算失败，使用默认仓位比例")
            dynamic_buy_ratio = params['BUY_RATIO']
        else:
            diff = atr - params['ATR_TARGET']
            volatility_factor = math.exp(-(diff ** 2) / (2 * params['ATR_BASE'] ** 2))
            volatility_factor = max(0.5, min(1.5, volatility_factor * 1.5))
            dynamic_buy_ratio = params['BUY_RATIO'] * volatility_factor
            dynamic_buy_ratio = max(params['BUY_RATIO_MIN'], min(params['BUY_RATIO_MAX'], dynamic_buy_ratio))
            logging.info(
                f"{symbol} 动态仓位比例: ATR={atr:.2f}, 目标ATR={params['ATR_TARGET']:.2f}, 基准ATR={params['ATR_BASE']:.2f}, volatility_factor={volatility_factor:.2f}, dynamic_buy_ratio={dynamic_buy_ratio:.2f}")

        # 获取账户余额和持仓
        usdt_balance, long_qty, short_qty, long_avg_price, short_avg_price, total_equity = get_balance(symbol)
        if usdt_balance < 0:
            logging.warning(f"{symbol} USDT余额无效: {usdt_balance:.2f}")
            return False
        if usdt_balance == 0 and total_equity == 0:
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

        logging.info(
            f"{symbol} 账户状态: USDT余额={usdt_balance:.2f}, 总权益={total_equity:.2f}, 多仓={long_qty:.2f}, 空仓={short_qty:.2f}")

        # 检查止盈止损
        pos_side, qty, reason = check_take_profit_stop_loss(symbol, long_qty, short_qty, long_avg_price,
                                                            short_avg_price, state[symbol]['current_price'], atr=atr)
        if pos_side and qty > 0:
            order = place_order(symbol, 'sell' if pos_side == 'long' else 'buy', pos_side, qty, atr=atr)
            if order:
                logging.info(f"{symbol} 平仓: {reason}, 数量 {qty:.2f} 张")
                usdt_balance, long_qty, short_qty, long_avg_price, short_avg_price, total_equity = get_balance(symbol)
                if usdt_balance is None:
                    logging.warning(f"{symbol} 平仓后获取余额失败，跳过开仓")
                    return True

        # 检查日线趋势：连续两个K线确认趋势
        if len(trend_1d) >= 2:
            last_two_trends_1d = trend_1d.iloc[-2:]
            if all(last_two_trends_1d == 1):
                trend_direction = 'long'
                logging.info(f"{symbol} 日线趋势: 连续两个上涨信号")
            elif all(last_two_trends_1d == -1):
                trend_direction = 'short'
                logging.info(f"{symbol} 日线趋势: 连续两个下跌信号")
            else:
                trend_direction = None
                logging.info(f"{symbol} 日线趋势: 未形成连续信号，跳过开仓")
        else:
            trend_direction = None
            logging.warning(f"{symbol} 日线数据不足，无法判断趋势")

        # 仅当无仓位且日线趋势明确时检查1H开仓信号
        if long_qty == 0 and short_qty == 0 and trend_direction:
            max_quantity = (total_equity * dynamic_buy_ratio) / state[symbol]['current_price'] * params['LEVERAGE']
            ct_val, min_qty, tick_sz, lot_sz = get_symbol_info(symbol)
            min_quantity = min_qty * ct_val

            if len(trend_1h) >= 2:
                last_two_trends_1h = trend_1h.iloc[-2:]
                if trend_direction == 'long' and all(last_two_trends_1h == 1):
                    quantity = min(
                        (usdt_balance * dynamic_buy_ratio) / state[symbol]['current_price'] * params['LEVERAGE'],
                        max_quantity)
                    if quantity >= min_quantity:
                        order = place_order(symbol, 'buy', 'long', quantity, atr=atr)
                        if order:
                            logging.info(
                                f"{symbol} SAT 开多: 数量 {quantity:.6f} {symbol.split('-')[0]} (约 {(quantity / ct_val):.2f} 张)")
                        else:
                            logging.warning(f"{symbol} SAT 未开多: 下单失败")
                        return True
                    else:
                        logging.info(f"{symbol} SAT 未开多: 数量 {quantity:.6f} 小于最小下单单位 {min_quantity:.6f}")
                elif trend_direction == 'short' and all(last_two_trends_1h == -1):
                    quantity = min(
                        (usdt_balance * dynamic_buy_ratio) / state[symbol]['current_price'] * params['LEVERAGE'],
                        max_quantity)
                    if quantity >= min_quantity:
                        order = place_order(symbol, 'sell', 'short', quantity, atr=atr)
                        if order:
                            logging.info(
                                f"{symbol} SAT 开空: 数量 {quantity:.6f} {symbol.split('-')[0]} (约 {(quantity / ct_val):.2f} 张)")
                        else:
                            logging.warning(f"{symbol} SAT 未开空: 下单失败")
                        return True
                    else:
                        logging.info(f"{symbol} SAT 未开空: 数量 {quantity:.6f} 小于最小下单单位 {min_quantity:.6f}")
                else:
                    logging.info(f"{symbol} 1H周期未形成连续信号或与日线趋势不一致")
            else:
                logging.warning(f"{symbol} 1H数据不足，无法判断开仓信号")
        else:
            logging.info(
                f"{symbol} 未开新仓: 当前持仓 多仓={long_qty:.2f}, 空仓={short_qty:.2f}, 日线趋势={trend_direction}")
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
        load_params()
        config = load_config()
        api_key = config.get('api_key') or os.getenv('OKX_API_KEY')
        api_secret = config.get('api_secret') or os.getenv('OKX_API_SECRET')
        passphrase = config.get('passphrase') or os.getenv('OKX_PASSPHRASE')
        if not all([api_key, api_secret, passphrase]):
            logging.error("API 密钥未配置")
            raise ValueError("API 密钥未配置")
        if init_okx(api_key, api_secret, passphrase, flag='0'):
            save_config(api_key, api_secret, passphrase)
        else:
            logging.error("API 初始化失败")
            raise Exception("API 初始化失败")
        for symbol in symbols:
            set_leverage(symbol)
        trading_cycle()
        logging.info("交易周期完成，程序退出")
    except Exception as e:
        logging.error(f"主程序错误: {str(e)}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
