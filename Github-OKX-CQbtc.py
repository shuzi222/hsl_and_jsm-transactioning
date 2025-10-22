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
SYMBOL_PARAMS = {}

def load_params():
    """加载精简交易参数"""
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
        # 🔥 精简必需字段：13个核心参数
        required_keys = [
            'RSI_TIMEFRAME', 'MACD_TIMEFRAME', 'RSI_BUY_VALUE', 'RSI_SELL_VALUE',
            'BUY_RATIO', 'LEVERAGE', 'MARGIN_MODE', 
            'TAKE_PROFIT', 'STOP_LOSS', 'MACD_TAKE_PROFIT', 'MACD_STOP_LOSS'
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

            # 参数类型验证
            numeric_keys = ['RSI_BUY_VALUE', 'RSI_SELL_VALUE', 'BUY_RATIO', 'LEVERAGE', 
                           'TAKE_PROFIT', 'STOP_LOSS', 'MACD_TAKE_PROFIT', 'MACD_STOP_LOSS']
            for key in numeric_keys:
                if not isinstance(SYMBOL_PARAMS[symbol][key], (int, float)):
                    raise ValueError(f"{symbol} 的 {key} 必须是数值类型")
            if SYMBOL_PARAMS[symbol]['MARGIN_MODE'] not in ['cross', 'isolated']:
                raise ValueError(f"{symbol} 的 MARGIN_MODE 必须是 'cross' 或 'isolated'")

        logging.info("✅ 精简交易参数加载成功 (13个核心参数)")
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
            logging.info(f"获取K线数据: {symbol}, {len(df)} 条, 最新时间: {df['ts'].iloc[-1]}")
            return df
        except Exception as e:
            logging.error(f"获取K线错误: {symbol} (尝试 {attempt + 1}/5): {str(e)}\n{traceback.format_exc()}")
            time.sleep(2)
    logging.error(f"获取K线失败: {symbol}, 重试次数耗尽")
    return None

def calculate_rsi(df, period=14, current_price=None):
    """RSI计算，支持实时价格替换最后一根K线收盘价"""
    try:
        if len(df) < period + 1:
            logging.warning(f"RSI 数据不足: {len(df)} 条")
            return None
        
        if current_price is not None:
            df = df.copy()
            df.iloc[-1, df.columns.get_loc('close')] = current_price
            logging.info(f"RSI 使用实时价格更新: {current_price:.2f}")

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
        logging.info(f"RSI 计算: 最新RSI={rsi[-1]:.2f} {'[实时价格]' if current_price is not None else '[整点收盘]'}")
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
        if np.any(np.isnan(close)) or np.any(close <= 0):
            logging.warning("K线数据包含无效收盘价")
            return None, None, None
        ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
        ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
        macd = ema_fast - ema_slow
        signal_line = pd.Series(macd).ewm(span=signal, adjust=False).mean().values
        histogram = macd - signal_line
        macd_series = pd.Series(macd, index=df.index)
        signal_series = pd.Series(signal_line, index=df.index)
        histogram_series = pd.Series(histogram, index=df.index)
        logging.info(f"MACD 计算: MACD={macd[-1]:.4f}, 信号线={signal_line[-1]:.4f}, 柱状图={histogram[-1]:.4f}")
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
        lot_sz = float(info['data'][0].get('lotSz', min_qty))
        return ct_val, min_qty, tick_sz, lot_sz
    except Exception as e:
        logging.error(f"获取交易对信息失败: {symbol}, {str(e)}")
        return 0.01, 0.01, 0.01, 0.01

def place_order(symbol, side, pos_side, quantity, signal_type="RSI"):
    """支持RSI/MACD独立止盈止损"""
    params = SYMBOL_PARAMS[symbol]
    try:
        ct_val, min_qty, tick_sz, lot_sz = get_symbol_info(symbol)
        quantity_in_contracts = quantity / ct_val
        quantity_in_contracts = max(round(quantity_in_contracts / lot_sz) * lot_sz, min_qty)
        if quantity_in_contracts < min_qty:
            logging.warning(f"下单失败: {symbol}, 数量 {quantity_in_contracts:.2f} < {min_qty:.2f}")
            return None

        current_price = state[symbol]['current_price']
        
        # 根据信号类型选择止盈止损
        if signal_type.upper() == "MACD":
            tp_pct = params['MACD_TAKE_PROFIT']
            sl_pct = params['MACD_STOP_LOSS']
        else:
            tp_pct = params['TAKE_PROFIT']
            sl_pct = params['STOP_LOSS']

        if pos_side == 'long':
            tp_price = round(current_price * (1 + tp_pct / 100), -int(np.log10(tick_sz)))
            sl_price = round(current_price * (1 - sl_pct / 100), -int(np.log10(tick_sz)))
        else:
            tp_price = round(current_price * (1 - tp_pct / 100), -int(np.log10(tick_sz)))
            sl_price = round(current_price * (1 + sl_pct / 100), -int(np.log10(tick_sz)))

        algo_order = {
            'tpTriggerPx': str(tp_price), 'tpOrdPx': '-1',
            'slTriggerPx': str(sl_price), 'slOrdPx': '-1',
            'tpOrdKind': 'condition', 'slTriggerPxType': 'last', 'tpTriggerPxType': 'last'
        }
        order_params = {
            'instId': symbol, 'tdMode': params['MARGIN_MODE'],
            'side': side.lower(), 'posSide': pos_side.lower(),
            'ordType': 'market', 'sz': str(round(quantity_in_contracts, 2)),
            'clOrdId': str(uuid.uuid4()).replace('-', '')[:32],
            'attachAlgoOrds': [algo_order]
        }
        
        order = trade_client.place_order(**order_params)
        if order['code'] == '0':
            action = '开多' if side == 'buy' and pos_side == 'long' else '开空' if side == 'sell' and pos_side == 'short' else '平仓'
            logging.info(f"{symbol} {action}: {quantity_in_contracts:.2f} 张, TP={tp_price:.2f}, SL={sl_price:.2f} [{signal_type}]")
            return order['data'][0]['ordId']
        else:
            logging.error(f"下单失败: {symbol}, {order['msg']}")
            return None
    except Exception as e:
        logging.error(f"下单失败: {symbol}, {str(e)}")
        return None

def get_balance(symbol):
    for attempt in range(3):
        try:
            balance = account_client.get_account_balance()
            if balance.get('code') != '0':
                raise Exception(f"获取余额失败: {balance.get('msg')}")
            
            if not balance.get('data') or not balance['data'][0].get('details'):
                usdt = 0.0
            else:
                usdt_asset = next((asset for asset in balance['data'][0]['details'] if asset['ccy'] == 'USDT'), {'availEq': '0'})
                usdt = float(usdt_asset['availEq']) if usdt_asset['availEq'] else 0.0
            total_equity = float(balance['data'][0]['totalEq']) if balance['data'][0].get('totalEq') else 0.0

            positions = account_client.get_positions(instType='SWAP', instId=symbol)
            if positions.get('code') != '0':
                raise Exception(f"获取持仓失败: {positions.get('msg')}")
            
            long_qty = short_qty = long_avg_price = short_avg_price = 0.0
            if positions.get('data'):
                for pos in positions['data']:
                    if pos['instId'] == symbol:
                        qty = float(pos['pos']) if pos['pos'] else 0.0
                        avg_price = float(pos['avgPx']) if pos['avgPx'] else 0.0
                        if pos['posSide'] == 'long':
                            long_qty, long_avg_price = qty, avg_price
                        elif pos['posSide'] == 'short':
                            short_qty, short_avg_price = qty, avg_price
            
            return usdt, long_qty, short_qty, long_avg_price, short_avg_price, total_equity
        except Exception as e:
            logging.error(f"获取余额失败: {symbol} (尝试 {attempt + 1}/3): {str(e)}")
            time.sleep(2)
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def check_take_profit_stop_loss(symbol, long_qty, short_qty, long_avg_price, short_avg_price, current_price, rsi, macd, signal):
    params = SYMBOL_PARAMS[symbol]
    try:
        pos_side = None
        qty = 0.0
        reason = None
        
        if long_qty > 0 and long_avg_price > 0:
            profit_diff = (current_price - long_avg_price) / long_avg_price * 100
            if profit_diff >= params['TAKE_PROFIT']:
                return 'long', long_qty, 'RSI止盈'
            elif profit_diff <= -params['STOP_LOSS']:
                return 'long', long_qty, 'RSI止损'
            elif rsi >= params['RSI_SELL_VALUE']:
                return 'long', long_qty, f'RSI>{params["RSI_SELL_VALUE"]}'
            elif (macd.iloc[-1] > 0 and macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]):
                return 'long', long_qty, 'MACD死叉'

        if short_qty > 0 and short_avg_price > 0:
            profit_diff = (short_avg_price - current_price) / short_avg_price * 100
            if profit_diff >= params['TAKE_PROFIT']:
                return 'short', short_qty, 'RSI止盈'
            elif profit_diff <= -params['STOP_LOSS']:
                return 'short', short_qty, 'RSI止损'
            elif rsi <= params['RSI_BUY_VALUE']:
                return 'short', short_qty, f'RSI<{params["RSI_BUY_VALUE"]}'
            elif (macd.iloc[-1] < 0 and macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]):
                return 'short', short_qty, 'MACD金叉'
                
        return None, 0.0, None
    except Exception as e:
        logging.error(f"检查止盈止损错误: {symbol}, {str(e)}")
        return None, 0.0, None

def execute_trading_logic(symbol):
    params = SYMBOL_PARAMS[symbol]
    try:
        # 获取K线数据
        df_rsi = get_klines(symbol, params['RSI_TIMEFRAME'])
        if df_rsi is None: return False
        
        df_macd = get_klines(symbol, params['MACD_TIMEFRAME'])
        if df_macd is None: return False

        # 获取实时价格并计算指标
        price = get_price(symbol)
        if not price: return False

        rsi = calculate_rsi(df_rsi, current_price=price)
        if rsi is None: return False
        state[symbol]['latest_rsi'] = rsi.iloc[-1]

        macd, signal_line, histogram = calculate_macd(df_macd)
        if macd is None: return False
        state[symbol]['latest_macd'] = macd.iloc[-1]
        state[symbol]['latest_signal'] = signal_line.iloc[-1]
        state[symbol]['latest_histogram'] = histogram.iloc[-1]

        # 获取账户状态
        usdt_balance, long_qty, short_qty, long_avg_price, short_avg_price, total_equity = get_balance(symbol)
        if usdt_balance <= 0 or total_equity <= 0: return False

        logging.info(f"{symbol} 状态: USDT={usdt_balance:.2f}, 多={long_qty:.2f}, 空={short_qty:.2f}, RSI={rsi.iloc[-1]:.1f}")

        # 检查平仓
        pos_side, qty, reason = check_take_profit_stop_loss(
            symbol, long_qty, short_qty, long_avg_price, short_avg_price, 
            price, rsi.iloc[-1], macd, signal_line
        )
        if pos_side and qty > 0:
            order = place_order(symbol, 'sell' if pos_side == 'long' else 'buy', pos_side, qty)
            if order:
                logging.info(f"{symbol} 平仓成功: {reason}")
                return True

        # 开仓 (仅无仓位时)
        if long_qty == 0 and short_qty == 0:
            quantity = (usdt_balance * params['BUY_RATIO'] / price) * params['LEVERAGE']
            ct_val, min_qty, _, lot_sz = get_symbol_info(symbol)
            min_quantity = min_qty * ct_val
            
            if quantity < min_quantity:
                logging.info(f"{symbol} 仓位不足最小单位: {quantity:.6f} < {min_quantity:.6f}")
                return True

            # RSI开仓
            if rsi.iloc[-1] <= params['RSI_BUY_VALUE']:
                order = place_order(symbol, 'buy', 'long', quantity, "RSI")
                if order: logging.info(f"{symbol} ✅ RSI开多成功")
                return True
            elif rsi.iloc[-1] >= params['RSI_SELL_VALUE']:
                order = place_order(symbol, 'sell', 'short', quantity, "RSI")
                if order: logging.info(f"{symbol} ✅ RSI开空成功")
                return True

            # MACD开仓
            elif len(macd) >= 2 and len(signal_line) >= 2:
                # 金叉开多
                if (macd.iloc[-1] < 0 and macd.iloc[-1] > signal_line.iloc[-1] and 
                    macd.iloc[-2] <= signal_line.iloc[-2] and histogram.iloc[-1] > 0):
                    order = place_order(symbol, 'buy', 'long', quantity, "MACD")
                    if order: logging.info(f"{symbol} ✅ MACD金叉开多")
                    return True
                
                # 死叉开空
                elif (macd.iloc[-1] > 0 and macd.iloc[-1] < signal_line.iloc[-1] and 
                      macd.iloc[-2] >= signal_line.iloc[-2] and histogram.iloc[-1] < 0):
                    order = place_order(symbol, 'sell', 'short', quantity, "MACD")
                    if order: logging.info(f"{symbol} ✅ MACD死叉开空")
                    return True

        logging.info(f"{symbol} 无开仓信号")
        return True
    except Exception as e:
        logging.error(f"{symbol} 交易逻辑错误: {str(e)}")
        return False

def trading_cycle():
    logging.info("🚀 开始交易周期")
    for symbol in symbols:
        logging.info(f"处理: {symbol}")
        success = execute_trading_logic(symbol)
        if not success:
            logging.warning(f"{symbol} 交易失败，重试...")
            execute_trading_logic(symbol)
    logging.info("✅ 交易周期完成")

def main():
    try:
        load_params()
        config = load_config()
        api_key = config.get('api_key') or os.getenv('OKX_API_KEY')
        api_secret = config.get('api_secret') or os.getenv('OKX_API_SECRET')
        passphrase = config.get('passphrase') or os.getenv('OKX_PASSPHRASE')
        
        if not all([api_key, api_secret, passphrase]):
            raise ValueError("API密钥未配置")

        if init_okx(api_key, api_secret, passphrase, flag='0'):
            save_config(api_key, api_secret, passphrase)
        
        for symbol in symbols:
            set_leverage(symbol)
        
        trading_cycle()
        logging.info("程序退出")
    except Exception as e:
        logging.error(f"主程序错误: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
