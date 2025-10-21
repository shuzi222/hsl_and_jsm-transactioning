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
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(
    handlers=[RotatingFileHandler('trading_bot.log', maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置文件
PARAMS_FILE = '参数.json'

# 全局变量
trade_client = None
market_client = None
account_client = None
public_client = None
symbols = ['BTC-USDT-SWAP']
SYMBOL_PARAMS = {}


def load_params():
    """加载交易参数"""
    global SYMBOL_PARAMS
    print("📁 加载交易参数...")

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
            'RSI_TIMEFRAME', 'RSI_BUY_VALUE', 'RSI_SELL_VALUE',
            'BUY_RATIO', 'LEVERAGE', 'MARGIN_MODE', 'TAKE_PROFIT'
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
            for key in ['RSI_BUY_VALUE', 'RSI_SELL_VALUE', 'BUY_RATIO', 'LEVERAGE', 'TAKE_PROFIT']:
                if not isinstance(SYMBOL_PARAMS[symbol][key], (int, float)):
                    raise ValueError(f"{symbol} 的 {key} 必须是数值类型")
            if SYMBOL_PARAMS[symbol]['MARGIN_MODE'] not in ['cross', 'isolated']:
                raise ValueError(f"{symbol} 的 MARGIN_MODE 必须是 'cross' 或 'isolated'")

        print("✅ 交易参数加载成功")
        for symbol in symbols:
            logging.info(f"{symbol} 参数: {SYMBOL_PARAMS[symbol]}")
        return True
    except Exception as e:
        logging.error(f"加载交易参数失败: {str(e)}")
        raise ValueError(f"加载交易参数失败: {str(e)}")


def get_api_credentials():
    """安全获取API密钥（GitHub Actions优先）"""
    api_key = os.getenv('OKX_API_KEY')
    api_secret = os.getenv('OKX_API_SECRET')
    passphrase = os.getenv('OKX_PASSPHRASE')

    if not all([api_key, api_secret, passphrase]):
        missing = []
        if not api_key: missing.append('OKX_API_KEY')
        if not api_secret: missing.append('OKX_API_SECRET')
        if not passphrase: missing.append('OKX_PASSPHRASE')

        error_msg = f"缺少环境变量: {', '.join(missing)}"
        print(f"❌ {error_msg}")
        print("💡 请在 GitHub Settings > Secrets and variables > Actions 添加密钥")
        raise ValueError(error_msg)

    print("✅ API密钥加载成功")
    return api_key, api_secret, passphrase


def init_okx(api_key, api_secret, passphrase, flag='0'):
    global trade_client, market_client, account_client, public_client
    try:
        print("🔌 初始化 OKX API...")
        trade_client = Trade.TradeAPI(api_key, api_secret, passphrase, use_server_time=False, flag=flag)
        market_client = MarketData.MarketAPI(flag=flag)
        account_client = Account.AccountAPI(api_key, api_secret, passphrase, use_server_time=False, flag=flag)
        public_client = PublicData.PublicAPI(flag=flag)

        response = account_client.get_account_balance()
        if response.get('code') != '0':
            raise Exception(f"账户余额检查失败: {response.get('msg', '未知错误')}")

        print(f"✅ OKX API 初始化成功，flag={flag}")
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
            print(f"🔧 设置杠杆成功: {symbol}, 杠杆={params['LEVERAGE']}, 模式={params['MARGIN_MODE']}")
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
            if df.empty or len(df) < 20:
                logging.warning(f"K线数据不足: {symbol}, {len(df)} 条")
                return None
            df = df.dropna(subset=['close'])
            df = df[df['close'] > 0]
            logging.info(f"📈 获取K线数据: {symbol}, {len(df)} 条, 周期: {interval}")
            return df
        except Exception as e:
            logging.error(f"获取K线错误: {symbol} (尝试 {attempt + 1}/5): {str(e)}")
            time.sleep(2)
    logging.error(f"获取K线失败: {symbol}")
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
        latest_rsi = rsi[-1]
        if np.isnan(latest_rsi):
            return None
        print(f"📊 RSI 计算: 最新RSI={latest_rsi:.2f}")
        return latest_rsi
    except Exception as e:
        logging.error(f"RSI 计算错误: {str(e)}")
        return None


def get_symbol_info(symbol):
    try:
        info = public_client.get_instruments(instType='SWAP', instId=symbol)
        if info.get('code') != '0':
            raise Exception(f"获取交易对信息失败: {info.get('msg')}")
        ct_val = float(info['data'][0]['ctVal'])
        min_qty = float(info['data'][0]['minSz'])
        tick_sz = float(info['data'][0]['tickSz'])
        lot_sz = float(info['data'][0].get('lotSz', min_qty))
        return ct_val, min_qty, tick_sz, lot_sz
    except Exception as e:
        logging.error(f"获取交易对信息失败: {symbol}, {str(e)}")
        return 0.01, 0.01, 0.01, 0.01


def place_order(symbol, side, pos_side, quantity):
    params = SYMBOL_PARAMS[symbol]
    try:
        ct_val, min_qty, tick_sz, lot_sz = get_symbol_info(symbol)
        quantity_in_contracts = quantity / ct_val
        quantity_in_contracts = max(round(quantity_in_contracts / lot_sz) * lot_sz, min_qty)

        if quantity_in_contracts < min_qty:
            logging.warning(f"下单失败: {symbol}, 数量 {quantity_in_contracts:.2f} 张 < 最小值 {min_qty}")
            return None

        current_price = get_price(symbol)
        if not current_price:
            return None

        if pos_side == 'long':
            tp_price = round(current_price * (1 + params['TAKE_PROFIT'] / 100), -int(np.log10(tick_sz)))
        else:
            tp_price = round(current_price * (1 - params['TAKE_PROFIT'] / 100), -int(np.log10(tick_sz)))

        algo_order = {
            'tpTriggerPx': str(tp_price),
            'tpOrdPx': '-1',
            'tpOrdKind': 'condition',
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

        print(f"📝 {symbol} 下单: {side} {pos_side}, 数量={quantity:.6f} {symbol.split('-')[0]}")
        order = trade_client.place_order(**order_params)

        if order['code'] == '0':
            action = '开多' if side == 'buy' else '开空'
            print(f"✅ {symbol} {action}成功: 数量 {quantity:.6f}, 止盈价格={tp_price:.2f}")
            return order['data'][0]['ordId']
        else:
            logging.error(f"下单失败: {symbol}, 错误={order['msg']}")
            return None
    except Exception as e:
        logging.error(f"下单异常: {symbol}, {str(e)}")
        return None


def get_balance(symbol):
    for attempt in range(3):
        try:
            balance = account_client.get_account_balance()
            if balance.get('code') != '0':
                raise Exception(f"获取余额失败: {balance.get('msg')}")

            usdt_asset = next((asset for asset in balance['data'][0]['details'] if asset['ccy'] == 'USDT'),
                              {'availEq': '0'})
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

            print(f"💰 {symbol} 余额: USDT={usdt:.2f}, 多仓={long_qty:.2f}, 空仓={short_qty:.2f}")
            return usdt, long_qty, short_qty, long_avg_price, short_avg_price, total_equity
        except Exception as e:
            logging.error(f"获取余额失败: {symbol} (尝试 {attempt + 1}/3): {str(e)}")
            time.sleep(2)
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def check_take_profit(symbol, long_qty, short_qty, long_avg_price, short_avg_price):
    params = SYMBOL_PARAMS[symbol]
    current_price = get_price(symbol)
    if not current_price:
        return None, 0.0, None

    if long_qty > 0 and long_avg_price > 0:
        profit_pct = (current_price - long_avg_price) / long_avg_price * 100
        if profit_pct >= params['TAKE_PROFIT']:
            return 'long', long_qty, f"止盈 {profit_pct:.2f}%"

    if short_qty > 0 and short_avg_price > 0:
        profit_pct = (short_avg_price - current_price) / short_avg_price * 100
        if profit_pct >= params['TAKE_PROFIT']:
            return 'short', short_qty, f"止盈 {profit_pct:.2f}%"

    return None, 0.0, None


def execute_trading_logic(symbol):
    params = SYMBOL_PARAMS[symbol]

    # 1. 获取K线数据
    df = get_klines(symbol, params['RSI_TIMEFRAME'])
    if df is None:
        print(f"⚠️  {symbol} 无K线数据，跳过")
        return False

    # 2. 计算RSI
    rsi = calculate_rsi(df)
    if rsi is None:
        print(f"⚠️  {symbol} RSI计算失败，跳过")
        return False

    # 3. 获取余额和持仓
    usdt_balance, long_qty, short_qty, long_avg_price, short_avg_price, total_equity = get_balance(symbol)
    if usdt_balance <= 0 and total_equity <= 0:
        print(f"⚠️  {symbol} 无可用资金")
        return False

    current_price = get_price(symbol)
    if not current_price:
        return False

    # 4. 检查平仓（止盈）
    pos_side, qty, reason = check_take_profit(symbol, long_qty, short_qty, long_avg_price, short_avg_price)
    if pos_side and qty > 0:
        order = place_order(symbol, 'sell' if pos_side == 'long' else 'buy', pos_side, qty)
        if order:
            print(f"✅ {symbol} 平仓成功: {reason}")
            time.sleep(1)
            _, long_qty, short_qty, _, _, _ = get_balance(symbol)
        else:
            print(f"⚠️  {symbol} 平仓失败: {reason}")

    # 5. 开仓逻辑（仅当无仓位时）
    if long_qty == 0 and short_qty == 0:
        ct_val, min_qty, _, _ = get_symbol_info(symbol)
        min_quantity = min_qty * ct_val

        quantity = (usdt_balance * params['BUY_RATIO'] / current_price) * params['LEVERAGE']

        if quantity < min_quantity:
            print(f"⚠️  {symbol} 仓位太小: {quantity:.6f} < {min_quantity:.6f}")
            return True

        if rsi <= params['RSI_BUY_VALUE']:
            order = place_order(symbol, 'buy', 'long', quantity)
            if order:
                print(f"🚀 {symbol} RSI开多: RSI={rsi:.2f} <= {params['RSI_BUY_VALUE']}")
            return True

        elif rsi >= params['RSI_SELL_VALUE']:
            order = place_order(symbol, 'sell', 'short', quantity)
            if order:
                print(f"🚀 {symbol} RSI开空: RSI={rsi:.2f} >= {params['RSI_SELL_VALUE']}")
            return True
        else:
            print(f"⏳ {symbol} RSI={rsi:.2f} 在区间[{params['RSI_BUY_VALUE']}, {params['RSI_SELL_VALUE']}]内，暂不交易")

    else:
        print(f"⏳ {symbol} 已有持仓，多={long_qty:.2f}, 空={short_qty:.2f}")

    return True


def trading_cycle():
    print("🎯 开始交易周期")
    for symbol in symbols:
        print(f"\n{'=' * 50}")
        print(f"📊 处理: {symbol}")
        success = execute_trading_logic(symbol)
        if not success:
            print(f"⚠️  {symbol} 交易逻辑失败")
    print("🎉 交易周期完成")


def main():
    try:
        print(" OKX RSI 交易机器人启动")
        print(f" 时间: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        load_params()
        api_key, api_secret, passphrase = get_api_credentials()

        if not init_okx(api_key, api_secret, passphrase):
            raise Exception("API初始化失败")

        for symbol in symbols:
            set_leverage(symbol)

        trading_cycle()
        print("🏁 程序退出")

    except Exception as e:
        print(f" 主程序错误: {str(e)}")
        logging.error(f"主程序错误: {str(e)}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()