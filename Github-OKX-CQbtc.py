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
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('trading_bot.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

print("🤖 OKX RSI 简化交易机器人 v3.4")
print(f"🕐 {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
logging.info("简化交易机器人启动")

PARAMS_FILE = '参数.json'
symbols = ['BTC-USDT-SWAP']
SYMBOL_PARAMS = {}
state = {symbol: {'current_price': 0.0, 'latest_rsi': None} for symbol in symbols}

trade_client = market_client = account_client = public_client = None

def load_params():
    """自动修复参数"""
    global SYMBOL_PARAMS
    
    DEFAULT_CONFIG = {
        "BTC-USDT-SWAP": {
            "RSI_TIMEFRAME": {"value": "1H"},
            "RSI_BUY_VALUE": {"value": 25.0},
            "RSI_SELL_VALUE": {"value": 70.0},
            "BUY_RATIO": {"value": 0.1},
            "LEVERAGE": {"value": 3.0},
            "MARGIN_MODE": {"value": "cross"},
            "TAKE_PROFIT": {"value": 3.3},
            "STOP_LOSS": {"value": 2.0}
        }
    }
    
    try:
        with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                raw_params = json.loads(content)
            else:
                raw_params = {}
    except:
        raw_params = {}
    
    # 强制使用默认配置（确保STOP_LOSS=2.0）
    raw_params = DEFAULT_CONFIG
    
    # 保存正确配置
    with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
        json.dump(raw_params, f, ensure_ascii=False, indent=2)
    
    for symbol in symbols:
        SYMBOL_PARAMS[symbol] = {k: v['value'] for k, v in raw_params[symbol].items()}
    
    params = SYMBOL_PARAMS[symbols[0]]
    print(f"✅ 参数加载成功")
    print(f"   📊 RSI: 买入≤{params['RSI_BUY_VALUE']} 卖出≥{params['RSI_SELL_VALUE']}")
    print(f"   💼 仓位: {params['BUY_RATIO']*100}% × {params['LEVERAGE']}x")
    print(f"   🎯 止盈: {params['TAKE_PROFIT']}% 止损: {params['STOP_LOSS']}%")
    logging.info(f"{symbols[0]} 参数: {SYMBOL_PARAMS[symbols[0]]}")

def get_api_credentials():
    api_key = os.getenv('OKX_API_KEY')
    api_secret = os.getenv('OKX_API_SECRET')
    passphrase = os.getenv('OKX_PASSPHRASE')
    
    if not all([api_key, api_secret, passphrase]):
        raise ValueError("缺少API密钥")
    
    global trade_client, market_client, account_client, public_client
    trade_client = Trade.TradeAPI(api_key, api_secret, passphrase, flag='0')
    market_client = MarketData.MarketAPI(flag='0')
    account_client = Account.AccountAPI(api_key, api_secret, passphrase, flag='0')
    public_client = PublicData.PublicAPI(flag='0')
    
    balance = account_client.get_account_balance()
    if balance['code'] == '0':
        print("✅ API初始化成功")
        return True
    raise Exception(f"API验证失败: {balance['msg']}")

def set_leverage(symbol):
    params = SYMBOL_PARAMS[symbol]
    try:
        response = account_client.set_leverage(
            instId=symbol, 
            lever=str(params['LEVERAGE']), 
            mgnMode=params['MARGIN_MODE']
        )
        if response['code'] == '0':
            print(f"🔧 杠杆设置: {params['LEVERAGE']}x {params['MARGIN_MODE']}")
            logging.info(f"杠杆设置成功: {symbol}")
            return True
        print(f"⚠️ 杠杆设置跳过: {response['msg']}")
    except Exception as e:
        print(f"⚠️ 杠杆设置异常: {e}")
    return False

def get_price(symbol):
    try:
        ticker = market_client.get_ticker(instId=symbol)
        if ticker['code'] == '0':
            price = float(ticker['data'][0]['last'])
            state[symbol]['current_price'] = price
            print(f"💹 价格: ${price:,.2f}")
            return price
    except Exception as e:
        logging.error(f"价格获取失败: {e}")
        return None

def get_klines(symbol, interval):
    try:
        klines = market_client.get_candlesticks(instId=symbol, bar=interval, limit='100')
        if klines['code'] == '0' and klines['data']:
            df = pd.DataFrame(klines['data'], columns=['ts','o','h','l','c','vol','volCcy','volCcyQuote','confirm'])
            df['c'] = df['c'].astype(float)
            print(f"📈 K线: {len(df)}条 {interval}")
            logging.info(f"K线数据: {symbol}, {len(df)}条, 周期: {interval}")
            return df
        else:
            logging.error(f"K线数据为空: {klines}")
            return None
    except Exception as e:
        logging.error(f"K线获取失败: {e}")
        return None

def calculate_rsi(df, period=14):
    global symbol  # 修复全局变量问题
    try:
        if df is None or len(df) < period + 1:
            print(f"⚠️ 数据不足: {len(df) if df is not None else 0}条")
            return None
        close = df['c'].values
        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        state[symbol]['latest_rsi'] = rsi
        print(f"📊 RSI({period}): {rsi:.2f}")
        return rsi
    except Exception as e:
        logging.error(f"RSI计算失败: {e}")
        return None

def get_symbol_info(symbol):
    try:
        info = public_client.get_instruments(instType='SWAP', instId=symbol)
        if info['code'] == '0':
            data = info['data'][0]
            return (float(data['ctVal']), float(data['minSz']), float(data['tickSz']))
    except:
        pass
    return 0.01, 0.01, 0.01

def get_balance(symbol):
    global symbol  # 修复全局变量
    try:
        balance = account_client.get_account_balance()
        if balance['code'] != '0':
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        details = balance['data'][0].get('details', [])
        usdt = next((float(d['availEq']) for d in details if d['ccy'] == 'USDT'), 0.0)
        
        positions = account_client.get_positions(instType='SWAP', instId=symbol)
        long_qty = short_qty = long_avg = short_avg = 0.0
        if positions['code'] == '0':
            for pos in positions['data']:
                if pos['instId'] == symbol and float(pos['pos']) > 0:
                    qty = float(pos['pos'])
                    avg = float(pos['avgPx'])
                    if pos['posSide'] == 'long':
                        long_qty, long_avg = qty, avg
                    else:
                        short_qty, short_avg = qty, avg
        
        print(f"💰 余额: USDT={usdt:.2f}, 多={long_qty:.2f}张, 空={short_qty:.2f}张")
        return usdt, long_qty, short_qty, long_avg, short_avg
    except Exception as e:
        logging.error(f"余额查询失败: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0

def place_order(symbol, side, pos_side, quantity):
    params = SYMBOL_PARAMS[symbol]
    try:
        ct_val, min_qty, tick_sz = get_symbol_info(symbol)
        qty_contracts = max(round(quantity / ct_val), min_qty)
        
        if qty_contracts < min_qty:
            print(f"⚠️ 数量太小: {qty_contracts} < {min_qty}")
            return None
        
        price = state[symbol]['current_price']
        
        if pos_side == 'long':
            tp_price = round(price * (1 + params['TAKE_PROFIT']/100), -int(np.log10(tick_sz)))
            sl_price = round(price * (1 - params['STOP_LOSS']/100), -int(np.log10(tick_sz)))
        else:
            tp_price = round(price * (1 - params['TAKE_PROFIT']/100), -int(np.log10(tick_sz)))
            sl_price = round(price * (1 + params['STOP_LOSS']/100), -int(np.log10(tick_sz)))
        
        algo_order = {
            'tpTriggerPx': str(tp_price), 'tpOrdPx': '-1',
            'slTriggerPx': str(sl_price), 'slOrdPx': '-1',
            'tpOrdKind': 'condition',
            'slTriggerPxType': 'last', 'tpTriggerPxType': 'last'
        }
        
        order_params = {
            'instId': symbol, 'tdMode': params['MARGIN_MODE'],
            'side': side.lower(), 'posSide': pos_side.lower(),
            'ordType': 'market', 'sz': str(qty_contracts),
            'clOrdId': str(uuid.uuid4())[:32], 'attachAlgoOrds': [algo_order]
        }
        
        order = trade_client.place_order(**order_params)
        if order['code'] == '0':
            action = '开多' if side == 'buy' else '开空'
            print(f"✅ {action}成功: {quantity:.6f} BTC")
            print(f"   🎯 止盈: ${tp_price:,.2f} 止损: ${sl_price:,.2f}")
            logging.info(f"{symbol} {action}: {quantity:.6f} BTC")
            return order['data'][0]['ordId']
        else:
            print(f"❌ 下单失败: {order['msg']}")
            return None
    except Exception as e:
        logging.error(f"下单异常: {e}")
        return None

def check_take_profit_stop_loss(symbol, long_qty, short_qty, long_avg, short_avg):
    params = SYMBOL_PARAMS[symbol]
    price = state[symbol]['current_price']
    rsi = state[symbol]['latest_rsi']
    
    if long_qty > 0 and long_avg > 0:
        profit_pct = (price - long_avg) / long_avg * 100
        if profit_pct >= params['TAKE_PROFIT'] or (rsi and rsi >= params['RSI_SELL_VALUE']):
            return 'long', long_qty, f"止盈{profit_pct:.1f}%"
        if profit_pct <= -params['STOP_LOSS']:
            return 'long', long_qty, f"止损{profit_pct:.1f}%"
    
    if short_qty > 0 and short_avg > 0:
        profit_pct = (short_avg - price) / short_avg * 100
        if profit_pct >= params['TAKE_PROFIT'] or (rsi and rsi <= params['RSI_BUY_VALUE']):
            return 'short', short_qty, f"止盈{profit_pct:.1f}%"
        if profit_pct <= -params['STOP_LOSS']:
            return 'short', short_qty, f"止损{profit_pct:.1f}%"
    
    return None, 0.0, None

def execute_trading_logic(symbol):
    global symbol  # 修复全局变量
    print(f"\n{'='*50}")
    print(f"🎯 RSI交易分析: {symbol}")
    print(f"{'='*50}")
    
    params = SYMBOL_PARAMS[symbol]
    
    # ✅ 修复：使用 df is None
    df = get_klines(symbol, params['RSI_TIMEFRAME'])
    if df is None:
        print("❌ K线数据获取失败")
        return False
    
    rsi = calculate_rsi(df)
    if rsi is None:
        print("❌ RSI计算失败")
        return False
    
    price = get_price(symbol)
    if price is None:
        print("❌ 价格获取失败")
        return False
    
    usdt, long_qty, short_qty, long_avg, short_avg = get_balance(symbol)
    
    # 平仓检查
    pos_side, qty, reason = check_take_profit_stop_loss(symbol, long_qty, short_qty, long_avg, short_avg)
    if pos_side:
        order = place_order(symbol, 'sell' if pos_side == 'long' else 'buy', pos_side, qty)
        if order:
            print(f"✅ 平仓成功: {reason}")
            usdt, long_qty, short_qty, _, _ = get_balance(symbol)
    
    # 开仓
    if long_qty == 0 and short_qty == 0 and usdt > 10:
        quantity = (usdt * params['BUY_RATIO'] / price) * params['LEVERAGE']
        
        ct_val, min_qty, _ = get_symbol_info(symbol)
        min_quantity = min_qty * ct_val
        
        if quantity >= min_quantity:
            if rsi <= params['RSI_BUY_VALUE']:
                order = place_order(symbol, 'buy', 'long', quantity)
                if order:
                    print(f"🚀 RSI开多: {rsi:.1f} ≤ {params['RSI_BUY_VALUE']}")
                    return True
                    
            elif rsi >= params['RSI_SELL_VALUE']:
                order = place_order(symbol, 'sell', 'short', quantity)
                if order:
                    print(f"🔻 RSI开空: {rsi:.1f} ≥ {params['RSI_SELL_VALUE']}")
                    return True
        else:
            print(f"⚠️ 仓位太小: {quantity:.6f} < {min_quantity:.6f}")
    else:
        status = "有仓位" if (long_qty > 0 or short_qty > 0) else "余额不足"
        print(f"⏳ 等待: RSI {rsi:.1f} ({params['RSI_BUY_VALUE']}-{params['RSI_SELL_VALUE']}) [{status}]")
    
    return True

def main():
    global symbol
    try:
        load_params()
        get_api_credentials()
        
        for symbol in symbols:
            set_leverage(symbol)
            if not execute_trading_logic(symbol):
                print(f"⚠️ {symbol} 交易失败")
        
        print(f"\n🎉 RSI交易周期完成!")
        logging.info("交易周期完成")
        
    except Exception as e:
        print(f"💥 错误: {e}")
        logging.error(f"主程序错误: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
