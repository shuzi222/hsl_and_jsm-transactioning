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
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

print("🤖 OKX RSI 交易机器人 v2.1")
print(f"🕐 {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
logging.info("程序启动")

PARAMS_FILE = '参数.json'
symbols = ['BTC-USDT-SWAP']
SYMBOL_PARAMS = {}
trade_client = market_client = account_client = public_client = None

def load_params():
    global SYMBOL_PARAMS
    with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
        raw_params = json.load(f)
    
    for symbol in symbols:
        SYMBOL_PARAMS[symbol] = {k: v['value'] for k, v in raw_params[symbol].items()}
    
    print("✅ 参数加载成功")
    params = SYMBOL_PARAMS[symbols[0]]
    print(f"📊 参数: RSI买入={params['RSI_BUY_VALUE']}, 卖出={params['RSI_SELL_VALUE']}, 仓位={params['BUY_RATIO']*100}%, 杠杆={params['LEVERAGE']}x, 止盈={params['TAKE_PROFIT']}%")
    logging.info(f"{symbols[0]} 参数: {SYMBOL_PARAMS[symbols[0]]}")
    return True

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
    raise Exception("API验证失败")

def set_leverage(symbol):
    params = SYMBOL_PARAMS[symbol]
    response = account_client.set_leverage(instId=symbol, lever=str(params['LEVERAGE']), mgnMode=params['MARGIN_MODE'])
    if response['code'] == '0':
        print(f"🔧 杠杆设置: {params['LEVERAGE']}x {params['MARGIN_MODE']}")
        return True
    print(f"⚠️ 杠杆设置失败: {response['msg']}")
    return False

def get_price(symbol):
    ticker = market_client.get_ticker(instId=symbol)
    if ticker['code'] == '0':
        return float(ticker['data'][0]['last'])
    return None

def get_klines(symbol, interval):
    klines = market_client.get_candlesticks(instId=symbol, bar=interval, limit='100')
    if klines['code'] == '0' and klines['data']:
        df = pd.DataFrame(klines['data'], columns=['ts','o','h','l','c','vol','volCcy','volCcyQuote','confirm'])
        df['c'] = df['c'].astype(float)
        print(f"📈 K线: {len(df)}条 {interval}")
        logging.info(f"📈 获取K线数据: {symbol}, {len(df)} 条, 周期: {interval}")
        return df
    return None

def calculate_rsi(df, period=14):
    try:
        close = df['c'].tail(period+1).values
        delta = np.diff(close)
        gain = np.mean(np.where(delta > 0, delta, 0)[-period:])
        loss = np.mean(np.where(delta < 0, -delta, 0)[-period:])
        if loss == 0:
            return 100
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        print(f"📊 RSI({period}): {rsi:.2f}")
        return rsi
    except Exception as e:
        print(f"❌ RSI计算失败: {e}")
        return None

def get_balance(symbol):
    # 余额
    balance = account_client.get_account_balance()
    usdt = 0.0
    if balance['code'] == '0':
        for asset in balance['data'][0]['details']:
            if asset['ccy'] == 'USDT':
                usdt = float(asset['availEq'])
    
    # 持仓
    positions = account_client.get_positions(instType='SWAP', instId=symbol)
    long_qty = short_qty = 0.0
    if positions['code'] == '0':
        for pos in positions['data']:
            if pos['instId'] == symbol:
                qty = float(pos['pos'])
                if pos['posSide'] == 'long':
                    long_qty = qty
                else:
                    short_qty = qty
    
    print(f"💰 余额: USDT={usdt:.2f}, 多仓={long_qty:.2f}, 空仓={short_qty:.2f}")
    return usdt, long_qty, short_qty

def execute_trading_logic(symbol):
    print(f"\n{'='*60}")
    print(f"🎯 处理 {symbol}")
    
    params = SYMBOL_PARAMS[symbol]
    
    # 1. 获取数据
    df = get_klines(symbol, params['RSI_TIMEFRAME'])
    if df is None:
        print("❌ K线获取失败")
        return False
    
    rsi = calculate_rsi(df)
    if rsi is None:
        print("❌ RSI计算失败")
        return False
    
    # 2. 获取余额持仓
    usdt, long_qty, short_qty = get_balance(symbol)
    price = get_price(symbol)
    if price:
        print(f"💹 价格: ${price:,.2f}")
    
    # 3. 交易决策
    if long_qty > 0 or short_qty > 0:
        pos_type = "多仓" if long_qty > 0 else "空仓"
        print(f"⏳ 已有{pos_type}: {long_qty if long_qty > 0 else short_qty:.2f}张")
        return True
    
    # 4. 开仓信号
    if rsi <= params['RSI_BUY_VALUE']:
        print(f"🚀 **买入信号**: RSI {rsi:.2f} ≤ {params['RSI_BUY_VALUE']}")
        quantity = (usdt * params['BUY_RATIO'] / price) * params['LEVERAGE']
        print(f"📦 计划开多: {quantity:.6f} BTC")
        
    elif rsi >= params['RSI_SELL_VALUE']:
        print(f"🔻 **卖出信号**: RSI {rsi:.2f} ≥ {params['RSI_SELL_VALUE']}")
        quantity = (usdt * params['BUY_RATIO'] / price) * params['LEVERAGE']
        print(f"📦 计划开空: {quantity:.6f} BTC")
        
    else:
        print(f"⏳ 等待: {params['RSI_BUY_VALUE']} < RSI {rsi:.2f} < {params['RSI_SELL_VALUE']}")
    
    return True

def main():
    try:
        load_params()
        get_api_credentials()
        
        for symbol in symbols:
            set_leverage(symbol)
            execute_trading_logic(symbol)
        
        print(f"\n🎉 交易周期完成 - {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logging.info("交易周期完成")
        
    except Exception as e:
        print(f"💥 错误: {e}")
        logging.error(f"主程序错误: {traceback.format_exc()}")

if __name__ == "__main__":
    main()

