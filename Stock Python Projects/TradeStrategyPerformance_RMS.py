import yfinance as yf
import numpy as np
import ta
import pandas as pd
import matplotlib.pyplot as plt
import os

df = yf.download('BTC-USD', start='2022-04-01', interval='30m')

df['%K'] = ta.momentum.stoch(df.High, df.Low, df.Close, window=14, smooth_window=3)
df['%D'] = df['%K'].rolling(3).mean()
df['rsi'] = ta.momentum.rsi(df.Close, window=14)
df['macd'] = ta.trend.macd_diff(df.Close)
df.dropna(inplace=True)


def getTriggers(df, lags, buy=True):
    dfx = pd.DataFrame()
    for i in range(1, lags+1):
        if buy:
          mask = (df['%K'].shift(i) < 20) & (df['%D'].shift(i) < 20)  
        
        else:
            mask = (df['%K'].shift(i) > 80) & (df['%D'].shift(i) > 80)
        dfx = dfx.append(mask, ignore_index=True)

    return dfx.sum(axis=0)

def profitCalculator():
    BuyingPrices = df.loc[finalTrades.BuyingDates].Open
    SellingPrices = df.loc[finalTrades.SellingDates].Open
    return (SellingPrices.values - BuyingPrices.values) / BuyingPrices.values



df['BuyTrigger'] = np.where(getTriggers((df), 4),1,0)
df['SellTrigger'] = np.where(getTriggers((df), 4, False),1,0)

df['Buy'] = np.where((df.BuyTrigger) & (df['%K'].between(20,80)) & (df['%D'].between(20,80)) & (df.rsi > 50) & (df.macd > 0), 1, 0)
df['Sell'] = np.where((df.SellTrigger) & (df['%K'].between(20,80)) & (df['%D'].between(20,80)) & (df.rsi < 50) & (df.macd < 0), 1, 0)

BuyingDates, SellingDates = [], []

for i in range(len(df) - 1):
    if df.Buy.iloc[i]:
        BuyingDates.append(df.iloc[i + 1].name)
        for num, j in enumerate(df.Sell[i:]):
            if j:
                SellingDates.append(df.iloc[i + num + 1].name)
                break


cut = len(BuyingDates) - len(SellingDates)
if cut:
    BuyingDates = BuyingDates[:-cut]

frame = pd.DataFrame({'BuyingDates': BuyingDates, 'SellingDates': SellingDates})
finalTrades = frame[frame.BuyingDates > frame.SellingDates.shift(1)]


os.system('clear')
avgProfits = ((profitCalculator().mean()) * 100)
print(f'Calculated Average Profit Returned is: {avgProfits.round(3)}%')
cumulativeProfits = (((((profitCalculator() + 1).prod()) - 1) * 100))
print(f'Calculated Cumulative Profit Returned is: {cumulativeProfits.round(3)}%')


plt.figure(figsize=(10, 5))
plt.plot(df.Close, color='k', alpha=0.7)
plt.scatter(finalTrades.BuyingDates, df.Open[finalTrades.BuyingDates], marker='^', color='g', s=150, label='BUY')
plt.scatter(finalTrades.SellingDates, df.Open[finalTrades.SellingDates], marker='v', color='r', s=150, label='SELL')
plt.title('---RSI + MACD + Stochastic---')
plt.suptitle('Trade Strategy Performance Tracking') 
plt.xlabel(f'Dates')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()
