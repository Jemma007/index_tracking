from WindPy import w
import pandas as pd
# 399108.SZ
w.start()

szb = pd.DataFrame()

w_szb = w.wsd("399108.SZ", "close", "2021-12-23", "2022-12-23", "TradingCalendar=SZSE;PriceAdj=F")
szb['time'] = w_szb.Times
szb['399108.SZ'] = w_szb.Data[0]

stocks_file = pd.read_excel('./data/指数样本股.xlsx')
stocks_code = list(stocks_file['证券代码'])
for i, stock in enumerate(stocks_code):
    stock = str(stock)+'.SZ'
    w_stock = w.wsd(stock, "close", "2021-12-23", "2022-12-23", "TradingCalendar=SZSE;PriceAdj=F")
    szb[stock] = w_stock.Data[0]
szb.to_csv('./data/szb_data.csv')


w.stop()