from yahoo_finance import Share
from data_handling import db_handling
import datetime

shares = {}

def datetime_to_str(dt):
    return dt.strftime("%Y-%m-%d")

def get_quote(symbol, date1, date2):
    if symbol.upper() not in shares.keys():
        shares[symbol] = Share(symbol)
    return shares[symbol.upper()].get_historical(datetime_to_str(date1),datetime_to_str(date2))

def open_to_open_yield(symbol, date1, date2):
   quote = get_quote(symbol, date1 , date2)
   open1 = float(quote[0]["Open"])
   open2 = float(quote[len(quote)-1]["Open"])
   return open2/open1 -1

def get_price(symbol, dt, price_df):
    dt_rounded = db_handling.round_to_next_minute(dt)
    assert dt_rounded.second == 0
    try:
        price = float(price_df.loc[dt_rounded, symbol])
    except KeyError:
        price = float('nan')
    return price
