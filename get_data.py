import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta, datetime
from full_fred.fred import Fred
import quandl

quandl.ApiConfig.api_key = "4dvrfm6eBSwRSxwBP1Jx"

def bitcoin_fear_and_greed():
    fng_bitcoin_base_url = "https://api.alternative.me/fng/?limit={limit}&format=json&date_format={date_format}"
    limit = 0
    date_format = "world"

    try:
        response = requests.get(fng_bitcoin_base_url.format(limit=limit, date_format=date_format))

        if response.status_code == 200:
            data = response.json()['data']
            df = pd.DataFrame(data=data)
            df = df.rename(columns={'value': 'BTCFNG', 'value_classification': 'class', 'timestamp': 'date', 'time_until_update': 'update'})
            df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")
            df = df.drop('update', axis=1)
            df = df.set_index('date')

            return df
    except:
        return None

    return None

def get_quandl_dataset(serie_id):
    df = quandl.get(serie_id)
    df.index.name = 'date'

    return df

def get_fred_dataset(serie_id, columns=None, rename_column=None):
    data = fred.get_series_df(serie_id)
    data['date'] = pd.to_datetime(data['date'], format="%Y/%m/%d")
    data['value'].replace(".", np.nan, inplace=True)
    data = data.set_index('date')
    data['value'] = data['value'].astype('float')

    if columns:
        for c in columns:
            column_name = c['name']
            column_type = c['type']
            periods = c['periods']

            if column_type == 'pct':
                data[column_name] = data['value'].pct_change(periods=periods) * 100

        data = data[[*[c['name'] for c in columns]]]
    else:
        data = data[['value']]

    if rename_column:
        data = data.rename(columns={'value': rename_column})

    return data


df = get_quandl_dataset('USTREASURY/YIELD')
print(df.head())

"""
fred = Fred('api_fred.txt')

list_ts = [
    # Fear and Greed
    {'tick': 'BTCFNG', 'name': 'BITCOIN Fear and Greed', 'source': bitcoin_fear_and_greed, 'freq': 'D'},

    # Major Index
    {'tick': 'SP500', 'name': 'S&P 500', 'source': 'fred', 'freq': 'D'},
    {'tick': 'NASDAQ100', 'name': 'NASDAQ 100 Index', 'source': 'fred', 'freq': 'D'},
    {'tick': 'DJIA', 'name': 'Dow Jones Industrial Average', 'source': 'fred', 'freq': 'D'},
    {'tick': 'VIXCLS', 'name': 'CBOE Volatility Index: VIX', 'source': 'fred', 'freq': 'D'},

    # Interes Rates and Yields
    {'tick': 'FEDFUNDS', 'name': 'Federal Funds Effective Rate', 'source': 'fred', 'freq': 'M'},
    {'tick': 'DFEDTAR', 'name': 'Federal Funds Target Rate (DISCONTINUED)', 'source': 'fred', 'freq': 'D'},
    {'tick': 'DFEDTARL', 'name': 'Federal Funds Target Range - Lower Limit', 'source': 'fred', 'freq': 'D'},
    {'tick': 'DFEDTARU', 'name': 'Federal Funds Target Range - Upper Limit', 'source': 'fred', 'freq': 'D'},
    {'tick': 'DFF', 'name': 'Federal Funds Effective Rate', 'source': 'fred', 'freq': 'D'},
    {'tick': 'WM2NS', 'name': 'M2', 'source': 'fred', 'freq': 'W'},
    {'tick': 'M2V', 'name': 'Velocity of M2 Money Stock', 'source': 'fred', 'freq': 'Q'},
    {'tick': 'USTREASURY/YIELD', 'name': 'Yield Curve', 'source': 'quandl', 'freq': 'D'},

    # US Economy
    {'tick': 'CPILFESL', 'name': 'Consumer Price Index for All Urban Consumers: All Items Less Food and Energy in U.S. City Average', 'source': 'fred', 'freq': 'M'},
    {'tick': 'CPIAUCSL', 'name': 'Consumer Price Index for All Urban Consumers: All Items in U.S. City Average', 'source': 'fred', 'freq': 'M'},
    {'tick': 'PCE', 'name': 'Personal Consumption Expenditures', 'source': 'fred', 'freq': 'M'},
    {'tick': 'PCEPILFE', 'name': 'Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index)', 'source': 'fred', 'freq': 'M'},
    {'tick': 'GDP', 'name': 'Gross Domestic Product', 'units': 'Billions of Dollars', 'source': 'fred', 'freq': 'Q'},
    {'tick': 'NETEXP', 'name': 'Net Exports of Goods and Services', 'source': 'fred', 'freq': 'Q'},
    {'tick': 'UNRATE', 'name': 'Unemployment Rate', 'source': 'fred', 'freq': 'M'},
    {'tick': 'PAYEMS', 'name': 'All Employees, Total Nonfarm', 'source': 'fred', 'freq': 'M'},
    {'tick': 'HSN1F', 'name': 'New One Family Houses Sold: United States', 'source': 'fred', 'freq': 'M'},


    # US Federal Debt
    {'tick': 'FYFSD', 'name': 'Federal Surplus or Deficit', 'source': 'fred', 'freq': 'Y'},
    {"tick": "Q1527BUSQ027NNBR", "name": "Government Purchases of Goods and Services for United States", 'source': 'fred', 'freq': 'Q'},
    {"tick": "Q15036USQ027SNBR", "name": "Federal Government Purchases of Goods and Services, National Defense for United States", 'source': 'fred', 'freq': 'Q'},
    {'tick': 'W068RCQ027SBEA', 'name': 'Government total expenditures', 'source': 'fred', 'freq': 'Q'},
    {'tick': 'GFDEBTN', 'name': 'Federal Debt: Total Public Debt', 'source': 'fred', 'freq': 'Q'},
    {'tick': 'GFDEGDQ188S', 'name': 'Federal Debt: Total Public Debt as Percent of Gross Domestic Product', 'source': 'fred', 'freq': 'Q'},


    # US Credit
    {'tick': 'BUSLOANS', 'name': 'Commercial and Industrial Loans, All Commercial Banks', 'source': 'fred', 'freq': 'M'},
    {'tick': 'DPSACBW027SBOG', 'name': 'Deposits, All Commercial Banks', 'source': 'fred', 'freq': 'W'},
    {'tick': 'CCLACBW027SBOG', 'name': 'Consumer Loans: Credit Cards and Other Revolving Plans, All Commercial Banks', 'source': 'fred', 'freq': 'W'},
    {'tick': 'DRSDCIS', 'name': 'Net Percentage of Domestic Banks Reporting Stronger Demand for Commercial and Industrial Loans from Small Firms', 'source': 'fred', 'freq': 'Q'},
    {'tick': 'DRTSCLCC', 'name': 'Net Percentage of Domestic Banks Tightening Standards for Credit Card Loans', 'source': 'fred', 'freq': 'Q'},

    # Inventories
    {'tick': 'ISRATIO', 'name': 'Total Business: Inventories to Sales Ratio', 'source': 'fred', 'freq': 'M'},
    {'tick': 'RETAILIRSA', 'name': 'Retailers: Inventories to Sales Ratio', 'source': 'fred', 'freq': 'M'},
    {'tick': 'MNFCTRIRSA', 'name': 'Manufacturers: Inventories to Sales Ratio', 'source': 'fred', 'freq': 'M'},
    {'tick': 'AISRSA', 'name': ' Auto Inventory/Sales Ratio', 'source': 'fred', 'freq': 'M'},
    {'tick': 'BUSINV', 'name': 'Total Business Inventories', 'source': 'fred', 'freq': 'M'},
    {'tick': 'RETAILIMSA', 'name': 'Retailers Inventories', 'source': 'fred', 'freq': 'M'},
    {'tick': 'MNFCTRIMSA', 'name': 'Manufacturers Inventories', 'source': 'fred', 'freq': 'M'},
    {'tick': 'AUINSA', 'name': 'Domestic Auto Inventories', 'source': 'fred', 'freq': 'M'},
    {'tick': 'RRSFS', 'name': 'Advance Real Retail and Food Services Sales', 'source': 'fred', 'freq': 'M'},

    # Money Markets
    {'tick': 'SOFR', 'name': 'Secured Overnight Financing Rate', 'source': 'fred', 'freq': 'D'},
    {'tick': 'WLRRAL', 'name': 'Liabilities and Capital: Liabilities: Reverse Repurchase Agreements: Wednesday Level', 'source': 'fred', 'freq': 'W'}, # Overnight Repo levels of trading
    {'tick': 'RPONTTLD', 'name': 'Overnight Repurchase Agreements: Total Securities Purchased by the Federal Reserve in the Temporary Open Market Operations', 'source': 'fred', 'freq': 'D'},
    {'tick': 'RPONTSYD', 'name': 'Overnight Repurchase Agreements: Treasury Securities Purchased by the Federal Reserve in the Temporary Open Market Operations', 'source': 'fred', 'freq': 'D'},
    {'tick': 'RRPONTSYD', 'name': 'Overnight Reverse Repurchase Agreements: Treasury Securities Sold by the Federal Reserve in the Temporary Open Market Operations', 'source': 'fred', 'freq': 'D'},
    {'tick': 'WORAL', 'name': 'Assets: Other: Repurchase Agreements: Wednesday Level', 'source': 'fred', 'freq': 'W'},
]

df_list = []

for k in list_ts:
    tick = k['tick']
    print(tick)
    freq = k['freq']
    source = k['source']

    df = None

    # Es una funcion custom
    if callable(source):
        df = source()
        print(df.head())
    elif source == 'fred':
        df = get_fred_dataset(tick, rename_column=tick)
    elif source == 'quandl':
        df = get_quandl_dataset(tick)

    df.fillna(method='ffill', inplace=True)
    df_list.append(df)

df_result = pd.concat(df_list, axis=1)
print(df_result.head())
print(df_result.columns)

"""