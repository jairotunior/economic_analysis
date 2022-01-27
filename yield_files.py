import os
import pandas as pd
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

YIELD_DIR = os.path.join(BASE_DIR, "src", "dataset", "yields")

yield_list = []

columns = ['5 YR', '1 YR', '3 YR', '10 YR', '20 YR']

for file in os.listdir(YIELD_DIR):
    df = pd.read_excel(os.path.join(YIELD_DIR, file))
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

    column = df.columns[-1]
    if column in columns:
        df = df[['Date', column]]

    df = df.rename(columns={'Date': 'date'})

    df = df.set_index('date')

    yield_list.append(df)

df_result = pd.concat(yield_list, axis=1)

start_date = datetime.date(1929, 1, 1)
end_date = datetime.date(1989, 12, 31)

mask = (df_result.index.date >= start_date) & (df_result.index.date <= end_date)

df_result = df_result.loc[mask]

df_result.fillna(method='ffill', inplace=True)

df_result.to_csv(os.path.join(BASE_DIR, "src", "dataset", "yields_until_1990.csv"))
