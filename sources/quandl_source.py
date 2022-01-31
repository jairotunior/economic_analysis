import re
import quandl
import pandas as pd
import numpy as np
from src.sources import Source
import datetime

class QuandlSource(Source):
    def __init__(self, api_key, **kwargs):
        assert api_key, "Favor suministrar una api_key"

        self.api_key = api_key

        name = kwargs.get('name', 'quandl')
        logo = 'https://mms.businesswire.com/media/20210629005397/en/839761/23/Quandl-Wordmark.jpg'
        header_color = 'black'
        header_background = '#2f2f2f'

        super().__init__(name=name, logo=logo, header_color=header_color, header_background=header_background, **kwargs)

        quandl.ApiConfig.api_key = self.api_key

        self.all_series = [
            {"name": "1 YR Yield", "id": "QUS01Y", 'observation_start': "1/1/1990",
             'observation_end': datetime.date.today(), 'frequency': "Daily", 'units': "Percentage",
             'seasonal_adjustment': "", 'notes': '', 'quandl_id': 'USTREASURY/YIELD', 'quandl_column': '1 YR'},
            {"name": "10 YR Yield", "id": "QUS10Y", 'observation_start': "1/1/1990",
             'observation_end': datetime.date.today(), 'frequency': "Daily", 'units': "Percentage",
             'seasonal_adjustment': "", 'notes': '', 'quandl_id': 'USTREASURY/YIELD', 'quandl_column': '10 YR'},
            {"name": "20 YR Yield", "id": "QUS10Y", 'observation_start': "1/1/1990",
             'observation_end': datetime.date.today(), 'frequency': "Daily", 'units': "Percentage",
             'seasonal_adjustment': "", 'notes': '', 'quandl_id': 'USTREASURY/YIELD', 'quandl_column': '20 YR'},
            {"name": "30 YR Yield", "id": "QUS30Y", 'observation_start': "1/1/1990",
             'observation_end': datetime.date.today(), 'frequency': "Daily", 'units': "Percentage",
             'seasonal_adjustment': "", 'notes': '', 'quandl_id': 'USTREASURY/YIELD', 'quandl_column': '30 YR'},
            {"name": "2 YR Yield", "id": "QUS02Y", 'observation_start': "1/1/1990",
             'observation_end': datetime.date.today(), 'frequency': "Daily", 'units': "Percentage",
             'seasonal_adjustment': "", 'notes': '', 'quandl_id': 'USTREASURY/YIELD', 'quandl_column': '2 YR'},
            {"name": "3 YR Yield", "id": "QUS03Y", 'observation_start': "1/1/1990",
             'observation_end': datetime.date.today(), 'frequency': "Daily", 'units': "Percentage",
             'seasonal_adjustment': "", 'notes': '', 'quandl_id': 'USTREASURY/YIELD', 'quandl_column': '3 YR'},
            {"name": "5 YR Yield", "id": "QUS05Y", 'observation_start': "1/1/1990",
             'observation_end': datetime.date.today(), 'frequency': "Daily", 'units': "Percentage",
             'seasonal_adjustment': "", 'notes': '', 'quandl_id': 'USTREASURY/YIELD', 'quandl_column': '5 YR'},
        ]

        self.search_result = []

    def do_search(self, search_word):
        assert type(search_word) is str, "El parametro search_word debe ser type str"

        self.search_result = [s for s in self.all_series if len(re.findall(search_word, s['name'].lower())) > 0]

    def get_data_serie(self, serie_id, columns=None, rename_column=None):
        serie_selected = None

        for s in self.all_series:
            if s['id'] == serie_id:
                serie_selected = s
                break

        quandl_id = serie_selected['quandl_id']

        df = quandl.get(quandl_id)
        df.index.name = 'date'

        if columns:
            for c in columns:
                column_name = c['name']
                column_type = c['type']
                periods = c['periods']

                if column_type == 'pct':
                    df[column_name] = df['value'].pct_change(periods=periods) * 100

            df = df[[*[c['name'] for c in columns]]]
        else:
            df = df[[serie_selected['quandl_column']]]

        if rename_column:
            df = df.rename(columns={serie_selected['quandl_column']: rename_column})

        return df

    def get_search_results(self):
        return self.search_result