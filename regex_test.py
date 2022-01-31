import re
import datetime

all_series = [
    {"name": "1 YR Yield", "id": "US1Y", 'observation_start': "1/1/1929",
     'observation_end': datetime.date.today(), 'frequency': "Daily", 'units': "Percentage",
     'seasonal_adjustment': "", 'notes': ''},
    {"name": "10 YR Yield", "id": "US10Y", 'observation_start': "1/1/1929",
     'observation_end': datetime.date.today(), 'frequency': "Daily", 'units': "Percentage",
     'seasonal_adjustment': "", 'notes': ''},
]

search_word = "yield"
search_result = []

search_result = [s for s in all_series if len(re.findall(search_word, s['name'].lower())) > 0]

