import pandas as pd
from full_fred.fred import Fred
import re


x = re.search("_base$", "PCE_base")

print(x)

fred = Fred('api_fred.txt')

#fred.get_series_df()
#print(fred.get_all_sources())

#print(fred.get_a_category(0))
#print(fred.get_related_categories(0))

#print(fred.search_for_series(["Consumer"])['seriess'])