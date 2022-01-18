import pandas as pd
from full_fred.fred import Fred


fred = Fred('api_fred.txt')

#print(fred.get_all_sources())

print(fred.get_a_category(0))
print(fred.get_related_categories(0))