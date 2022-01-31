import pandas as pd
import numpy as np
from src.transformers import Transformation


class PercentageChange(Transformation):

    def __init__(self, **kwargs):
        name = kwargs.get('name', "Percentage Change")
        kwargs['name'] = name
        super().__init__(**kwargs)

        self.periods = kwargs.get('periods', 1)

    def transform(self, series, **kwargs):
        return series.pct_change(periods=self.periods) * 100