import pandas as pd
import numpy as np
from src.transformers import Transformation

class Differentiation(Transformation):

    def __init__(self, **kwargs):
        name = "Differentiation"
        super().__init__(name=name, **kwargs)

    def transform(self, series, **kwargs):
        periods = kwargs.get('periods', 1)
        df = series.diff(periods=periods)
        return df