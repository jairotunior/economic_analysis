import pandas as pd
import numpy as np
from src.transformers import Transformation


class FractionalDifferentiation(Transformation):

    def __init__(self, **kwargs):
        name = kwargs.get('name', "Fractional Differentiation")
        super().__init__(name=name, **kwargs)

    def _get_weights_ffd(self, d, size):
        w = [1.]
        for k in range(1, size):
            w_ = -w[-1] / k * (d - k + 1)
            w.append(w_)

        w = np.array(w[::-1]).reshape(-1, 1)
        return w

    def transform(self, series, **kwargs):
        d = kwargs.get('d')
        thres = kwargs.get('thres', 1e-5)

        # 1. Compute weights for the longest series
        w = self._get_weights_ffd(d, thres)
        width = len(w) - 1

        # 2. Apply weights to values
        df = {}

        for name in series.columns:
            seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
            for iloc1 in range(width, seriesF.shape[0]):
                loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
                if not np.isfinite(series.loc[loc1, name]): continue # Exclude NAs
                df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
            df[name] = df_.copy(deep=True)

        df = pd.concat(df, exit=1)

        return df