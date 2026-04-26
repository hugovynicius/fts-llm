import pandas as pd
import numpy as np

class DataSetUtil():
    def __init__(self):
        # debug attributes
        self.name = 'DataSet Util'
        self.shortname = 'dfutil'

    def get_samples_data_frame(self, data, perc):
        df = pd.DataFrame(data)
        df_sample = df.head(int(len(df) * perc))
        df_complement = df.iloc[max(df_sample.index):]
        return df_sample, df_complement, max(df_sample.index)

    def sample_first_prows(self, data, perc):
        return data.head(int(len(data) * (perc)))

    def clean_dataset(self, df):
        assert isinstance(df, pd.DataFrame)
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    # convert series to supervised learning
    def series_to_supervised_mimo(self, data, n_in, n_out, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            #names += [(df.columns[j] + '(t-%d)' % (i)) for j in range(n_vars)]
            names += [(df.columns[j] + '(t)') for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            names += [(df.columns[j] + '(t+%d)' % (i+1)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def series_to_supervised_miso(data, n_in, n_out, endog_var, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            #names += [(df.columns[j] + '(t-%d)' % (i)) for j in range(n_vars)]
            names += [(df.columns[j] + '(t)') for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df[endog_var].shift(-i))
            if i == 0:
                names += [(endog_var + '(t+1)')]
            else:
                names += [(endog_var + '(t+%d)' % (i+1))]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg