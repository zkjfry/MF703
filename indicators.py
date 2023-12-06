import logging

import numpy as np
import pandas as pd

# Init Logging Facilities
log = logging.getLogger(__name__)


def macd(df, n_fast, n_slow):
    """Calculate MACD, MACD Signal and MACD difference
    :param df: pandas.DataFrame
    :param n_fast:
    :param n_slow:
    :return: pandas.DataFrame
    """
    EMAfast = pd.Series(df['Close'].ewm(span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df['Close'].ewm(span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df


def relative_strength_index(df, n, adjust=False):
    """Calculate Relative Strength Index(RSI) for given data.
    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    delta = df['Close'].diff(1).dropna()
    loss = delta.copy()
    gains = delta.copy()

    gains[gains < 0] = 0
    loss[loss > 0] = 0

    gain_ewm = gains.ewm(com=n - 1, adjust=adjust).mean()
    loss_ewm = abs(loss.ewm(com=n - 1, adjust=adjust).mean())

    RS = gain_ewm / loss_ewm
    RSI = 100 - 100 / (1 + RS)
    print(df)
    print(RSI)
    df['RSI_14'] = RSI

    return df
#     i = df.index[0]
#     UpI = [0]
#     DoI = [0]
#     while i + 1 <= df.index[-1]:
#         UpMove = float(df.loc[i + 1, 'High']) - float(df.loc[i, 'High'])
#         DoMove = float(df.loc[i, 'Low']) - float(df.loc[i + 1, 'Low'])
#         if UpMove > DoMove and UpMove > 0:
#             UpD = UpMove
#         else:
#             UpD = 0
#         UpI.append(UpD)
#         if DoMove > UpMove and DoMove > 0:
#             DoD = DoMove
#         else:
#             DoD = 0
#         DoI.append(DoD)
#         i = i + 1
#     UpI = pd.Series(UpI)

#     DoI = pd.Series(DoI)
#     PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
#     NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())

#     # rsi = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
#     rsi = pd.DataFrame(PosDI / (PosDI + NegDI), columns=['RSI_' + str(n)])
#     rsi = rsi.set_index(df.index)
#     df = df.join(rsi)
#     return df
