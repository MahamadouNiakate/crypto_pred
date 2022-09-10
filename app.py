import streamlit as st

import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
# import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

start_date_string = '2017-01-01'
btc_ticker = yf.Ticker('BTC-USD')
# btc = btc_ticker.history(period="max")
today = datetime.today().strftime('%Y-%m-%d')
btc = btc_ticker.history(start=start_date_string, end=today)
btc = btc.drop(columns=['Dividends', 'Stock Splits'])

def preprocess(btc):

    # monthly_close = btc.groupby(btc.index.month)['Close'].tail(1)

    month_close_df = btc.copy()[btc.index.is_month_end]
    month_close_df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)

    month_close_df.rename(columns={'Close':'month_close'}, inplace=True)


    # month_close_df
    btc_month_mean = btc.resample('M').mean()
    btc_month_mean.rename(columns={'Open':'month_ave_open', 'High':'month_ave_high', 'Low':'month_ave_low', 'Close':'month_ave_close', 'Volume':'month_ave_volume'}, inplace=True)
    # btc_month_mean.tail()


    btc_month_mean['month_ave_close_shift'] = btc_month_mean['month_ave_close'].shift(-1)
    # btc_month_mean.dropna(inplace=True)
    # btc_month_mean

    btc_month_mean['month_ave_close_shift_diff'] = btc_month_mean['month_ave_close_shift'].diff()
    btc_month_mean.dropna(inplace=True)
    # btc_month_mean

    btc_month_mean['month_ave_close_shift_diff_percent'] = None

    for i in range(1, len(btc_month_mean)):
        btc_month_mean['month_ave_close_shift_diff_percent'][i] = \
        (btc_month_mean['month_ave_close_shift'][i] - btc_month_mean['month_ave_close_shift'][i-1]) / btc_month_mean['month_ave_close_shift'][i-1]

    btc_month_mean.dropna(inplace=True)
    # btc_month_mean

    train_test_split = int(btc_month_mean.shape[0] * 0.75)
    train = btc_month_mean.copy().iloc[:train_test_split]
    test = btc_month_mean.copy().iloc[train_test_split:]
    # train.shape, test.shape

    # scaler = StandardScaler()
    scaler = MinMaxScaler()

    #hide warnings
    train_scaled = pd.DataFrame()
    test_scaled = pd.DataFrame()

    train_scaled[['month_ave_open', 'month_ave_high', 'month_ave_low', 'month_ave_volume', 'month_ave_close', 'month_ave_close_shift_diff_percent']] = \
    scaler.fit_transform(train[['month_ave_open', 'month_ave_high', 'month_ave_low', 'month_ave_volume', 'month_ave_close', 'month_ave_close_shift_diff_percent']])
    test_scaled[['month_ave_open', 'month_ave_high', 'month_ave_low', 'month_ave_volume', 'month_ave_close', 'month_ave_close_shift_diff_percent']] = \
    scaler.transform(test[['month_ave_open', 'month_ave_high', 'month_ave_low', 'month_ave_volume', 'month_ave_close', 'month_ave_close_shift_diff_percent']])

    train_scaled.index = train.index
    test_scaled.index = test.index

    X_train = train_scaled.copy().drop(columns=['month_ave_close_shift_diff_percent'])
    y_train = train_scaled.copy()['month_ave_close_shift_diff_percent']

    X_test = test_scaled.copy().drop(columns=['month_ave_close_shift_diff_percent'])
    y_test = test_scaled.copy()['month_ave_close_shift_diff_percent']

    # X_train.shape, y_train.shape, X_test.shape, y_test.shape
    return btc_month_mean, month_close_df, test, X_train, y_train, X_test, y_test

def append_gain_loss(month_close_df, df, debug=False):

  df = df.copy().merge(month_close_df, how='left', left_index=True, right_index=True)

  df['gain_loss'] = None
  df['gain_loss_percent'] = None
  for i in range(1, len(df)):

    df['gain_loss'][i] = df['month_close'][i] - df['month_close'][i-1]
    df['gain_loss_percent'][i] = ((df['month_close'][i] - df['month_close'][i-1])/((df['month_close'][i]+df['month_close'][i-1])/2))

    # df['month_close'][i]/((df['month_close'][i] - df['month_close'][i-1]))
  df.dropna(inplace=True)
  return df

def calc_gain_loss(df, threshold, column, debug=False):
#   print(f'======calc_gain_loss')
  gain_loss = []
  for index, row in df.iterrows():
    if row[column] > threshold:
      if debug:
        print(f'{index}::{row[column]}')

      # gain_loss.append(row['Close_shift'] - row['Close'])
      gain_loss.append(row['gain_loss'])
    else:
      gain_loss.append(0)

  if debug:
    print(f'GAIN/LOSS: {sum(gain_loss)}')

#   return sum(gain_loss)
  return gain_loss

def calc_gain_loss_2(df, threshold, column, investment_amount, debug=False):
#   print(f'======calc_gain_loss_2')

  gain_loss = []

  for index, row in df.iterrows():
    if row[column] > threshold:
      if debug:
        print(f'{index}::{row[column]}::{row["gain_loss_percent"]}')

      # gain_loss.append(row['Close_shift'] - row['Close'])
      gain_loss.append(investment_amount * row['gain_loss_percent'])
    else:
      gain_loss.append(0)

  if debug:
    print(f'GAIN/LOSS: {sum(gain_loss)}')

#   return sum(gain_loss)
  return gain_loss







'''----------------------'''
'''----------------------'''


# def calc_gain_loss_3(df, threshold, column, investment_amount, debug=False):
#   print(f'======calc_gain_loss_3')
'''
return:

gain/loss
dict date:portfolio_gain_loss


'''



#   gain_loss = {}
#   df.columns
#   for index, row in df.iterrows():

#     break
#     if row[column] > threshold:
#       if debug:
#         print(f'{index}::{row[column]}::{row["gain_loss_percent"]}')

#       # gain_loss.append(row['Close_shift'] - row['Close'])
#       gain_loss.append(investment_amount * row['gain_loss_percent'])
#     else:
#       gain_loss.append(0)

#   if debug:
#     print(f'GAIN/LOSS: {sum(gain_loss)}')

# #   return sum(gain_loss)
#   return gain_loss


'''----------------------'''
'''----------------------'''


def standard_lr(test, X_train, y_train, X_test, y_test):

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    X_test_predict = lr.predict(X_test)

    X_test_predict_df = pd.DataFrame(X_test_predict, index=X_test.index, columns=['Y_PRED'])
    predict_df = test.merge(X_test_predict_df, how='left', on='Date')

    return predict_df

def calc_sliding_window(window_size, input_df, debug=False):
  predict_df = pd.DataFrame(columns=['Y_PRED'])
  # btc_month_mean.shape

  for i in range(window_size, input_df.shape[0]+1):

    window_data = input_df[i-window_size:i]
    if debug:
      print(f'{i-window_size}->{i}')

    X = window_data.copy()[['month_ave_open', 'month_ave_high', 'month_ave_low', 'month_ave_close', 'month_ave_volume']]

    X_train = X[:X.shape[0]-1]
    X_pred = X[X.shape[0]-1:]

    y = window_data['month_ave_close_shift_diff_percent']
    y_train = y[:X.shape[0]-1]

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_pred)[0]

    y_pred_df = pd.DataFrame(data=[y_pred], columns=['Y_PRED'], index=[X_pred.index[0]])
    predict_df = pd.concat([predict_df, pd.DataFrame(y_pred_df)], ignore_index=False)

  return predict_df

def calc_sliding_window2(model_name, model, window_size, input_df, debug=False):
  predict_df = pd.DataFrame(columns=['Y_PRED'])

  for i in range(window_size, input_df.shape[0]+1):

    window_data = input_df[i-window_size:i]
    if debug:
      print(f'{i-window_size}->{i}')

    X = window_data.copy()[['month_ave_open', 'month_ave_high', 'month_ave_low', 'month_ave_close', 'month_ave_volume']]

    X_train = X[:X.shape[0]-1]
    X_pred = X[X.shape[0]-1:]

    y = window_data['month_ave_close_shift_diff_percent']
    y_train = y[:X.shape[0]-1]

    fit = model.fit(X_train, y_train)
    y_pred = model.predict(X_pred)[0]

    y_pred_df = pd.DataFrame(data=[[model_name, y_pred]], columns=['model_name', 'Y_PRED'], index=[X_pred.index[0]])
    predict_df = pd.concat([predict_df, pd.DataFrame(y_pred_df)], ignore_index=False)

  return predict_df

def calc_with_params(start_date, bet_size):

#   print(f'::{start_date}, {bet_size}')

  window_sizes = [3, 6, 12]
  gain_percents = [0, .05, .1]
  model_list = [('lr', LinearRegression()),
                # ('xgb', xgb.XGBRegressor(objective ='reg:squarederror')),
               ('knn', KNeighborsRegressor(n_neighbors=2)),
                ('rf', RandomForestRegressor()) ]

  window_df = btc_month_mean.copy()[start_date:]
#   print(f'window_df[0:1]: {window_df[0:1]}')
  result_df = pd.DataFrame(columns=['window', 'percent', 'model_name', 'gain_loss', f'gain_loss_{bet_size}'])

  for window_size in window_sizes:
    for gain_percent in gain_percents:
      for model_name, model in model_list:
        # st.write(f'{window_size}:{gain_percent}:{model_name}')
        test_df = calc_sliding_window2(model_name, model, window_size, window_df, False)

        test_df = append_gain_loss(month_close_df, test_df)

        gain_loss_list = calc_gain_loss(test_df, gain_percent, 'Y_PRED')
        l = len(list(filter(lambda x: (x < 0), gain_loss_list)))
        w = len(list(filter(lambda x: (x > 0), gain_loss_list)))
        p = len(list(filter(lambda x: (x == 0), gain_loss_list)))
        # print(f'b gain_losslist: {w}/{l}/{p}::{len(gain_loss_list)}::{gain_loss_list}--{test_df.shape}')
        # st.write(f'gain_loss_list: {gain_loss_list}')
        gain_loss = sum(gain_loss_list)

        gain_loss_bet_size_list = calc_gain_loss_2(test_df, gain_percent, 'Y_PRED', bet_size, False)
        l_100 = len(list(filter(lambda x: (x < 0), gain_loss_bet_size_list)))
        w_100 = len(list(filter(lambda x: (x > 0), gain_loss_bet_size_list)))
        p_100 = len(list(filter(lambda x: (x == 0), gain_loss_bet_size_list)))
        # print(f'gain_loss_100_list: {w_100}/{l_100}/{p_100}::{len(gain_loss_bet_size_list)}::{gain_loss_bet_size_list}--{test_df.shape}')
        # print(f'gain_loss_bet_size_list: {gain_loss_bet_size_list}')

        # dostuff = calc_gain_loss_3(test_df, gain_percent, 'Y_PRED', bet_size, True)

        cum_gain_loss_100 = np.cumsum(np.array(gain_loss_bet_size_list))
        gain_loss_bet_size = sum(gain_loss_bet_size_list)
        # print(f'gain_loss_bet_size_list: {gain_loss_bet_size_list}')
        # print(f'cum_gain_loss_100: {cum_gain_loss_100}')

        loop_df = pd.DataFrame(data=[[window_size, gain_percent, model_name, gain_loss,
                                      w, l, p, gain_loss_bet_size, w_100, l_100, p_100, cum_gain_loss_100]],
                               columns=['window', 'percent', 'model_name', 'gain_loss',
                                        'gl_win', 'gl_loss', 'gl_pass', f'gain_loss_{bet_size}',
                                        'gl_win_100', 'gl_loss_100', 'gl_pass_100', 'cum_gain_loss_100'])

        # print(f'-->{result_df.shape}, {loop_df.shape}, {len(gain_loss_bet_size_list)}')
        result_df = pd.concat([result_df, loop_df], ignore_index=True)

  result_df["gl_win_100"] = result_df["gl_win_100"].astype(int)
  result_df["gl_loss_100"] = result_df["gl_loss_100"].astype(int)
  result_df["gl_pass_100"] = result_df["gl_pass_100"].astype(int)
  # st.write(f'd result_df.shape: {result_df.shape}')

  result_df["gain_loss"] = round(result_df["gain_loss"], 2)
  result_df["gain_loss_100"] = round(result_df["gain_loss_100"], 2)

  result_df = result_df.sort_values(by='gain_loss_100', ascending=False)
  # st.write(f'e result_df.shape: {result_df.shape}')

  result_df = result_df[['window', 'percent', 'model_name', 'gain_loss_100', 'gl_win_100', 'gl_loss_100', 'gl_pass_100', 'cum_gain_loss_100']]
  # # st.write(f'f result_df.shape: {result_df.shape}')
  result_df = result_df.rename(columns={"window": "window", "percent": "threshold", "model_name": "model",
                                        "gain_loss_100": "gain_loss", "gl_win_100": "win",
                                        "gl_loss_100": "loss", "gl_pass_100": "pass", "cum_gain_loss_100":"cum_gain_loss_100"})
  # # st.write(f'g result_df.shape: {result_df.shape}')
  result_df = result_df.assign(hack='').set_index('hack')
  # st.write(f'h result_df.shape: {result_df.shape}')




  return result_df

def draw_charts(df):

    min_len = 100
    max_len = 0
    #find max/min length
    for loc in range(df.shape[0]):

        if len(df.iloc[loc]['cum_gain_loss_100']) > max_len:
            max_len = len(df.iloc[loc]['cum_gain_loss_100'])

        if len(df.iloc[loc]['cum_gain_loss_100']) < min_len:
            min_len = len(df.iloc[loc]['cum_gain_loss_100'])
    # print(f'min_len: {min_len}')
    # print(f'max_len: {max_len}')

    #bring all arrays into the same length
    for loc in range(df.shape[0]):
        loop_array = df['cum_gain_loss_100'].iloc[loc]
        loop_list = loop_array.tolist()
        while len(loop_list) < max_len:
            # print(f'lengthening: {type(loop_array)}')

            loop_list.insert(0, -1)
            # loop_list = loop_list.insert(0, -1)
        loop_array = np.asarray(loop_list)
        # print(f'A len->{len(loop_array)}')

        df['cum_gain_loss_100'].iloc[loc] = loop_array
        # print(f'loop_array: {len(loop_array)}')

    #setup our dataframe for the line graph
    chart_df_data = []
    for loc in range(df.shape[0]):
        data = df['cum_gain_loss_100'].iloc[loc]
        # print(f'A data{loc}:  {data.shape}')
        # print(f'B data{loc}:  {type(data)}')
        # print(f'C data{loc}:  {data.reshape(-1, 1).shape}')

        chart_df_data.append(data.reshape(-1, 1))
    # print(f'len: {len(chart_df_data)}')
    # print(f'type: {type(chart_df_data)}')
    # print(f'chart_df_data.shape: {chart_df_data.shape}')

    y=np.array([np.array(xi) for xi in chart_df_data])
    # print(f'type: {type(y)}')
    print(f'shape: {y.shape}')
    y = y.reshape(y.shape[1], y.shape[0])
    # print(f'shape: {y.shape}')

    df0 = pd.DataFrame(y)
    # print(f'df0.shape: {df0.shape}')
    st.line_chart(df0)

    # df = pd.DataFrame(
    #         np.random.randn(20, 3),
    #         columns=['a', 'b', 'c']
    #     )
    # st.line_chart(df)

    # print(f'df0.shape: {df0.shape}')
    # print(f'df.shape: {df.shape}')
'''------------------------------------------------'''

btc_month_mean, month_close_df, test, X_train, y_train, X_test, y_test = preprocess(btc)

'''------------------------------------------------'''

# from_date = '2010-01-01'
# st.write(f'Starting: {from_date}')
# a = calc_with_params(from_date, 100)
# st.write(a.head())

# from_date = '2020-01-01'
# st.write(f'Starting: {from_date}')
# a = calc_with_params(from_date, 100)
# st.write(a.head())

# from_date = '2021-01-01'
# st.write(f'Starting: {from_date}')
# a = calc_with_params(from_date, 100)
# st.write(a.head())

to_index = 3

# from_date = '2021-07-01'
# st.write(f'Starting: {from_date}')
# a = calc_with_params(from_date, 100)
# a_prime = a[0:to_index]
# st.write(a_prime.drop(columns=['cum_gain_loss_100']).head(to_index))
# draw_charts(a_prime)

from_date = '2020-07-01'
st.write(f'Starting: {from_date}')
a = calc_with_params(from_date, 100)
a_prime = a[0:to_index]
# st.write(a_prime.head(to_index))
st.write(a_prime.drop(columns=['cum_gain_loss_100']).head(to_index))
draw_charts(a_prime)

from_date = '2011-07-01'
st.write(f'Starting: {from_date}')
a = calc_with_params(from_date, 100)
a_prime = a[0:to_index]
# st.write(a_prime.head(to_index))
st.write(a_prime.drop(columns=['cum_gain_loss_100']).head(to_index))
draw_charts(a_prime)

from_date = '2011-07-01'
st.write(f'Starting: {from_date}')
a = calc_with_params(from_date, 100)
a_prime = a[0:5]
# st.write(a_prime.head(to_index))
st.write(a_prime.drop(columns=['cum_gain_loss_100']).head(to_index))
draw_charts(a_prime)

# st.write(a_prime.iloc[0]['cum_gain_loss_100'])


# from_date = '2022-01-01'
# st.write(f'Starting: {from_date}')
# a = calc_with_params(from_date, 100)
# st.write(a.head())

# with st.form("my_form"):

#     from_date = st.text_input('from date YYYY-MM-DD')

#     # Every form must have a submit button.
#     submitted = st.form_submit_button("Submit")
#     if submitted:
#         st.write("from_date", from_date, "invest_amount", 100)
#         user_result = calc_with_params(from_date, 100)
#         st.write(user_result.head())
