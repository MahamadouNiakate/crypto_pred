import streamlit as st

import pandas as pd
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

  df = df.merge(month_close_df, how='left', left_index=True, right_index=True)

  df['gain_loss'] = None
  df['gain_loss_percent'] = None
  for i in range(1, len(df)):

    df['gain_loss'][i] = df['month_close'][i] - df['month_close'][i-1]
    df['gain_loss_percent'][i] = ((df['month_close'][i] - df['month_close'][i-1])/((df['month_close'][i]+df['month_close'][i-1])/2))

    # df['month_close'][i]/((df['month_close'][i] - df['month_close'][i-1]))
  df.dropna(inplace=True)
  return df

def calc_gain_loss(df, threshold, column, debug=False):
  print(f'======calc_gain_loss')
  gain_loss = []
  for index, row in df.iterrows():
    if row[column] > threshold:
      if debug:
        print(f'{index}::{row[column]}')

      # gain_loss.append(row['Close_shift'] - row['Close'])
      gain_loss.append(row['gain_loss'])

  if debug:
    print(f'GAIN/LOSS: {sum(gain_loss)}')

  return sum(gain_loss)

def calc_gain_loss_2(df, threshold, column, investment_amount, debug=False):
  print(f'======calc_gain_loss_2')

  gain_loss = []

  for index, row in df.iterrows():
    if row[column] > threshold:
      if debug:
        print(f'{index}::{row[column]}::{row["gain_loss_percent"]}')

      # gain_loss.append(row['Close_shift'] - row['Close'])
      gain_loss.append(investment_amount * row['gain_loss_percent'])

  if debug:
    print(f'GAIN/LOSS: {sum(gain_loss)}')

  return sum(gain_loss)

def standard_lr(test, X_train, y_train, X_test, y_test):

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    X_test_predict = lr.predict(X_test)

    X_test_predict_df = pd.DataFrame(X_test_predict, index=X_test.index, columns=['Y_PRED'])
    predict_df = test.merge(X_test_predict_df, how='left', on='Date')


    #dbd todo: Plot chart of actual v/s prredicted movements
    # predict_df['month_ave_close_shift_diff_percent'].plot(c='red')
    # predict_df['Y_PRED'].plot(c='green')

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

  window_sizes = [3, 6, 12]
  gain_percents = [0, .05, .1]
  model_list = [('lr', LinearRegression()),
                # ('xgb', xgb.XGBRegressor(objective ='reg:squarederror')),
                ('knn', KNeighborsRegressor(n_neighbors=2)),
                ('rf', RandomForestRegressor()) ]

  window_df = btc_month_mean.copy()[start_date:]

  result_df = pd.DataFrame(columns=['window', 'percent', 'model_name', 'gain_loss', f'gain_loss_{bet_size}'])

  for window_size in window_sizes:
    for gain_percent in gain_percents:
      for model_name, model in model_list:
        # print(f'{window_size}:{gain_percent}:{model_name}')
        test_df = calc_sliding_window2(model_name, model, window_size, window_df, False)

        test_df = append_gain_loss(month_close_df, test_df)

        gain_loss = calc_gain_loss(test_df, gain_percent, 'Y_PRED')
        gain_loss_bet_size = calc_gain_loss_2(test_df, gain_percent, 'Y_PRED', bet_size, False)

        loop_df = pd.DataFrame(data=[[window_size, gain_percent, model_name, gain_loss, gain_loss_bet_size]], columns=['window', 'percent', 'model_name', 'gain_loss', f'gain_loss_{bet_size}'])

        result_df = pd.concat([result_df, loop_df], ignore_index=True)
  return result_df

'''------------------------------------------------'''

st.write('DEFINE month_ave_close_shift_diff_percent')
'''

seasonal_decompose(btc_month_mean['month_ave_close_shift_diff_percent']).plot()
print("Dickey–Fuller test: p=%f" % adfuller(btc_month_mean['month_ave_close_shift_diff_percent'])[1])

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(btc_month_mean['month_ave_close_shift_diff_percent'], lags=24);
plot_pacf(btc_month_mean['month_ave_close_shift_diff_percent'], lags=24);

'''
# Plain Linear Regression
st.write(f'Starting: {start_date_string}')

btc_month_mean, month_close_df, test, X_train, y_train, X_test, y_test = preprocess(btc)
predict_df = standard_lr(test, X_train, y_train, X_test, y_test)
predict_df = append_gain_loss(month_close_df, predict_df)



#dbd todo
# seasonal_decompose(btc_month_mean['month_ave_close_shift_diff_percent']).plot()
# print("Dickey–Fuller test: p=%f" % adfuller(btc_month_mean['month_ave_close_shift_diff_percent'])[1])

# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# plot_acf(btc_month_mean['month_ave_close_shift_diff_percent'], lags=24);
# plot_pacf(btc_month_mean['month_ave_close_shift_diff_percent'], lags=24);


thresholds = [0, .1, .2]
results = []
for threshold in thresholds:
  gain_loss = calc_gain_loss(predict_df, threshold, 'Y_PRED')
  gain_loss_100 = calc_gain_loss_2(predict_df, threshold, 'Y_PRED', 100, False)
  results.append({'threshold':threshold, 'gain_loss':gain_loss, 'gain_loss_100':gain_loss_100})

static_results_df = pd.DataFrame(data=results, columns=['threshold', 'gain_loss', 'gain_loss_100'])
static_results_df = static_results_df[['threshold', 'gain_loss_100']]
static_results_df = static_results_df.sort_values('gain_loss_100', ascending=False)

st.write(static_results_df.head())

'''------------------------------------------------'''

st.write(f'Starting: {start_date_string}')

# Linear regression with sliding window
window_sizes = [3, 6, 12]
gain_percents = [0, .05, .1, .2]

result_df = pd.DataFrame(columns=['window', 'percent', 'gain_loss'])

bet_size = 100
for window_size in window_sizes:
  for gain_percent in gain_percents:

    test_df = calc_sliding_window(window_size, btc_month_mean, False)
    test_df = append_gain_loss(month_close_df, test_df)

    gain_loss = calc_gain_loss(test_df, gain_percent, 'Y_PRED')
    gain_loss_bet_size = calc_gain_loss_2(test_df, gain_percent, 'Y_PRED', bet_size, False)

    loop_df = pd.DataFrame(data=[[window_size, gain_percent, gain_loss, gain_loss_bet_size]], columns=['window', 'percent', 'gain_loss', f'gain_loss_{bet_size}'])
    result_df = pd.concat([result_df, loop_df], ignore_index=True)

sorted_result_df = result_df.sort_values('gain_loss_100', ascending=False)
sorted_result_df = sorted_result_df.assign(hack='').set_index('hack')
sorted_result_df = sorted_result_df[['window', 'percent', 'gain_loss_100']]

st.write(sorted_result_df.head())

'''------------------------------------------------'''
# Multiple regression techniques with sliding window

# st.write('Starting: 2020-01-01')
# a = calc_with_params('2020-01-01', 100)
# a = a[['window', 'percent', 'model_name', 'gain_loss_100']]
# a.sort_values(by='gain_loss_100', ascending=False)
# a = a.assign(hack='').set_index('hack')
# st.write(a.head())

# st.write('Starting: 2021-01-01')
# b = calc_with_params('2021-01-01', 100)
# b = b[['window', 'percent', 'model_name', 'gain_loss_100']]
# b.sort_values(by='gain_loss_100', ascending=False).head()
# b = b.assign(hack='').set_index('hack')
# st.write(b.head())

# st.write('Starting: 2022-01-01')
# c = calc_with_params('2022-01-01', 100)
# c = c[['window', 'percent', 'model_name', 'gain_loss_100']]
# c.sort_values(by='gain_loss_100', ascending=False).head()
# c = c.assign(hack='').set_index('hack')
# st.write(c.head())


with st.form("my_form"):
    st.write("Inside the form")
    # slider_val = st.slider("Form slider")
    # checkbox_val = st.checkbox("Form checkbox")


    from_date = st.text_input('from date YYYY-MM-DD')
    invest_amount = st.text_input('Invest Amount')

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("from_date", from_date, "invest_amount", invest_amount)
        user_result = calc_with_params(from_date, int(invest_amount))
        user_result = user_result[['window', 'percent', 'model_name', 'gain_loss_100']]
        user_result.sort_values(by='gain_loss_100', ascending=False).head()
        user_result = user_result.assign(hack='').set_index('hack')
        st.write(user_result.head())

st.write("Outside the form")
