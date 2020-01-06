import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("sphist.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] > datetime(year=2015, month=4, day=1)
df = df.sort_values(by=['Date'], ascending=True)

# Gerating indicators
df['5 Days Open'] = df['Open'].rolling(center=False, window=5).mean()
df['5 Days High'] = df['High'].rolling(center=False, window=5).mean()
df['5 Days Low'] = df['Low'].rolling(center=False, window=5).mean()
df['5 Days Volume'] = df['Volume'].rolling(center=False, window=5).mean()
df['Year'] = df['Date'].apply(lambda x: x.year)

# Shift the column by one
df['5 Days Open'] = df['5 Days Open'].shift(periods=1)
df['5 Days High'] = df['5 Days High'].shift(periods=1)
df['5 Days Low'] = df['5 Days Low'].shift(periods=1)
df['5 Days Volume'] = df['5 Days Volume'].shift(periods=1)

df = df[df['Date'] > datetime(year=1951, month=1, day=3)]
df = df.dropna(axis=0)

train = df[df['Date'] < datetime(year=2013, month=1, day=1)]
test = df[df['Date'] >= datetime(year=2013, month=1, day=1)]

lr = LinearRegression()
features = ['5 Days Open', '5 Days Volume', '5 Days High', '5 Days Low', 'Year']
lr.fit(train[features], train['Close'])
predictions_test = lr.predict(test[features])

mae = mean_absolute_error(predictions_test, test['Close'])

print(mae)
