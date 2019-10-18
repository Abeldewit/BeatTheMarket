#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import datetime, os
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, LSTM
from keras import regularizers
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

info_data = pd.read_csv("data/info.txt", sep='\s+')
market_analysis = pd.read_csv("data/market_analysis.txt", sep='\s+')
market_segments = pd.read_csv("data/market_segments.txt", sep='\s+')
stock_prices = pd.read_csv("data/stock_prices.txt", sep='\s+')

# Do something with all our data so we can feed it to the NN
dataframe = info_data
dataframe["stock-price"] = stock_prices['stock-price']

# 'One hot encoding' the segments
dataframe["IT"] = dataframe['company'].apply(lambda x: 0 if x == 1 else 1)
dataframe["BIO"] = dataframe['company'].apply(lambda x: 1 if x == 1 else 0)

# Setting the indexes as the date
dataframe.set_index(['year', 'day'], inplace=True)

company_0 = dataframe[dataframe['company'] == 0]
company_1 = dataframe[dataframe['company'] == 1]
company_2 = dataframe[dataframe['company'] == 2]

del company_0['company']
del company_0['quarter']
del company_1['company']
del company_1['quarter']
del company_2['company']
del company_2['quarter']

company_0['stock-price-binary'] = np.where(company_0['stock-price'] > company_0['stock-price'].shift(), 1, 0)
del company_0['stock-price']
company_1['stock-price-binary'] = np.where(company_1['stock-price'] > company_1['stock-price'].shift(), 1, 0)
del company_1['stock-price']
company_2['stock-price-binary'] = np.where(company_2['stock-price'] > company_2['stock-price'].shift(), 1, 0)
del company_2['stock-price']

stock_prices = pd.DataFrame()

stock_prices['spb_0'] = company_0['stock-price-binary']
del company_0['stock-price-binary']

stock_prices['spb_1'] = company_1['stock-price-binary']
del company_1['stock-price-binary']

stock_prices['spb_2'] = company_2['stock-price-binary']
del company_2['stock-price-binary']

big_dataframe = pd.concat([company_0, company_1], axis=1)
big_dataframe = pd.concat([big_dataframe, company_2], axis=1)
big_dataframe = pd.concat([big_dataframe, stock_prices], axis=1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(big_dataframe)
scaled = pd.DataFrame(data=scaled, columns=big_dataframe.columns)

scaler_filename = "models/scaler Company 13.save"
joblib.dump(scaler, scaler_filename)

X = scaled.iloc[:, :-3]
y = scaled.iloc[:, -3:]

# Now we split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

num_features = X.shape[1]

# Create a model.
model = Sequential()
model.add(Dense(128, input_dim=num_features, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.summary()

# Tensorboard stuff
log_dir = os.path.join(
    "logs",
    "Company 1-3",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir, histogram_freq=1)

# Train the model
model.fit(x=X_train,
          y=y_train,
          epochs=100,
          batch_size=50,
          shuffle=False,
          validation_data=(X_val, y_val),
          callbacks=[tensorboard_callback],
          verbose=1)

model.save('models/c13.h5')

y_pred = model.predict(X_test)
print(X_test)

for r in range(y_pred.shape[0]):
    for c in range(y_pred.shape[1]):
        if y_pred[r, c] > 0.5:
            y_pred[r, c] = 1
        else:
            y_pred[r, c] = 0

print(y_pred)
