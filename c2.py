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

print(tf.__version__)
pd.set_option('mode.chained_assignment', None)

# # Deep Learning
# ## Assignment 1 - Beat the market
# ### Abel de Wit & Malin Hjärtström
#

# Getting the data (commented for local use)
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

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

company_1 = dataframe[dataframe['company'] == 1]

# Let's see how their stocks are doing
company_1.plot(y='stock-price').set_title('Company 1')

del company_1['company']
del company_1['quarter']

# We want to predict wether the stock goes up or not
# so we have to change the stock price values in such a way that it is binary.
company_1['stock-price-binary'] = np.where(company_1['stock-price'] > company_1['stock-price'].shift(), 1, 0)
del company_1['stock-price']


# Model definition
# So now we have the data in a nice table, split into seperate companies, we can do some machine learning!
def train_company(company, name):
    # Scale data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(company)
    scaled = pd.DataFrame(data=scaled, columns=company.columns)
    
    scaler_filename = "models/scaler " + name + ".save"
    joblib.dump(scaler, scaler_filename) 

    X = scaled.loc[:, scaled.columns != 'stock-price-binary']
    y = scaled['stock-price-binary']
    
    num_features = X.shape[1]
    # print(num_features)

    # Now we split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    
    # Create a model.
    model = Sequential()
    model.add(Dense(64, input_dim=num_features, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, input_dim=num_features, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model.summary()

    # Tensorboard stuff
    log_dir = os.path.join(
        "logs",
        name,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir, histogram_freq=1)

    # Train the model
    model.fit(x=X_train, 
              y=y_train, 
              epochs=100,
              batch_size = 50,
              shuffle=False,
              validation_data=(X_val, y_val),
              callbacks=[tensorboard_callback],
              verbose=0)
    return model, X_test, y_test


model_company, X_test, y_test = train_company(company_1, "Company 1")
y_pred = model_company.predict_classes(X_test)
y_pred = y_pred[:, 0]
    
    
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_pred)
print("-"*20)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)
print("-"*20)
model_company.save('models/c2.h5')




