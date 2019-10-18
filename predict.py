#!/usr/bin/env python
# coding: utf-8

# In[94]:


"""You can define global variables here """

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import pandas as pd

np.set_printoptions(suppress=True, formatter={'float_kind':'{:16.3f}'.format}, linewidth=80)

global model # We want to use the same model throughout the interface
global company_num
correct_count = 0
days_back = 100
verbose = False


def start( company, name ):
    """Called once at the beginning of testing period.
        Company is the number of the company you are predicting (0-2), or 3 if you are to
        predict the value of all companies.
        Name is the (path and) name of your model.
        You should load the model you use throughout here (no training to take place) """
    print("Starting model of company %d" % company)
    file_path = "models/" + name
    print("Loading model from \"" + file_path + "\"")
    from keras.models import load_model
    global model 
    global company_num
    company_num = company
    model = load_model(file_path)
    return


def predict( stock_prices, market_analysis, market_segments, info ):
    """Called on consecutive days of the test period (when stock markets are open).
       The arguments are provided as lists of tuples, ordered in an increasing
       order by date (i.e., the last tuple in stock_prices and info, are for the day you are
       predicting. You are not allowed to change the arguments.

       You are to predict the stock-price movement for the date indicated by the last record in
       the info list. Note, that the lists also contain information about older dates, but you are
       free to ignore them if not using them.

       Returns a 3-list of predictions (company 0, 1, 2, respectively). Each value is True if
       prediction is that stock price for respective company will go up, otherwise False
       (i.e. if you think company 0 will go up, but not 1 and 2, then return [True, False, False]).
       If you are only predicting a single company, set the respective field but keep the others False
       (i.e. if company is number 1 and you predict its stock will go up, then return [False, True, False]).
    """
    prediction = [False, False, False]
    correct = [False, False, False]
    global days_back
    
    # If it is a single company we run this...
    if company_num != 3:
        # Getting the info of today
        info_today = info[len(info) - 1]
    
        # Getting stock_price of today
        stock_today = stock_prices[len(stock_prices) - 1]
        stock_yesterday = stock_prices[len(stock_prices) - 2]
        
        # Get the info we use in our model
        X = info_today[4:]
        
        if company_num != 1:
            # IT = [1,0]
            X = np.hstack((X, [1, 0]))
        else:
            X = np.hstack((X, [0, 1]))
        
        # Make the stock price binary based on yesterday
        if stock_today[4] > stock_yesterday[4]:
            X = np.hstack((X, [1]))
            correct[company_num] = True
        else:
            X = np.hstack((X, [0]))
        if verbose: print("Should be: ", X[-1])
        
        # Load the scaler used to normalize the training data
        scaler = joblib.load("models/scaler Company " + str(company_num) + ".save")
        X = X.reshape(1, -1)
        X_test = scaler.transform(X)
        X_test = np.array(X_test[0][:-1])
        
        # Prediction time!
        predict = model.predict_classes(X_test.reshape(1,-1))
        if predict[0][0] == 1:
            prediction[company_num] = True
        
    # Now for the combination!
    elif company_num == 3:
        # Company 0
        info_today_0 = info[len(info) - 3]
        X = info_today_0[4:]
        X = np.hstack((X, [1, 0]))
        stock_today_0 = stock_prices[len(stock_prices) - 3]
        stock_yesterday_0 = stock_prices[len(stock_prices) - 6]
        
        # Company 1
        info_today_1 = info[len(info) - 2][4:]
        X = np.hstack((X, info_today_1))
        X = np.hstack((X, [0, 1]))
        stock_today_1 = stock_prices[len(stock_prices) - 2]
        stock_yesterday_1 = stock_prices[len(stock_prices) - 5]
        
        # Company 2
        info_today_2 = info[len(info) - 1][4:]
        X = np.hstack((X, info_today_2))
        X = np.hstack((X, [0, 1]))
        stock_today_2 = stock_prices[len(stock_prices) - 1]
        stock_yesterday_2 = stock_prices[len(stock_prices) - 4]
        
        # Putting it all together
        all_info = [info_today_1,info_today_2]
        all_stock = [stock_today_0,stock_today_1,stock_today_2]
        all_stock_y = [stock_yesterday_0, stock_yesterday_1, stock_yesterday_2]
            
        for i in range(len(all_stock)):
            if all_stock[i][4] > all_stock_y[i][4]:
                X = np.hstack((X, [1]))
                correct[i] = True
            else:
                X = np.hstack((X, [0]))
            if verbose: print(str(i) + " should be: ", X[-1])
                
        scaler = joblib.load("models/scaler Company 13.save")
        X = X.reshape(1, -1)
        X_test = scaler.transform(X)
        X_test = np.array(X_test[0][:-3])
        
        prob = model.predict(X_test.reshape(1,-1))
        for i in range(3):
            if prob[0][i] > 0.5:
                prediction[i] = True
                
    global correct_count
    if prediction == correct:
        correct_count += 1
        
    return prediction

def end():
    """Called once at the end of the testing period. Optional if you do anything here."""
    ...
    return

def my_little_tester(module ):
    info_data = pd.read_csv("data/info.txt", sep='\s+')
    market_analysis = pd.read_csv("data/market_analysis.txt", sep='\s+')
    market_segments = pd.read_csv("data/market_segments.txt", sep='\s+')
    stock_prices = pd.read_csv("data/stock_prices.txt", sep='\s+')
    
    info_0 = info_data[info_data['company'] == 0]
    info_0 = info_0[-days_back:]
    info_0 = [tuple(x) for x in info_0.values]
    stock_prices_0 = stock_prices[stock_prices['company'] == 0]
    stock_prices_0 = stock_prices_0[-days_back:]
    stock_prices_0 = [tuple(x) for x in stock_prices_0.values]
    
    info_1 = info_data[info_data['company'] == 1]
    info_1 = info_1[-days_back:]
    info_1 = [tuple(x) for x in info_1.values]
    stock_prices_1 = stock_prices[stock_prices['company'] == 1]
    stock_prices_1 = stock_prices_1[-days_back:]
    stock_prices_1 = [tuple(x) for x in stock_prices_1.values]
    
    info_2 = info_data[info_data['company'] == 2]
    info_2 = info_2[-days_back:]
    info_2 = [tuple(x) for x in info_2.values]
    stock_prices_2 = stock_prices[stock_prices['company'] == 2]
    stock_prices_2 = stock_prices_2[-days_back:]
    stock_prices_2 = [tuple(x) for x in stock_prices_2.values]
    
    
    
    name_list = ["c1.h5", "c2.h5", "c3.h5", "c13.h5"]
    start(module, name_list[module])
    daily_stock = []
    daily_info = []
    if module != 3:
        for i in range(0,days_back):
            day = -days_back + i
            daily_stock.append(stock_prices_0[day])
            daily_info.append(info_0[day])
            prediction = predict( daily_stock, 0, 0, daily_info )
            if verbose: print(prediction)
            if verbose: print("-"*15)
    elif module == 3:
        for i in range(0,days_back):
            day = -days_back + i
            daily_stock.append(stock_prices_0[day])
            daily_info.append(info_0[day])
            daily_stock.append(stock_prices_1[day])
            daily_info.append(info_1[day])
            daily_stock.append(stock_prices_2[day])
            daily_info.append(info_2[day])
            prediction = predict( daily_stock, 0, 0, daily_info )
            if verbose: print(prediction)
            if verbose: print("-"*15)


def main():
    global correct_count
    
    my_little_tester(0)
    
    print(correct_count, "/", days_back)
    correct_count = 0
    
    my_little_tester(1)
    
    print(correct_count, "/", days_back)
    correct_count = 0
    
    my_little_tester(2)
    
    print(correct_count, "/", days_back)
    correct_count = 0
    
    my_little_tester(3)
    print(correct_count, "/", days_back)
    correct_count = 0


main()

