
"""You can define global variables here """

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

def start( company, name ):
    """Called once at the beginning of testing period.
        Company is the number of the company you are predicting (0-2), or 3 if you are to
        predict the value of all companies.
        Name is the (path and) name of your model.
        You should load the model you use throughout here (no training to take place) """
    print("Starting model of company %d" % company)
    file_path = "models/" + name
    print("Loading model from \"" + file_path + "\"")

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
    ...
    prediction = [False, False, False]
    return prediction

def end():
    """Called once at the end of the testing period. Optional if you do anything here."""
    ...
    return


def main():
    start(0, "c1.h5")

if __name__ == "__main__":
    main()
