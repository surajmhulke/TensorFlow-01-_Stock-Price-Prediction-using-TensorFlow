# TensorFlow-01-_Stock-Price-Prediction-using-TensorFlow


## Introduction
In this project, we will build a Stock Price Prediction model using TensorFlow. Stock market price analysis is a time-series forecasting task, and we will use a Recurrent Neural Network (RNN) to perform this prediction. TensorFlow, an open-source machine learning framework, will be our primary tool to develop this model. TensorFlow simplifies the process of building neural networks, allowing us to implement complex functionalities with just a few lines of code.

## Table of Contents
- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Importing Dataset](#importing-dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development and Evaluation](#model-development-and-evaluation)
- [Conclusion](#conclusion)

## Importing Libraries
We begin by importing the necessary Python libraries to help us with data handling, visualization, and building the predictive model. These libraries include:
- Pandas: For data manipulation and analysis.
- NumPy: To work with efficient numerical arrays.
- Matplotlib/Seaborn: For data visualization.
- TensorFlow: Our primary tool for building machine learning models.
- Other utilities for managing date data and suppressing warnings.

 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

Importing Dataset

We load the stock price dataset into a Pandas DataFrame. You can download the dataset from here. The dataset contains stock price information over a five-year period.

 

data = pd.read_csv('./s_p_stock/all_stocks_5yr.csv')
print(data.shape)
print(data.sample(7))

Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is a crucial step in understanding the dataset. We begin by converting date data to the DateTime data type. Then, we perform various visualizations to understand the stock price trends of different companies.

 

data['date'] = pd.to_datetime(data['date'])

# Visualize open and close stock prices for selected companies
# ... (code for plotting stock prices)

Feature Engineering

We prepare the data for training by applying feature scaling and creating training features and labels (x_train and y_train). This involves using a rolling window of data to create sequences for training.

 

# ... (code for feature scaling and preparing training data)

Model Development and Evaluation

We create a Gated RNN-LSTM network using TensorFlow. LSTM is used for sequence modeling and time series data. We compile the model and train it using historical stock price data. We also evaluate the model's performance using testing data.

 

# ... (code for building and training the model)

Conclusion

In this project, we successfully built a Stock Price Prediction model using TensorFlow and LSTM. We loaded the dataset, performed EDA to understand the data, engineered features, and developed the model. The model's performance was evaluated, and predictions were made for future stock prices.
