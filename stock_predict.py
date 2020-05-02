# https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

# url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
# url = 'https://github.com/varun-v2021/ML-SampleData/blob/master/TATAMOTORS-history.csv'
url = 'https://raw.githubusercontent.com/varun-v2021/ML-SampleData/master/TATAMOTORS-history.csv'
dataset_train = pd.read_csv(url)
# get 'Open' column from the dataset into training set
training_set = dataset_train.iloc[:, 1:2].values
print('DATASET TO TRAIN')
print(dataset_train.head())
print('TRAINING SET')
print(training_set)
sc = MinMaxScaler(feature_range=(0,1))
# training_set_scaled will have multiple rows which have been transformed for values between 0 & 1
# having single column. (as we have iloc[i, 1:2] reads second column (1) till third column 2 without including
# '2' column)
training_set_scaled = sc.fit_transform(training_set)

X_train = []
Y_train = []
for i in range(50,2542 ): #60,2035 for tataglobal (total no of rows in historic data)
    # slicing training_set_scaled
    # training_set_scaled[start_row_index:end_row_index,column index]
    # X_train will become 1D array of arrays of rows under single column 0
    # value of array of ([array of 0 to 50 rows], [array 1 to 51 rows], ...)
    # X_train will contain data in timesteps
    X_train.append(training_set_scaled[i-50:i, 0]) #60
    # Y_train will become 1D array indexed under single column 0
    # Y_train will have array of values of 50, 51, 52 ... 2541 of training_set_scaled
    Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)
# shape[0] gives no of rows and shape[1] gives no of columns
# 2D to 3D - 3D array is required for LSTM modelling
# The reshape function can be used directly, specifying the new dimensionality.
# This is clear with an example where each sequence has multiple time steps with
# one observation (feature) at each time step.
# We can use the sizes in the shape attribute on the array to specify the number of samples (rows)
# and columns (time steps) and fix the number of features at 1.
# shape[0] - rows, shape[1] - columns
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True)) #50
model.add(Dropout(0.2))

model.add(LSTM(units=60,return_sequences=True))#50
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(X_train,Y_train,epochs=100,batch_size=32) #epochs=100
pickle.dump(model, open("stock-model.pkl", "wb"))

# url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv'
# url = 'https://github.com/varun-v2021/ML-SampleData/blob/master/TATAMOTORS-quarterly.csv'
url = 'https://raw.githubusercontent.com/varun-v2021/ML-SampleData/master/TATAMOTORS-monthly.csv'
dataset_test = pd.read_csv(url)
# get 'Open' column from the dataset like the training set
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 50:].values #60
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(50, 76): #60,76
    X_test.append(inputs[i-50:i, 0]) #60

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'black', label = 'TATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()