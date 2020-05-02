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
training_set = dataset_train.iloc[:, 1:2].values
print(dataset_train.head())
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
Y_train = []
for i in range(50,2542 ): #60,2035 for tataglobal
    X_train.append(training_set_scaled[i-50:i, 0]) #60
    Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)
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
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(50, 90): #60,76
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