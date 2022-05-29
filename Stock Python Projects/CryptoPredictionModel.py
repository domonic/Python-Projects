from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

import os

asset = 'BTC'
cash = 'USD'

startDate = dt.datetime(2021, 5, 28)
endDate = dt.datetime.now()

data = web.DataReader(f'{asset}-{cash}', 'yahoo', startDate, endDate)

# Data Preparation
scaler = MinMaxScaler(feature_range=(0,1))
scaledData = scaler.fit_transform(data['Close'].values.reshape(-1,1))

predictionDays = 60
futureDays = 30

x_train, y_train = [], []

for x in range(predictionDays, len(scaledData)-futureDays):
    x_train.append(scaledData[x-predictionDays:x, 0])
    y_train.append(scaledData[x+futureDays, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Neural Network Creation
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Model Testing

testStart = dt.datetime(2021, 5, 28)
testEnd = dt.datetime.now()

testData = web.DataReader(f'{asset}-{cash}', 'yahoo', testStart, testEnd)
actualPrice = testData['Close'].values

totalDataset = pd.concat((data['Close'], testData['Close']), axis=0)

modelInputs = totalDataset[len(totalDataset) - len(testData) - predictionDays:].values
modelInputs = modelInputs.reshape(-1, 1)
modelInputs = scaler.fit_transform(modelInputs)

x_test = []

for x in range(predictionDays, len(modelInputs)):
    x_test.append(modelInputs[x-predictionDays:x, 0])


x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


predictedPrice = model.predict(x_test)
predictedPrice = scaler.inverse_transform(predictedPrice)

plt.plot(actualPrice, color='Red', label='Actual Price')
plt.plot(predictedPrice, color='Green', label='Predicted Price')
plt.title(f'{asset} Price Prediction')
plt.xlabel(f'BEGIN DATE {startDate} - END DATE {endDate}')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()

#Following Day Prediction
realData = [modelInputs[len(modelInputs) + futureDays - predictionDays:len(modelInputs) + futureDays, 0]]
realData = np.array(realData)
realData = np.reshape(realData, (realData.shape[0], realData.shape[1], 1))


prediction = model.predict(realData)
prediction = scaler.inverse_transform(prediction)

os.system('clear')
print(f'In {futureDays} days {asset} predicted price is: ${prediction[0][0]}')









