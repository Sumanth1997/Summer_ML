'''About: The code effectively builds and evaluates a stock price prediction model for Tesla using a Random Forest classifier. It demonstrates key machine learning steps, including data preprocessing, feature engineering, model training, and evaluation. The use of rolling windows for backtesting allows the model to simulate real-time prediction performance. Enhanced feature engineering, including rolling averages and trend indicators, aims to improve the model's predictive power. This project provides a practical application of machine learning in financial markets, showcasing techniques for handling time-series data and evaluating model performance through precision scores and prediction analysis.'''

import pandas as pd

# Load data
data = pd.read_csv('Projects/Tesla_Stock_Price/Tesla Dataset.csv')

# Convert date column to datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)


# print(data.head())

import matplotlib.pyplot as plt

# Plot closing prices
# plt.figure(figsize=(10, 6))
# plt.plot(data['Close'])
# plt.title('Tesla Closing Prices')
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.show()
pd.set_option('display.max_columns', None)

data['Tomorrow'] = data['Close'].shift(-1)

# print(data.head())


data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)

# print(data.head())

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = data.iloc[:-100]
test = data.iloc[-100:]

predictors = ['Close','Volume','Open','High','Low']
# model.fit(train[predictors] ,train['Target'])

from sklearn.metrics import precision_score

# y_pred = model.predict(test[predictors])

# y_pred = pd.Series(y_pred,index = test.index)

# print(precision_score(test['Target'],y_pred))


# combined = pd.concat([test['Target'],y_pred],axis=1)

# combined.plot()
# plt.show()

def prediction(train,test,predictors,model):
    model.fit(train[predictors],train['Target'])
    y_pred = model.predict(test[predictors])
    y_pred = pd.Series(y_pred,index = test.index,name='Predictions')
    combined = pd.concat([test['Target'],y_pred],axis=1)
    return combined

def backtest(data,model,predictors,start=500,step=250):
    all_predictions = []

    for i in range(start,data.shape[0],step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = prediction(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(data,model,predictors)

print(predictions['Predictions'].value_counts())

print(precision_score(predictions['Target'],predictions['Predictions']))

print(predictions['Target'].value_counts()/predictions.shape[0])

horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = data.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    data[ratio_column] = data['Close'] / rolling_averages['Close']

    trend_column = f"Trend_{horizon}"
    data[trend_column] = data.shift(1).rolling(horizon).sum()['Target']

    new_predictors += [ratio_column,trend_column]

# print(data)

data = data.dropna()

model = RandomForestClassifier(n_estimators=200,min_samples_split=50,random_state=1)

def prediction_enhanced(train,test,predictors,model):
    model.fit(train[predictors],train['Target'])
    y_pred = model.predict_proba(test[predictors])[:,1]
    y_pred[y_pred >= .6] =1
    y_pred[y_pred < .6] = 0
    y_pred = pd.Series(y_pred,index = test.index,name='Predictions')
    combined = pd.concat([test['Target'],y_pred],axis=1)
    return combined


def backtest_enhanced(data,model,predictors,start=500,step=250):
    all_predictions = []

    for i in range(start,data.shape[0],step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = prediction_enhanced(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest_enhanced(data,model,new_predictors)

print(predictions['Predictions'].value_counts())

print(precision_score(predictions['Target'],predictions['Predictions']))