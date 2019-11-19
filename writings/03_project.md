# Stock Price Indicator

## Project Overview


Investment firms, hedge funds and even individuals have been using financial models to better understand market behaviour and make profitable investments and trades. A wealth of information is available in the form of historical stock prices and company performance data, suitable for machine learning algorithms to process.


Here, I build a stock price predictor that takes daily trading data over a certain date range as input, and outputs projected estimates for given query dates. Inputs contain multiple metrics:
 * opening price (Open)
 * highest price the stock traded at (High)
 * how many stocks were traded (Volume)
Whereas the prediction is made on:
 * closing price adjusted for stock splits and dividends (Adjusted Close)

We want to explore, where possible, S&P500 companies. The list of the listed companies is taken from [here](https://datahub.io/core/s-and-p-500-companies). 
The trading data comes from [Quandl End-of-Day US stock prices](https://www.quandl.com/data/EOD-End-of-Day-US-Stock-Prices) and it is downloaded through their [API](https://www.quandl.com/data/EOD-End-of-Day-US-Stock-Prices/usage/quickstart/api) and stored locally. Details are provided into the `Data Exploration` section below.

#### Project Format and Features
The project is developed by referring to the cross industry standard process for data mining (CRISP-DM) methodology. Here we summarise the steps undertaken:

- Business Understanding
- Data Understanding
- Data Preparation
- Modelling
- Evaluation
- Deployment

The project is developed in the form of a simple script.
<br/>


## Problem Statement

The aim of the project is to predict stock's adjusted closing prices at 1, 7, 14 and 28 days, by giving stock data a time window of $n$ precedent days. The model should work for highly relevant companies, therefore a few from S&P500 are selected due to data availability.

Data is collected and wrangled to provide training and testing data in the format of features (open, min, max, close, split and dividend) and output vector (the adjusted closing price at the relative time shifts). Data is standardised and a experiments are conducted to test and improve performances.

<br/>

## Metrics

I used Mean Squared Error to measure performance of the model during training. As a reminder, it is the mean of the squared difference between predicted and true values:

$MSE=\frac{1}{n} \Sigma(Y_i - \hat{Y_i})^2$

which is preferred to the mean absolute error due to its tendency to penalise more bigger errors. By doing so, The aim is to drive to model to avoid big errors rather than producing more equally distributed ones.

<br/>

## Requirements

Requirements for the python environment can be found in `<repo_folder>/requirements.yml`.

<br/>

## Analysis

### Data Exploration

Features and calculated statistics relevant to the problem have been reported and discussed related to the dataset, and a thorough description of the input space or input data has been made. Abnormalities or characteristics about the data or input that need to be addressed have been identified.

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividend</th>
      <th>Split</th>
      <th>Adj_Open</th>
      <th>Adj_High</th>
      <th>Adj_Low</th>
      <th>Adj_Close</th>
      <th>Adj_Volume</th>
      <th>Ticker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-12-28</td>
      <td>171.00</td>
      <td>171.850</td>
      <td>170.480</td>
      <td>171.08</td>
      <td>16480187.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>165.971205</td>
      <td>166.796208</td>
      <td>165.466497</td>
      <td>166.048853</td>
      <td>16480187.0</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-12-27</td>
      <td>170.10</td>
      <td>170.780</td>
      <td>169.710</td>
      <td>170.60</td>
      <td>21498213.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>165.097672</td>
      <td>165.757675</td>
      <td>164.719142</td>
      <td>165.582968</td>
      <td>21498213.0</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-12-26</td>
      <td>170.80</td>
      <td>171.470</td>
      <td>169.679</td>
      <td>170.57</td>
      <td>33185536.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>165.777087</td>
      <td>166.427383</td>
      <td>164.689053</td>
      <td>165.553851</td>
      <td>33185536.0</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-12-22</td>
      <td>174.68</td>
      <td>175.424</td>
      <td>174.500</td>
      <td>175.01</td>
      <td>16349444.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>169.542983</td>
      <td>170.265103</td>
      <td>169.368277</td>
      <td>169.863278</td>
      <td>16349444.0</td>
      <td>AAPL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-12-21</td>
      <td>174.17</td>
      <td>176.020</td>
      <td>174.100</td>
      <td>175.01</td>
      <td>20949896.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>169.047981</td>
      <td>170.843576</td>
      <td>168.980040</td>
      <td>169.863278</td>
      <td>20949896.0</td>
      <td>AAPL</td>
    </tr>
  </tbody>
</table>

</div>

```
Date          0
Open          0
High          0
Low           0
Close         0
Volume        0
Dividend      0
Split         0
Adj_Open      0
Adj_High      0
Adj_Low       0
Adj_Close     0
Adj_Volume    0
Ticker        0
```



</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Dividend</th>
      <th>Split</th>
      <th>Adj_Open</th>
      <th>Adj_High</th>
      <th>Adj_Low</th>
      <th>Adj_Close</th>
      <th>Adj_Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>31610</td>
      <td>31610</td>
      <td>31610</td>
      <td>31610</td>
      <td>31610</td>
      <td>31610</td>
      <td>31610</td>
      <td>31610</td>
      <td>31610</td>
      <td>31610</td>
      <td>31610</td>
      <td>31610</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>95.491669</td>
      <td>96.171887</td>
      <td>94.818211</td>
      <td>95.521521</td>
      <td>1.226938e+07</td>
      <td>0.009154</td>
      <td>1.000316</td>
      <td>81.437157</td>
      <td>82.014518</td>
      <td>80.866317</td>
      <td>81.463475</td>
      <td>1.286934e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>59.333158</td>
      <td>59.766797</td>
      <td>58.916946</td>
      <td>59.355129</td>
      <td>1.500732e+07</td>
      <td>0.085350</td>
      <td>0.038147</td>
      <td>43.340742</td>
      <td>43.637986</td>
      <td>43.056648</td>
      <td>43.358424</td>
      <td>1.609089e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.350000</td>
      <td>17.400000</td>
      <td>17.250000</td>
      <td>17.360000</td>
      <td>3.053580e+05</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>16.190719</td>
      <td>16.250819</td>
      <td>16.102600</td>
      <td>16.186081</td>
      <td>3.053580e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>55.372500</td>
      <td>55.870000</td>
      <td>54.980000</td>
      <td>55.395000</td>
      <td>3.847515e+06</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>47.642711</td>
      <td>48.065195</td>
      <td>47.263770</td>
      <td>47.650090</td>
      <td>4.104089e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>87.845000</td>
      <td>88.400000</td>
      <td>87.170000</td>
      <td>87.840000</td>
      <td>7.197633e+06</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>76.139854</td>
      <td>76.664216</td>
      <td>75.628460</td>
      <td>76.147526</td>
      <td>7.490536e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>118.560000</td>
      <td>119.290000</td>
      <td>117.707500</td>
      <td>118.580000</td>
      <td>1.526566e+07</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>105.912443</td>
      <td>106.673449</td>
      <td>105.180398</td>
      <td>105.948241</td>
      <td>1.574847e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>649.900000</td>
      <td>651.260000</td>
      <td>644.470000</td>
      <td>647.350000</td>
      <td>6.166205e+08</td>
      <td>3.290000</td>
      <td>7.000000</td>
      <td>286.481931</td>
      <td>286.913264</td>
      <td>284.785353</td>
      <td>285.542583</td>
      <td>6.166205e+08</td>
    </tr>
  </tbody>
</table>

</div>


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>start date</th>
      <th>end date</th>
      <th>difference</th>
    </tr>
    <tr>
      <th>Ticker</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AAPL</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>AXP</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>BA</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>CAT</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>CSCO</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>CVX</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>GE</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>GS</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>HD</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>IBM</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>INTC</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>JNJ</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>JPM</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>KO</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>MCD</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>MMM</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>MRK</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>MSFT</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>NKE</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>PFE</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>PG</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>TRV</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>UNH</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>UTX</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>V</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>VZ</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>WMT</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
    <tr>
      <th>XOM</th>
      <td>2013-09-03</td>
      <td>2017-12-28</td>
      <td>1577 days</td>
    </tr>
  </tbody>
</table>
</div>



### Data Visualization

Build data visualizations to further convey the information associated with your data exploration journey. Ensure that visualizations are appropriate for the data values you are plotting.

![png](output_7_1.png)

### Features Correlation






![png](output_8_1.png)


## Methodology

### Data Pre-processing

Pre-processing steps consist of:
* **scaling (standardization)**: whereas stock prices and dividends are of similar nature (currency), volume and split occur at a different unit.
* **windowing**: as the data comes in the form of a time series, it is required to be stored as an time-related array of features.
* **closing price projection**: we extract the adjusted closing price for each window, at 1, 7, 14 and 28 days.
* **split**: train and test datasets are created in a ratio of 70-30.

### Implementation

Here, a Random Forest Regressor is trained in order to predict a 4-class vector which represent the closing stock price at 1,7,14 and 28 days. Eventually, 4 different models which handle 1-d output might be trained for the scope. Here, grid search is not performed yet, as it might lead to a data leak due to the previous scaling. Mean squared error is used to evaluate the model performances on the training and test sets. <br/>
The first run produces a model whose performances show a clear overfitting (MSE on training is about 6 times lower than the MSE on the test set).

<br/>

### Refinement

To avoid overfitting, I repeat what above, but by building a pipeline object which comprehends a scaler to then perform a grid search to tune the hyper-parameters. Therefore, the windowing is done again, but omitting the scaling.


```python
def preprocess_data(df, window, to_drop=['Date','Ticker'], y_label='Adj_Close', grouper='Ticker'):
    """ Trim data by a given time window, and the associated the adjusted closing price at 1,7,14 and 28 days.
    
    Parameters
    ----------
    df : DataFrame
        data
    
    window : int
        window size (days)
    
    to_drop : array-like
        [optional] column names to drop, default=['Date','Ticker']
    
    y_label : str
        [optional] label to predict, default='Adj_Close'
    
    grouper : string
        [optional] name of the column which represents the company - typically, the ticker, default='Ticker'
    
    Returns
    -------
    X : array
        input tensor for the specificed window. If data is of shape (n,m), X will have shape (n,w,m)
        NOTE: m is the numer of features, but the ones to drop
    
    y : array
        output vector for the adjusted closing price at 1,7,14 and 28 days, of shape=(n,1,4)
    """
    # SCALE
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    # X works on the whole df, minus the non-numeric columns
    scaler_X.fit(df.drop(columns=to_drop,inplace=False))
    # y on a vector shape (4,) (4 times y_label)
    scaler_y.fit(pd.concat([df[y_label]]*4,axis=1))
    
    # WINDOWING
    groups = df.groupby(grouper)
    X = []
    y = []
    
    for n,g in groups:
        # NOTE: we could add a company-related information (e.g. volatility index)
        for i in range(window, len(g)-28):
            g = g.reset_index(drop=True)
            # drop and transform X
            data = scaler_X.transform(g[i-window:i].drop(columns=to_drop,inplace=False))
            # transform y - remember to increase dimensionality of the array
            pred = np.expand_dims(np.array([g.loc[i+1,y_label],g.loc[i+7,y_label],g.loc[i+14,y_label],g.loc[i+28,y_label]]),0)
            pred = scaler_y.transform(pred)
            # done-
            X.append(data)
            y.append(pred)
                 
    return np.array(X),np.array(y),scaler_y

X,y,scaler_y = preprocess_data(df,5)
```


```python
df.shape, X.shape, y.shape
```




    ((31610, 14), (30653, 5, 12), (30653, 1, 4))




```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, shuffle=False)

print()
print("TRAIN Mean Squared Error:",mse_train)
print("TEST  Mean Squared Error:",mse_test)
```

    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=10,
                          n_jobs=None, oob_score=False, random_state=None,
                          verbose=0, warm_start=False)
    
    TRAIN Mean Squared Error: 2.0895968540428473
    TEST  Mean Squared Error: 11.871575319294674






![png](output_13_0.png)



![png](output_13_1.png)



![png](output_13_2.png)



![png](output_13_3.png)


#### Refinement


```python
def window_data(df, window, to_drop=['Date','Ticker'], y_label='Adj_Close', grouper='Ticker'):
    """ Trim data by a given time window, and the associated the adjusted closing price at 1,7,14 and 28 days.
    
    Parameters
    ----------
    df : DataFrame
        data
    
    window : int
        window size (days)
    
    to_drop : array-like
        [optional] column names to drop, default=['Date','Ticker']
    
    y_label : str
        [optional] label to predict, default='Adj_Close'
    
    grouper : string
        [optional] name of the column which represents the company - typically, the ticker, default='Ticker'
    
    Returns
    -------
    X : array
        input tensor for the specificed window. If data is of shape (n,m), X will have shape (n,w,m)
        NOTE: m is the numer of features, but the ones to drop
    
    y : array
        output vector for the adjusted closing price at 1,7,14 and 28 days, of shape=(n,1,4)
    """
```


```python
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, train_size=0.7, random_state=42, shuffle=False)
```


```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', RandomForestRegressor())
])

params = {
    'reg__n_estimators':[10,15,20],
    'reg__max_depth':[8,10,12],
    'reg__min_samples_leaf':[2,3,4],
    'reg__min_samples_split':[4,5,6]
}

cv = GridSearchCV(pipeline, param_grid=params)
cv.fit(
    X_train2.reshape((X_train2.shape[0],X_train2.shape[1]*X_train2.shape[2])),y_train2.reshape(y_train2.shape[0],4)
)

mdl2 = cv.best_estimator_
print(mdl2)

y_pred_train2 = mdl2.predict(
    X_train2.reshape((X_train2.shape[0],X_train2.shape[1]*X_train2.shape[2]))
)
y_pred_test2 = mdl2.predict(
    X_test2.reshape((X_test2.shape[0],X_test2.shape[1]*X_test2.shape[2]))
)

mse_train2 = regression.mean_squared_error(y_train2.reshape(y_train2.shape[0],4),y_pred_train2)
mse_test2  = regression.mean_squared_error(y_test2.reshape(y_test2.shape[0],4),y_pred_test2)

print("TRAIN Mean Squared Error:",mse_train2)
print("TEST  Mean Squared Error:",mse_test2)
```




    Pipeline(memory=None,
             steps=[('scaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('reg',
                     RandomForestRegressor(bootstrap=True, criterion='mse',
                                           max_depth=8, max_features='auto',
                                           max_leaf_nodes=None,
                                           min_impurity_decrease=0.0,
                                           min_impurity_split=None,
                                           min_samples_leaf=3, min_samples_split=6,
                                           min_weight_fraction_leaf=0.0,
                                           n_estimators=20, n_jobs=None,
                                           oob_score=False, random_state=None,
                                           verbose=0, warm_start=False))],
             verbose=False)
    TRAIN Mean Squared Error: 10.253177392106185
    TEST  Mean Squared Error: 9.409037824036808



```python
days=[1,7,14,28]
```


![png](output_18_0.png)



![png](output_18_1.png)



![png](output_18_2.png)



![png](output_18_3.png)



## Results

### Model Evaluation and Validation

The model is evaluated against its ability to predict returns. The returns at 1,7,14 and 28 days are calculated for the truth and the predicted prices. Then, their difference is analysed in order to show the error. More specifically:
* The mean of the difference shows if our model has the tendendcy to produce pessimistic/optimistic prices in the case of a predominant negative or positive error.
* The standard deviation of the difference shows how confident we can be about the model's outcome. It appears that the model requires outstanding improvements before being taken seriously, as we discuss the results below.



```python
y2_pred = mdl2.predict(X2.reshape(X2.shape[0],X2.shape[1]*X2.shape[2]))
```


```python
# Open, High, Low, Close, Volume, Dividend, Split, Adj_Open, Adj_High, Adj_Low, Adj_Close, Adj_Volume
RoR = {}

for i,(d,t) in enumerate(zip(dates,tickers)):
    if not t in RoR.keys():
        RoR[t] = []
    ror = ((X2[i][0][10]-X2[i][0][7])/X2[i][0][7])*100
    RoR[t].append(ror)
```


```python
truth = {}
pred  = {}
diff  = {}
tcks  = {'Ticker':tickers}
dts   = {'Date':dates}

for i in range(4):
    t = 100*(y2[:,0,i]-X2[:,4,7])/X2[:,4,7]
    p = 100*(y2_pred[:,i]-X2[:,4,7])/X2[:,4,7]
    d = p - t
    truth[i] = t
    pred[i]  = p
    diff[i]  = d
```

#### Evaluating Rate on Returns

Here, we store the returns into a dataframe, reporing truth, predicted and differnce (error) values.


```python
# result dataframe
df_results = pd.concat([pd.DataFrame(truth,),#columns=['Tr_1day','Tr_7d','Tr_14d','Tr_28d']
                        pd.DataFrame(pred, ),#columns=['Pr_1day','Pr_7d','Pr_14d','Pr_28d']
                        pd.DataFrame(diff, ),#columns=['Er_1day','Er_7d','Er_14d','Er_28d']
                        pd.DataFrame(tcks),pd.DataFrame(dts)],axis=1)
df_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>Ticker</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.212436</td>
      <td>-0.861228</td>
      <td>-1.332032</td>
      <td>0.618887</td>
      <td>-1.094278</td>
      <td>-2.356069</td>
      <td>-3.602979</td>
      <td>-5.855530</td>
      <td>-1.306714</td>
      <td>-1.494841</td>
      <td>-2.270947</td>
      <td>-6.474417</td>
      <td>AAPL</td>
      <td>2017-12-20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.886373</td>
      <td>-3.145194</td>
      <td>-3.082290</td>
      <td>0.421240</td>
      <td>-1.770366</td>
      <td>-3.097669</td>
      <td>-4.459263</td>
      <td>-6.254812</td>
      <td>-2.656738</td>
      <td>0.047525</td>
      <td>-1.376973</td>
      <td>-6.676052</td>
      <td>AAPL</td>
      <td>2017-12-19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.605610</td>
      <td>-3.262298</td>
      <td>-1.119808</td>
      <td>-0.484625</td>
      <td>-1.862309</td>
      <td>-3.265297</td>
      <td>-4.630666</td>
      <td>-6.472597</td>
      <td>-1.256698</td>
      <td>-0.002999</td>
      <td>-3.510858</td>
      <td>-5.987972</td>
      <td>AAPL</td>
      <td>2017-12-18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.521043</td>
      <td>-3.356587</td>
      <td>-0.451738</td>
      <td>-0.718336</td>
      <td>-0.401614</td>
      <td>-1.450729</td>
      <td>-2.102951</td>
      <td>-2.581210</td>
      <td>1.119429</td>
      <td>1.905858</td>
      <td>-1.651212</td>
      <td>-1.862874</td>
      <td>AAPL</td>
      <td>2017-12-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.783275</td>
      <td>-2.297990</td>
      <td>0.771756</td>
      <td>-1.007854</td>
      <td>-1.208674</td>
      <td>-2.912847</td>
      <td>-4.246914</td>
      <td>-7.193211</td>
      <td>-0.425400</td>
      <td>-0.614857</td>
      <td>-5.018670</td>
      <td>-6.185357</td>
      <td>AAPL</td>
      <td>2017-12-14</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_results.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30653.000000</td>
      <td>30653.000000</td>
      <td>30653.000000</td>
      <td>30653.000000</td>
      <td>30653.000000</td>
      <td>30653.000000</td>
      <td>30653.000000</td>
      <td>30653.000000</td>
      <td>30653.000000</td>
      <td>30653.000000</td>
      <td>30653.000000</td>
      <td>30653.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.064943</td>
      <td>-0.349022</td>
      <td>-0.660375</td>
      <td>-1.253191</td>
      <td>-0.087890</td>
      <td>-0.419967</td>
      <td>-0.768576</td>
      <td>-1.406980</td>
      <td>-0.022946</td>
      <td>-0.070945</td>
      <td>-0.108201</td>
      <td>-0.153789</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.390063</td>
      <td>3.087831</td>
      <td>4.229838</td>
      <td>5.919779</td>
      <td>0.800452</td>
      <td>1.209313</td>
      <td>1.749152</td>
      <td>2.747072</td>
      <td>1.493278</td>
      <td>2.914476</td>
      <td>3.855022</td>
      <td>5.176159</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-10.403576</td>
      <td>-15.705247</td>
      <td>-18.640535</td>
      <td>-27.767035</td>
      <td>-4.372759</td>
      <td>-8.737075</td>
      <td>-13.316881</td>
      <td>-19.459532</td>
      <td>-12.320688</td>
      <td>-17.627251</td>
      <td>-19.984434</td>
      <td>-27.860756</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.749415</td>
      <td>-2.114191</td>
      <td>-3.228079</td>
      <td>-5.100384</td>
      <td>-0.598413</td>
      <td>-1.113717</td>
      <td>-1.733514</td>
      <td>-2.857268</td>
      <td>-0.855000</td>
      <td>-1.763839</td>
      <td>-2.444663</td>
      <td>-3.347510</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.082707</td>
      <td>-0.439099</td>
      <td>-0.825059</td>
      <td>-1.607244</td>
      <td>-0.132948</td>
      <td>-0.537654</td>
      <td>-0.956268</td>
      <td>-1.653505</td>
      <td>-0.065770</td>
      <td>-0.091137</td>
      <td>-0.088434</td>
      <td>-0.074702</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.583501</td>
      <td>1.263256</td>
      <td>1.706667</td>
      <td>2.224064</td>
      <td>0.358805</td>
      <td>0.196716</td>
      <td>0.053994</td>
      <td>-0.273753</td>
      <td>0.773997</td>
      <td>1.662177</td>
      <td>2.296371</td>
      <td>3.236225</td>
    </tr>
    <tr>
      <th>max</th>
      <td>18.741436</td>
      <td>21.471487</td>
      <td>29.706575</td>
      <td>40.148487</td>
      <td>12.121322</td>
      <td>15.348821</td>
      <td>18.503428</td>
      <td>26.788822</td>
      <td>11.028177</td>
      <td>15.413901</td>
      <td>16.637842</td>
      <td>26.235896</td>
    </tr>
  </tbody>
</table>
</div>



### Justification

Here, the final results are discussed in detail. We focus on the difference between predicted and true returns. As previously stated, we look at its mean to evaluate if our model has the tendendcy to produce pessimistic/optimistic prices in the case of a predominant negative or positive error. Its standard deviation helps us in calculating the confidence interval of the produced prediction.

Mean and 95% confidence intervals are plotted below. The error at 1 day shows how the return might be wrong by a nominal -3/+3%. This is quite a lot, so to have an useful guess, we should look for a predicted outcome >3% (albeit further considerations are required). This makes the model being hardly useful as increments >3% in one day refers to quite specific events. A behavioural approach to investment might be considered in such case, however, the model could help in supporting that.

Then, the confidence intervals at 7, 14 and 28 days increase steadily by a further 30% for each days, making the model quite uneffective. We want also to plot a few examples about the returns, to see if it is possible to notice any patterns.


```python
dds = ['1 day','7 days', '14 days', '28 days']
fig, axs = plt.subplots(2,2,figsize=(12,8))
x = np.arange(-20,20,0.1)
for i in range(len(axs)):
    for j in range(len(axs[i])):
        axs[i,j].hist(df_results.iloc[:,8+i*2+j],bins=20, label=dds[i*2+j],density=True)
        mean = df_results.iloc[:,8+i*2+j].mean()
        std  = df_results.iloc[:,8+i*2+j].std()
        y    = stats.norm.pdf(x,mean,std)
        axs[i,j].plot(x,y,color='red')
        axs[i,j].text(mean+0.1,y.max(),round(mean,3),)
        z1 = stats.norm(mean,std).ppf(0.025)
        z2 = stats.norm(mean,std).ppf(1-0.025)
        axs[i,j].axvline(z1, ls=':', color='orange', label=z1)
        axs[i,j].text(z1+0.1,.1,round(z1,3),rotation=90)
        axs[i,j].axvline(z2, ls=':', color='orange', label=z2)
        axs[i,j].text(z2+0.1,.1,round(z2,3),rotation=90)
        axs[i,j].set_xlim((-12,12))
        axs[i,j].set_ylim((0,0.35))
        axs[i,j].set_title(dds[i+j])
        
plt.tight_layout()

```


![png](output_28_0.png)


Here, returns are plotted for 3 companies. The model produces much stable outputs compared to the real returns.


```python
fig,axs = plt.subplots(3,1,figsize=(17,12))

def show_returns(to_show, shift=0):
    """Show returns for a set of companies at 1,7,14 or 28 days.
    
    Parameters
    ----------
    to_show : array-like
        contains the tickers
        
    shift : int
        0 = 1 day
        1 = 7 days
        2 = 14 days
        3 = 28 days
    """
    gs = df_results.groupby('Ticker')

    for n,g in gs:
        if n in to_show:
            x = g['Date'].values
            y1 = g.iloc[:,shift+0].values
            y2 = g.iloc[:,shift+4].values
            y3 = g.iloc[:,shift+8].values
            axs[0].plot(x,y1,label=n)
            axs[1].plot(x,y2,label=n)
            axs[2].plot(x,y3,label=n)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[0].set_ylim((-10,15))
    axs[1].set_ylim((-10,15))    
    axs[2].set_ylim((-10,15))
    axs[0].set_title('Returns - Truth')
    axs[1].set_title('Returns - Predictions')
    axs[2].set_title('Returns - Errors')
    
# df_results['Ticker'].unique()
show_returns(['AAPL','AXP','CAT'],0)
```

![png](output_30_0.png)

<br/>


## Conclusion

### Reflection

The end-to-end product of this project tries to bring together historical data about some of the S&P500 companies, where available through Quandl API. The data did not present inconsistencies, nor missing values, requiring only standard scaling practice before proceeding through picking a model to train and test. 

The selection fell on scikit-learn's random forest regressor. An initial overfitting happened due to lack of tuning hyper parameters, corrected by performing a grid search, and training the model through cross-validation (this was possible by creating a pipeline in order to avoid data leaks on cross validation).

Results were tested against analysing rate on returns, or RoR, and so by comparing the predicted with true returns. By doing so, it was possible to define a confidence interval relative to each time range. 7-days returns shows a 95% confidence interval of about -/+5% which was within the project's guidelines, however, by far a result to be considered good in  real-world business scenario.

Developing knowledge of time series as well as of investment principles and returns calculation was one of the most interesting aspect of the project. 

<br/>

### Improvement

The nature of the problem, as well as the data here digested presents a high degree of non-linearity. Improvements to the system can be made by choosing a different model (e.g. SVM regressor to be trained on each time range) which is more prone to capture highly non-linear patterns. Eventually, going for a deep neural network might help: nlp-related machine learning techniques such as RNN or LSTM might be used to also include the temporality of the data (here, the window is digested as a flat vector by the model).

Then, further feature engineering might be required. For instance, returns might be included. Moreover, company-specific indices might help in determining the volatility of the company, or their financial status such as quarterly reports data.

Finally, test of the window size might be run in order to see how this affect our model. However, to avoid increasing the dimensionality of data too much without providing a time-related approach to train our model (such in the case of a LSTM), might produce worse rather than better results.

<br/>