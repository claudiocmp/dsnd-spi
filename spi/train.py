# basics
import os
# data
import json
import numpy as np
import scipy.stats as stats
import pandas as pd
import pickle
# ML
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,log_loss,regression,r2_score
# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# settings
sns.set(style="white")



class Trainer(object):

    def __init__(self, data_fn="dataset.csv"):
        """ 
        """
        # data folder
        self._folder = os.path.join(os.getcwd(),"..\\data")
        self.dataset_fn, self.tickers_fn = self.read_data(data_fn)
        

    def read_data(self, fn):
        """
        """
        df = pd.read_csv(os.path.join(self._folder,fn), parse_dates=['Date'], index_col='Unnamed: 0',)
        return df, df['Ticker'].unique().values


    # verifying data integrity and basic stats
    print(df.isna().sum())
    df.describe()


    # check whether the period is shared across companies, bringin together first date, last date and the difference between the two
    m = pd.merge(
        pd.merge(
            df.groupby('Ticker')['Date'].min(),
            df.groupby('Ticker')['Date'].max(),
            left_index=True,
            right_index=True),
        df.groupby('Ticker')['Date'].max()-df.groupby('Ticker')['Date'].min(),
        left_index=True,
        right_index=True
    )
    m.columns = ['start date','end date', 'difference']
    m



    def window_data(self, df, window, to_drop=['Date','Ticker'], y_label='Adj_Close', grouper='Ticker'):
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
        # WINDOWING
        groups = df.groupby(grouper)
        X = []
        y = []
        tickers = []
        dates = []
        for n,g in groups:
            # NOTE: we could add a company-related information (e.g. volatility index)
            for i in range(window, len(g)-28):
                g = g.reset_index(drop=True)
                # drop and transform X
                data = g[i-window:i].drop(columns=to_drop,inplace=False)
                # transform y - remember to increase dimensionality of the array
                pred = np.expand_dims(np.array([g.loc[i+1,y_label],g.loc[i+7,y_label],g.loc[i+14,y_label],g.loc[i+28,y_label]]),0)
                # done-
                X.append(data.values)
                y.append(pred)
                tickers.append(n)
                dates.append(g.loc[i]['Date'])
        return np.array(X),np.array(y),np.array(tickers),np.array(dates)

    X2,y2,tickers,dates = window_data(df.drop(columns=['Open','Close','Low','High']),5)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, train_size=0.7, random_state=42, shuffle=False)


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
    pipeline