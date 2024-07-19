import pandas as pd 
import numpy as np 

from sklearn.model_selection import RandomizedSearchCV,TimeSeriesSplit,cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression



# Load the self.df

df = pd.read_csv('TSLA.csv') 

# The date column is an object, we need to convert it to a datetime object

df['Date'] = pd.to_datetime(df['Date'])

class ClassifyTeslaStocks:

    def __init__(self,df):
        self.df = df

    def target_creation(self):
        """
        First, we calculate the daily close change by taking the difference between the close price of the current day 
        and the close price of the previous day. This difference is then divided by the close price of the previous
        day to convert it into a percentage change. Next, we create the target variable by checking if the daily close 
        change is greater than or equal to 5%. Initially, we tried using a threshold value of 5%, but it created an 
        unbalanced self.dfset, with the number of 1s (indicating a significant price increase) being very low compared to 
        the number of 0s. Since we cannot shuffle the self.dfset due to its time series nature, we experimented with a 1% 
        threshold, which resulted in a more balanced number of 1s and 0s. However, a 1% change was considered too 
        insignificant to provide meaningful business value, so we decided to use the 5% threshold despite the resulting 
        imbalance. After creating the target variable, we drop the 'Daily_Close_Change' column since the target variable 
        has been derived from it.
        """

        self.df['Daily_Close_Change'] = (self.df['Close'] - self.df['Close'].shift(1))/self.df['Close'].shift(1)

        self.df['Target'] = np.where(self.df['Daily_Close_Change'] >= 0.05, 1, 0)

        self.df.drop(columns = 'Daily_Close_Change', inplace=True)

        return self.df

    def exponential_moving_average(self,columns_to_process=['Open', 'High', 'Low', 'Adj Close'],window_sizes=[7, 15, 21, 30]):

        """
        This function calculates the Exponential Moving Average (EMA) for the columns 
        specified in the columns_to_process list. Exponential Moving Average is a type of moving average that
        places a greater weight and significance on the most recent self.df points. 
        The percentage change between the original self.df and EMA is calculated and stored in new columns.
        This result can be used to identify the trend of the self.df. If the percentage change is positive, it indicates that the
        self.df is above the moving average and the trend is upward. If the percentage change is negative, it indicates that the self.df
        is below the moving average and the trend is downward.
        The epsilon parameter is added to avoid division by zero error.
        """
        epsilon = 1e-8

        for col in columns_to_process:
            for window in window_sizes:

                ema_col_name = f'{col}_EMA_{window}_Percentage'
                ema = self.df[col].ewm(span=window, adjust=False).mean() + epsilon
                self.df[ema_col_name] = (self.df[col] - ema) / self.df[col]
        
        return self.df


    def stochastic_oscillator_feature(self,column_of_low_value='Low',column_of_high_value='High'):

        """ 
        The Stochastic Oscillator is a momentum indicator that shows the location of the close relative to 
        the high-low range over a set number of periods. This function calculates the Stochastic Oscillator for
        two different windows, 5 and 15.
        """

        lowest_5 = self.df[column_of_low_value].rolling(window = 5).min()
        highest_5 = self.df[column_of_high_value].rolling(window = 5).max()

        self.df['Stochastic_5'] = ((self.df['Close'] - lowest_5)/(highest_5 - lowest_5))*100

        lowest_15 = self.df[column_of_low_value].rolling(window = 15).min()
        highest_15 = self.df[column_of_high_value].rolling(window = 15).max()

        self.df['Stochastic_15'] = ((self.df['Close'] - lowest_15)/(highest_15 - lowest_15))*100

        return(self.df)


    def relative_strength_index(self):
        
        """
        The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
        The RSI oscillates between 0 and 100. Traditionally, and according to Wilder, RSI is considered overbought when 
        above 70 and oversold when below 30. This function calculates the RSI for two different windows, 5 and 15. Then,
        it calculates the ratio between the two RSI values. The RSI ratio can be used to identify the trend of the self.df.
        After the RSI and RSI ratio are calculated, the function drops the intermediate columns used for the calculation.
        """
        
        self.df['Diff'] = self.df['Close'].diff()
        self.df['Up'] = self.df['Diff']
        self.df.loc[(self.df['Up']<0), 'Up'] = 0
        
        self.df['Down'] = self.df['Diff']
        self.df.loc[(self.df['Down']>0), 'Down'] = 0 
        self.df['Down'] = abs(self.df['Down'])

        self.df['Up_5MA'] = self.df['Up'].rolling(window=5).mean()
        self.df['Down_5MA'] = self.df['Down'].rolling(window=5).mean()

        self.df['Up_15MA'] = self.df['Up'].rolling(window=15).mean()
        self.df['Down_15MA'] = self.df['Down'].rolling(window=15).mean()

        self.df['RS_5'] = self.df['Up_5MA'] / self.df['Down_5MA']
        self.df['RS_15'] = self.df['Up_15MA'] / self.df['Down_15MA']

        self.df['RSI_5'] = 100 - (100/(1+self.df['RS_5']))
        self.df['RSI_15'] = 100 - (100/(1+self.df['RS_15']))

        self.df['RSI_ratio'] = self.df['RSI_5']/self.df['RSI_15']

        # Drop the columns that are not used to create a feature and may result multicollinearity in the future 

        self.df.drop(columns = ['Diff','Up','Down','Up_5MA','Down_5MA','Up_15MA','Down_15MA','RS_5','RS_15','RSI_5','RSI_15'], inplace=True)

        return self.df

    def df_before_modelling(self):
        """
        This function fills the infinitive values with NaN and then fills the NaN values with with the available next value.
        If there is no next value, it fills the NaN value with the previous value. After filling the NaN values, the function
        drops the columns that are used to create features and are not needed for the model. Lastly, it selects the columns that 
        decided with correlation analysis and feature importance based on Logistic Regression Model. 
        """
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df_filled = self.df.bfill().ffill()

        self.df_filled.drop(columns = ['Open','High','Low','Close','Adj Close','Volume','Stochastic_5'],inplace=True)

        self.df_filled = self.df_filled[["Date","Target","Open_EMA_7_Percentage","Open_EMA_15_Percentage","Open_EMA_30_Percentage",
                            "High_EMA_7_Percentage","High_EMA_15_Percentage","Low_EMA_7_Percentage",	"Low_EMA_15_Percentage",
                            "Adj Close_EMA_7_Percentage","Adj Close_EMA_15_Percentage","Stochastic_15","RSI_ratio"]]

        return(self.df_filled)

    def model(self):

        self.df_sorted = self.df_filled.sort_values('Date')
        X = self.df_sorted.drop(['Target', 'Date'], axis=1)
        y = self.df_sorted['Target']

        split_index = int(len(self.df_sorted) * 0.7)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        tscv = TimeSeriesSplit(n_splits=5,gap = 20)

        selected_model = LogisticRegression(class_weight='balanced')
        params = {'penalty': ['l1', 'l2'],
            'C': np.logspace(-4, 4, 20),
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'] 
        }
        clf = RandomizedSearchCV(
        estimator=selected_model,
        param_distributions=params,
        n_iter=100,
        cv=tscv,
        n_jobs=-1,
        scoring='roc_auc',
        refit=True,
        return_train_score=True
    )

        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        best_model.fit(X_train, y_train)
        cv_results = cross_val_score(best_model, X_train, y_train, cv=tscv, scoring='roc_auc')

        y_pred_proba = best_model.predict_proba(X_test)
        y_pred_true_label = y_pred_proba[:, 1]

        train_score = np.mean(cv_results)
        test_score = roc_auc_score(y_test, y_pred_true_label)

        # For logistic regression, coef_ is a 2D array, so we need to access the first element
        coefficients = best_model.coef_[0]  
        feature_names = X_test.columns
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': np.abs(coefficients)})
        coef_df_sorted = coef_df.sort_values(by='Coefficient', ascending=False)

        print(f'Mean Train Score of Cross Validation with Random Search on Logistic Regression model: {train_score}')
        print(f'Test Score of Logistic Regression model: {test_score}')
        print(f'Best Parameters of Logistic Regression model: {clf.best_params_}')
        print(f'Feature Importance of Logistic Regression model: {coef_df_sorted}')

        return self.df_filled
    
    def execute_all_methods(self):
        method_sequence = [
            'target_creation',
            'exponential_moving_average',
            'stochastic_oscillator_feature',
            'relative_strength_index',
            'df_before_modelling',
            'model'
        ]

        for method_name in method_sequence:
            method = getattr(self, method_name)
            method()

if __name__ == "__main__":
    tesla_model = ClassifyTeslaStocks(df)
    tesla_model.execute_all_methods()
            