import os
import re
import datetime
import random
from datetime import datetime

# Data Processing
import pandas as pd
import numpy as np

#statsmodels
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Statistics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, mean_squared_error, root_mean_squared_error, mean_absolute_error

#scipy
import scipy
from scipy import stats
#plots
import matplotlib.pyplot as plt

#model and information formats
import json
import joblib

class Prediction_Base():
    def __init__(self,datadir = '../data/',
                 dataname = 'msft_hist.csv', 
                 automate_lag = True,
                 no_of_lag = 2,
                 train_years = [2022],
                 test_years = [2023], 
                 model='RandomForest'):

        self.time = str(datetime.now().timestamp())

        
        self.datadir = datadir
        self.dataname = dataname

        #this is the path for the plots for each experiment. Seeds added separately 
        # time is removed for visibility. Each model config should run only once anyway, so no problem there.
        if not os.path.isdir(f'../results'):
            os.mkdir(f'../results')
        if not os.path.isdir(f'../results/{model}/'):
                os.mkdir(f'../results/{model}/')
                os.mkdir(f'../results/{model}/{dataname}')
                os.mkdir(f'../results/{model}/{dataname}/{no_of_lag}')
                os.mkdir(f'../results/{model}/{dataname}/{no_of_lag}/{train_years}-{test_years}')
        if not os.path.isdir(f'../results/{model}/{dataname}'):
                os.mkdir(f'../results/{model}/{dataname}')
                os.mkdir(f'../results/{model}/{dataname}/{no_of_lag}')
                os.mkdir(f'../results/{model}/{dataname}/{no_of_lag}/{train_years}-{test_years}')
        if not os.path.isdir(f'../results/{model}/{dataname}/{no_of_lag}'):
                os.mkdir(f'../results/{model}/{dataname}/{no_of_lag}')
                os.mkdir(f'../results/{model}/{dataname}/{no_of_lag}/{train_years}-{test_years}')
        if not os.path.isdir(f'../results/{model}/{dataname}/{no_of_lag}/{train_years}-{test_years}'):
                os.mkdir(f'../results/{model}/{dataname}/{no_of_lag}/{train_years}-{test_years}')
        if not os.path.isdir(f'../results/{model}/{dataname}/{no_of_lag}/{train_years}-{test_years}/hitratio'):
                os.mkdir(f'../results/{model}/{dataname}/{no_of_lag}/{train_years}-{test_years}/hitratio')
        #where all the plots and data are saved
        self.experimentpath = f'../results/{model}/{dataname}/{no_of_lag}/{train_years}-{test_years}/'
        #used in lag_creation
        self.no_of_lag = no_of_lag # ranges from 2 to 28 (btc/regular)
        #the amount of time the model can look into the past - eg. 7 means that 7 days are available on step 8 for it to look at.
        self.lags = None
        # is the modelname - used to create a directory
        self.model = model

        #used in scaling
        self.scaled_data = None
        #boolean, decide whether or not the lag should be automated. is set to true when the lag value inserted is 'auto'
        self.automate_lag = automate_lag
        
        #datasets prior to split
        self.train = None
        self.test = None

        
        #initialization of years between test and train
        self.train_years = train_years
        self.test_years = test_years

        #used for scaling
        self.X_train_scaled = None
        self.y_train_scaled = None
        self.X_test_scaled = None
        self.y_test_scaled = None

        #The acf values - upon which the auto is decided.
        self.acf_values=None

        # The seeds are created with below functioncall. The function returns a list of 100 values ranging from 1 to 100.
        self.random_seeds = self.create_random_seeds()

        # a catchall dictionary
        self.regressors = {}
        # collects the MSE values for the model
        self.mse_values = {}
        # collects the MSE values for the model
        self.stats_mse = {}

    def create_random_seeds(self):
        ##
        # returns a list of 100 values
        ##
        return list(range(1,101))

    def create_random_walk(self):
        """
        This function creates a randomwalk. 
        The randomwalk is then cumulated. 
        The randomwalk is produced to check whether the model is currently looking into the future and thus able to memorize the randomwalk.
        For reproducibility, the seed is set to 42. 
        The randomwalk has a mean of 100, and a standard deviation of 1. 
        The randomwalk has a length of 5000, where each point is a Close value of said day.
        The dataset starts at 2014-01-01.
        """
        #Define a daterange and turn it into a dataframe. 
        # If changes are made here below the 2024 date, the model will not be able to run due to lack of data for these years.
        date_range = pd.date_range(start=pd.to_datetime('2014-01-01').tz_localize('Europe/Zurich'), periods=5000, freq='D')
        date_range = pd.to_datetime(date_range)
        # Set a seed, so that our random walk can be reproduced by anyone.
        np.random.seed(42)
        # Create random noise - with the length of the daterange.
        random_noise = np.random.normal(loc=0, scale=1, size=len(date_range))
        # The cumulative sum of the randomwalk is made here. Further, the mean is shifted to 100. 
        # This is necessary to allow the log/diff transformation to happen - and prevents values under 0 from making problems
        cumulative_sum = np.cumsum(random_noise)
        random_walk = cumulative_sum + 100
        # In this for loop, the dataframe is created. 
        # Due to the Nature of the data consisting usually of 8 columns, the randomwalk created above is replicated 8 times and assigned to 
        # the corresponding columns.
        rw_ls = []
        for i in range(8):
            rw_ls.append(random_walk)
        r_w_df = np.array(rw_ls).T
        # Create a DataFrame with the random walk
        raw_data = pd.DataFrame(r_w_df, index=date_range, columns=['Open','High','Low','Close','Volume','Dividends','Stock Splits','Capital Gains'])
        raw_data.index.name = 'Date'
        # The randomwalk is saved in the Folder ../data/.
        raw_data.to_csv('../data/random_walk.csv')

    def set_seed(self, seed):
        """
        seed: current seed of the model
        sets self.seed to the current seed, aswell as setting the random seed for that framework.
        """
        random.seed(seed)
        self.seed = seed

    def read_in_data(self):
        """
        This function reads in data.
        The path is given through self.datadir
        Makes sure the index is assigned correctly, and that these are sorted.
        Calls the acf_plot2 function, if the lag is to be automated due to a dependency in a function later in the call path.
        """
        for i in os.listdir(self.datadir):
            if i == self.dataname:
                self.raw_data = pd.read_csv(self.datadir+i, index_col = False)
                self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'], utc=True)
                self.raw_data.set_index('Date', inplace=True)
                self.raw_data.sort_index(inplace=True)                
        
        if self.automate_lag:
            self.acf_plot()

    def plot_data_for_comparison(self):
        """
        This provides an overview over the provided dataset over all years.
        Following transformations are visualized:
        - None - just the regular close
        - log - removes the exponentiality of the data
        - diff(log) - removes exponentiality and compares two values, takes the difference
        - Volume - just plot the traded volume
        The function provides a good overview of the dataset/timeseries used.
        It is not of relevance to the overall prediction, but important due to it providing necessary context.
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Prices
        axs[0, 0].plot(self.raw_data.index, self.raw_data['Close'], color='blue')
        axs[0, 0].grid(True)
        axs[0, 0].set_title('Prices')

        # Log-prices
        axs[0, 1].plot(self.raw_data.index, np.log(self.raw_data['Close']), color='blue')
        axs[0, 1].grid(True)
        axs[0, 1].set_title('Log-prices')

        # Log-returns
        axs[1, 0].plot(self.raw_data.index[:-1], np.diff(np.log(self.raw_data['Close'])), color='blue')
        axs[1, 0].grid(True)
        axs[1, 0].set_title('Log-returns')

        # Log-volumes
        try:
            axs[1, 1].plot(self.raw_data.index, np.log(self.raw_data['Volume']), color='blue')
            axs[1, 1].grid(True)
            axs[1, 1].set_title('Log-volumes')
        except:
            pass
        # Adjust layout to not overlap and show the plots
        plt.tight_layout()
        plt.savefig(f'../results/{self.model}/{self.dataname}/original_comparison_plot')
        plt.close()

    def acf_plot(self):
        """
        This makes the acf and pacf plots. 
        It also ensures that the lags are correctly made - in case the lag is 'auto', the choice of lag is not upon the 
        user, but instead the first significant value is used. 
        The significant value is decided upon the 95% boundary using standard deviation and the simplified +-1.96 * se
        Assigns self.no_of_lag
        """
        if self.no_of_lag == 'auto':
            self.automate_lag=True
            self.no_of_lag = 2
        
        if not os.path.isdir(self.experimentpath + 'ACFPlot/'):
            os.mkdir(self.experimentpath + 'ACFPlot/')
        
        if self.acf_values is None:
            self.acf_values = np.diff(np.log(self.raw_data["Close"]))
        acf_values = acf(self.acf_values)  # changed to default, so the confint on the plot seems to be correct
        
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        plot_acf(self.acf_values, ax=ax[0])
        plot_pacf(self.acf_values, ax=ax[1])
        fig.savefig(self.experimentpath + f'ACFPlot/ACF')
        plt.close()
        
        if self.automate_lag:
            n = len(self.acf_values) 
            se = 1 / np.sqrt(n)  # Standard error -> possible to do Bartlett confint
            upper_bound = 1.96 * se
            lower_bound = -1.96 * se
            self.automate_lag=False
            for lag, acf_v in enumerate(acf_values):
                if lag == 0:  # Skip the first lag, because it's always 1
                    continue
                if acf_v > upper_bound or acf_v < lower_bound:
                    self.no_of_lag = int(lag)
                    if self.no_of_lag <=1:
                        self.no_of_lag = 2
                    break 

    def create_lags(self):
        """
        This function creates the lags, as decided in the function acf_plot2. 
        It creates a new dataset, hereby called self.lags.
        This dataset consists of the last X days, where X = number of lags.
        The values are shifted, after a diff and log transformation.
        It also isolates the first values of raw data. Its value is the same as the first value of train
        The lag columns are named after their day, increasing with distance.
        """
        print(f'-----------------#of lag created: {self.no_of_lag}------------------')
        self.first_value_raw_data = self.raw_data["Close"].iloc[0]
        self.raw_after_diff_and_log = pd.DataFrame(np.diff(np.log(self.raw_data["Close"])),
                                           index = self.raw_data.index[1:], columns=['Close'])
        
        self.lags = [self.raw_after_diff_and_log.shift(i) for i in range(self.no_of_lag+1)]
        #make self.lags into a dataframe again
        self.lags = pd.concat(self.lags, axis=1)
        self.lags.columns = ['Lag_'+str(i) for i in range(self.no_of_lag+1)]
        self.lags = self.lags.dropna()

    def split_dataset_by_years(self):
        """
        In this function, the dataset is split into two parts, defined by train_years and test_years.
        The test year is always one year, where we attempt to predict said year with the information provided by the 
        previous years.
        It performs a check, if the provided years are lists, and makes them into lists if not.

        The function further extracts the first values of train firt - before any transformation is done. This is required for a 
        backtransformation in the final steps to check prediction performance.

        Assings: first_raw_value_train, first_raw_value_test, last_raw_value_train, last_raw_value_test
        """
        if not isinstance(self.train_years, list):
            self.train_years = [self.train_years]
        if not isinstance(self.test_years, list):
            self.train_years = [self.test_years]
        self.train = self.lags[self.lags.index.year.isin(self.train_years)]
        self.test = self.lags[self.lags.index.year.isin(self.test_years)]
        self.first_raw_value_train = self.raw_data[self.raw_data.index.year.isin(
            self.train_years)]["Close"].iloc[0]
        self.first_raw_value_test = self.raw_data[self.raw_data.index.year.isin(
            self.test_years)]["Close"].iloc[0]
        self.last_raw_value_train = self.raw_data[self.raw_data.index.year.isin(
            self.train_years)]["Close"].iloc[-1]
        self.last_raw_value_test = self.raw_data[self.raw_data.index.year.isin(
            self.test_years)]["Close"].iloc[-1]
    
    def split_dataset_by_train_test(self):
        """
        The dataset gets split apart by the commonly used X_train, y_train, X_test, y_test.
        The train values contain all the lags, while the test values contain all the true predictions.

        Assigns X_train, y_train, X_test, y_test
        
        """
        
        self.y_train = self.train.iloc[:,0]
        self.X_train = self.train.iloc[:,1:]
        self.y_test = self.test.iloc[:,0]
        self.X_test = self.test.iloc[:,1:]
        #print(self.y_train.shape, self.X_train.shape, self.y_test.shape, self.X_test.shape)
        #print("train and testplit have been made")

    def return_to_original_values(self):
        """
        This returns the values we transformed with log/diff - into the original values. 
        This is important for the confirmation of the prediction, eg. what are the actual predictions and are they valid.
        Creates the TruePrediction Folder in experimentpath.

        Assigns: descaled_y_train_pred, descaled_y_test_pred

        Is run for each model.

        Saves the Figure under the relative path: '../results/{model}/{dataname}/{no_of_lag}/{train_years}-{test_years}/TruePrediction/{self.seed}'
        Example: "../results/RandomForest/btc_hist.csv/auto/[2022,2023]-[2024]/TruePrediction/1.png"
        """
        
        if not os.path.isdir(self.experimentpath +'TruePrediction/'):
            os.mkdir(self.experimentpath +'TruePrediction/')
        self.descaled_y_train_pred = pd.DataFrame(np.exp(np.concatenate([[np.log(self.first_raw_value_train)], np.cumsum(self.y_train_pred) + np.log(self.first_raw_value_train)]))[1:],
                                                 index = self.y_train.index)
        self.descaled_y_test_pred = pd.DataFrame(np.exp(np.concatenate([[np.log(self.first_raw_value_test)], np.cumsum(self.y_test_pred) + np.log(self.first_raw_value_test)]))[1:],
                                                index = self.y_test.index)
        
        plt.plot(self.raw_data[self.raw_data.index.year.isin(self.train_years)]["Close"], label='True Trainvalue')
        plt.plot(self.raw_data[self.raw_data.index.year.isin(self.test_years)]["Close"], label='True Testvalue')
        plt.plot(self.descaled_y_train_pred, label = 'Train Prediction')
        plt.plot(self.descaled_y_test_pred, label = 'Test Prediction')
        #plt.plot(self.y_train)
        plt.legend(loc='upper left')
        plt.savefig(self.experimentpath + f'TruePrediction/{self.seed}')
        plt.close()
        #print("Plots have been done.")

    def append_regressors(self):
        """
        This function is called to collect all the available information in a dictionary for each model. 
        Serves as a rallying point for each model. 

        Assigns: regressors
        """
        self.regressors[self.seed] = {'y_train_pred': self.descaled_y_train_pred,
                                      'y_test_pred': self.descaled_y_test_pred,
                                      'y_test_pred_scaled': self.y_test_pred,
                                      'y_train_pred_scaled' : self.y_train_pred}
        
    def plot_performance(self):
        """
        Calculate, plot and save the performance. 
        The model predicts log/diff values.
        
        Plots the cumsum of the true values and the cumsum of predictions.
        
        We calculate the following values
            - perf_svr
            - sharpe_svr
            - cumulative_perf_svr
            - first_date
            - last_date
            - nomenclature
        Assigned: cumulative_perf_svr

        Is done for each model.
        
        Saves the Figure under the relative path: '../results/{model}/{dataname}/{no_of_lag}/{train_years}-{test_years}/PerformancePlots/{self.seed}'
        Example: "../results/RandomForest/btc_hist.csv/auto/[2022,2023]-[2024]/PerformancePlots/1.png"
        
        """
        if not os.path.isdir(self.experimentpath +'PerformancePlots/'):
            os.mkdir(self.experimentpath +'PerformancePlots/')
        
        if len(self.y_test_pred) != len(self.y_test):
            raise ValueError("y_test_pred and target_out length mismatch. Check for leap year issues.")
        #Sharpe Ratio for display on the Plot
        perf_svr = np.sign(self.y_test_pred)*self.y_test
        sharpe_svr = np.sqrt(len(self.y_test)) * perf_svr.mean()/ perf_svr.std()
        cumulative_perf_svr = np.cumsum(perf_svr)
        #required for writing into mseofmodels.json
        self.cumulative_perf_svr = cumulative_perf_svr
        first_date = self.y_train.index[0]
        last_date =  self.y_test.index[-1]
        nomenclature = f"{first_date.strftime('%Y-%m-%d')}_to_{last_date.strftime('%Y-%m-%d')}, {self.model}"
        
        # Plotting
        plt.figure(figsize=(10, 6))  # Set the figure size
        plt.plot(cumulative_perf_svr, label='Cumulative Performance')
        plt.plot(np.cumsum(self.y_test), label='Buy&Hold', color = 'black', linewidth = 2)
        plt.title(f"Random Forest cumulated performances out-of-sample, Sharpe={round(sharpe_svr, 2)}")
        plt.xlabel('Time')
        plt.ylabel('Cumulative Performance')
        plt.legend()
        plt.savefig(self.experimentpath + f'PerformancePlots/{self.seed}')
        plt.close()
    def plot_train_comparison(self):
        """
        Compares the true train value and the predicted train value on the original scale. 

        Assigns: Nothing
        
        Is done for each model
        
        Saves the Figure under the relative path: '../results/{model}/{dataname}/{no_of_lag}/{train_years}-{test_years}/TrainComparison/{self.seed}'
        Example: "../results/RandomForest/btc_hist.csv/auto/[2022,2023]-[2024]/TrainComparison/1.png"
        """
        
        if not os.path.isdir(self.experimentpath +'TrainComparison/'):
            os.mkdir(self.experimentpath +'TrainComparison/')
        
        
        plt.plot(self.raw_data[self.raw_data.index.year.isin(self.train_years)]["Close"], label='True data', color = 'black', linewidth = 2)
        for i in self.regressors:
            plt.plot(self.regressors[i]['y_train_pred'], label = 'Predictions')
        plt.savefig(self.experimentpath + f'TrainComparison/All_Train.png')
        plt.close()

    def plot_test_comparison(self):
        """
        Compares the true test value and the predicted test value on the original scale. 

        Assigns: Nothing
        
        Is done for each model
        
        Saves the Figure under the relative path: '../results/{model}/{dataname}/{no_of_lag}/{train_years}-{test_years}/TestComparison/All_Test'
        Example: "../results/RandomForest/btc_hist.csv/auto/[2022,2023]-[2024]/TestComparison/All_Test.png"
        """
        
        if not os.path.isdir(self.experimentpath +'TestComparison/'):
            os.mkdir(self.experimentpath +'TestComparison/')
        plt.plot(self.raw_data[self.raw_data.index.year.isin(self.test_years)]["Close"], label='True Data', color = 'black', linewidth = 2)
        for i in self.regressors:
            plt.plot(self.regressors[i]['y_test_pred'], label = 'Predictions')
        
        plt.savefig(self.experimentpath + f'TestComparison/All_Test')
        plt.close()

    def model_statistics_unscaled(self):
        """
        Prints the unscaled statistical valus. 
        Values chosen: MSE between raw_data[train_years] and the prediction of the model on train
                       MSE between raw_data[test_years] and the prediction of the model on test
        """
        if self.mse_values !={}:
            self.mse_values = {}
        for i in self.regressors:
            for j in self.regressors[i]:
                self.mse_values[i] = {'y_train_mse': mean_squared_error(self.raw_data[self.raw_data.index.year.isin(self.train_years)]["Close"][-len(self.regressors[i]['y_train_pred']):], self.regressors[i]['y_train_pred'][-len(self.regressors[i]['y_train_pred']):])}
                self.mse_values[i]['y_test_mse'] = mean_squared_error(self.raw_data[self.raw_data.index.year.isin(self.test_years)]["Close"], self.regressors[i]['y_test_pred'])
                self.mse_values[i]['y_train_pred_scaled'] = mean_squared_error(self.y_train, self.regressors[i]['y_train_pred_scaled'])
                self.mse_values[i]['y_test_pred_scaled'] = mean_squared_error(self.y_test, self.regressors[i]['y_test_pred_scaled'])
        self.test_mse = []
        self.train_mse = []
        self.y_test_pred_mse = []
        self.y_train_pred_mse = []
        for i in self.mse_values:
            self.train_mse.append(self.mse_values[i]['y_train_mse'])
            self.test_mse.append(self.mse_values[i]['y_test_mse'])
            self.y_train_pred_mse.append(self.mse_values[i]['y_train_pred_scaled'])
            self.y_test_pred_mse.append(self.mse_values[i]['y_test_pred_scaled'])

    def hit_ratio(self):
        """
        We calculate and save the hitrate for train and test, and save them correspondingly. 

        Assigns: Nothing
        
        Is done for each model configuration
        """
        perf_svr = np.sign(self.y_test_pred)*self.y_test
        sharpe_svr = np.sqrt(len(self.y_test)) * perf_svr.mean()/ perf_svr.std()
        count = 0
        for index,value in enumerate(np.sign(self.y_train)):
            if value == np.sign(self.y_train_pred)[index]:
                count+=1
        count_test = 0
        for index,value in enumerate(np.sign(self.y_test)):
            if value == np.sign(self.y_test_pred)[index]:
                count_test+=1
        hit_ratios = {'Hit Ratio Train': count/len(self.y_train),'Hit Ratio Test': count_test/len(self.y_test), 'Sharpe':sharpe_svr}
        try:
            with open(self.experimentpath+f'hitratio/{self.seed}.txt', 'w') as file:
                file.write(json.dump(str(hit_ratios), file))
        except:
            pass
    
    def write_stats_to_file(self):
        if not os.path.isdir(self.experimentpath+f'statistics/'):
            os.mkdir(self.experimentpath+f'statistics/')
        
        all_statistics_model = {'unscaled statistics': 
                                         {'Pearson': scipy.stats.pearsonr(self.train_mse,self.test_mse),
                                         'Covariance Matrix': np.cov(self.train_mse,self.test_mse, bias=True),
                                         'T-Test': stats.ttest_ind(self.train_mse,self.test_mse),
                                         'Mean Squared Error': mean_squared_error(self.y_test_pred, self.y_test)},
                                    'scaled statistics': 
                                         {'Pearson': scipy.stats.pearsonr(self.y_train_pred[-len(self.y_test_pred):],self.y_test_pred[-len(self.y_train_pred):]),
                                         'Spearman': scipy.stats.spearmanr(self.y_train_pred[-len(self.y_test_pred):],self.y_test_pred[-len(self.y_train_pred):]),
                                         'Covariance Matrix': np.cov(self.y_train_pred[-len(self.y_test_pred):],self.y_test_pred[-len(self.y_train_pred):], bias=True),
                                         'T-Test': stats.ttest_ind(self.y_train_pred[-len(self.y_test_pred):],self.y_test_pred[-len(self.y_train_pred):])},
                                     'MSE Comparison': {
                                         'Pearson': scipy.stats.pearsonr(self.y_test_pred[-len(self.y_train_pred):], self.y_train_pred[-len(self.y_test_pred):]),
                                         'Spearman': scipy.stats.spearmanr(self.y_test_pred[-len(self.y_train_pred):], self.y_train_pred[-len(self.y_test_pred):]),
                                         'Covariance Matrix': np.cov(self.y_test_pred[-len(self.y_train_pred):], self.y_train_pred[-len(self.y_test_pred):], bias=True),
                                         'T-Test': stats.ttest_ind(self.y_test_pred[-len(self.y_train_pred):], self.y_train_pred[-len(self.y_test_pred):])
                                        }
                                    }
        try:
            with open(self.experimentpath+f'statistics/stats.json', 'w') as file:
                file.write(json.dump(str(all_statistics_model), file))
        except:
            pass


    def save_model(self):
        print('save model not implemented')

    def load_model(self):
        print('load model not implemented')

    def register_mse_of_model(self):
        print('register_mse_of_model not implemented')
        
    def run_model_configuration(self):
        print('model configuration not implemented')

    def run_model(self):
        print('run_model not implemented')