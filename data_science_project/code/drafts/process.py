# Take output from dendra_api and:
# perform simple cleaning
# plot by tree and by sensor
# retain the ability to hook up to other sensor information
from socketserver import ThreadingUnixDatagramServer
import numpy as np
import helpers
import string
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt

### Cleaning methods
#real_df = pd.read_csv('SapTemperatureDifference_test_export.csv')
def main():
    df = pd.read_csv('emdata/SapTemperatureDifference.csv')
    print(df)
    print(df.shape)

    df = clean_sapflux(df)
    df_sapflux = calc_sapflux(df)
    
    plot_trees(df_sapflux)
@helpers.timer

def clean_sapflux(df, lowpass_threshold = 1, warmup_timestep = 12, time_col = 'timestamp_local'):
    """
    Clean raw dT data for sapflux. Includes function to set datetime index as well as 2 data cleaning functions, 
    which each take a np array. Each function is then applied to each column of the dataframe, which is returned at the end.
    First: Apply a low pass filter to remove all values below threshold. 
    Next: Remove [timestep] number of data points after NaN to account for heater warming up after power outage.
    Note that the timestep of the data is 5 minutes, so timestep = 12 removes 60 minutes of data.
    
    Args:
        df: dataframe with raw dT (sapflow) data where each sensor is a column
        lowpass_threshold (int, optional): threshold below which data will be discarded. Default = 1.
        warmup_timestep (int, optional): number of timesteps which will be discarded after a NaN value. Defaults to 12, which is 12x5 minutes = 1 hour.
        time_col (str, optional): name of the column containing time information for set_time(). Default = 'timestamp_local'.
    Returns:
        df
    """

    def set_time(df, time_col):
        df[time_col] = pd.to_datetime(df[time_col]) # set time column of choice to datetime
        df = df.set_index(time_col, drop = True).resample('5Min').asfreq() # set timeindex and resample
        df = df.select_dtypes([np.number]) # drop any non-number columns
        
        #print('\tDuplicated timestamps before resmaple:', df[df.duplicated(subset = 'time')].shape[0]) # Eren mentions fixing this, but I don't see any issues yet
        #print('\tTimesteps with NaN from resampling:', df.shape[0] - df.shape[0])
        print('Start:', df.index.min())
        print('Stop:', df.index.max())
        
        return df
    
    def clean_lowpass(col, lowpass_threshold = 1):
        col = np.where(col > lowpass_threshold, col, np.nan)
        return col
    
    def clean_warmup(col, warmup_timestep = 12):
        counter = 0
        while counter < warmup_timestep:
            col = np.where(np.isnan(col[-1]), np.nan, col)
            counter += 1
        return col
    
    print('\nImporting and cleaning raw data...')
    df = set_time(df, time_col)
    df = df.apply(clean_lowpass, axis = 1, raw = True)
    df = df.apply(clean_warmup, axis = 1, raw = True)
    
    return df



### Calculation methods
@helpers.timer
def calc_sapflux(df, rolling_period = 6):
    """
    Loop through columns and get the dtMax for 6 day windows and fillna
    """
    print('\nCalculating sapflux...')
    sapflux = pd.DataFrame()
    # Ensure rolling period is even, and if not, add a day
    if rolling_period%2 != 0:
        print(f'\tNote: Calculating dT taken with {rolling_period + 1} window instead of {rolling_period} days.')
        rolling_period += 1
    
    roll = int((rolling_period) / 2)

    for col in df:
        # Compute max rolling dT over [rolling_period] days, centered on day of interest
        df[col + '_max'] = df[col].shift(periods = roll*-1, freq = "D").rolling(f'{roll}d', min_periods=1).max()
        df[col + '_max'] = df[col + '_max'].fillna(method="ffill")
                            
        # Compute sapflux
        sapflux[col] = 42.84 * ((df[col + '_max'] - df[col]) / df[col])**1.231
        
    return sapflux


def calc_error():
    pass

### Plotting methods

def plot_sensors():
    pass

def plot_trees(df, raw = False):
    
    trees = list(set(([i.split('_')[5] for i in df.columns])))
    for t in trees:
        print('TREE:', t)
        df_tree = df.filter(like=t)
        fig = plt.figure(dpi = 300)
        for col in df_tree:
            plt.plot(df.index, df[col], label = col.split('_')[6], lw = 1)
            level = 'Level ' + col.split('_')[0][-2] + '-' + col.split('_')[0][-1] + ': ' + t
            title = 'L' + col.split('_')[0][-2] + col.split('_')[0][-1] + '_' + t
        plt.xticks(rotation = '45')
        plt.ylabel('Sap flux (cm/hr)')
        if raw: plt.ylabel('dT (deg C)')
        plt.title(level) 
        plt.tight_layout()
        plt.legend(ncol = 2, loc = 'upper right')
        plt.savefig('emfigs/sapflow2/' + title + '.png')
        plt.close()
        
"""
    if plot_tree:
        # This makes sure that all 4 sensors for a tree are plotted together
        for i in np.arange(1,21, step = 2):
            if tree == str(i):
                identifier.append(level + '_' + str(i+1))
        
    if id not in identifier:
        # Close out that figure if the next column is a new level/tree
        plt.legend()
        plt.savefig(save_folder + str(identifier[-1]) + '.png')
        plt.close()
        
        # Begin a new figure
        fig = plt.figure(dpi = 300)
        plt.plot(df.index, df[col], label = col, lw = 1)
        plt.xticks(rotation = '45')
        plt.ylabel('dT (deg C)')
        plt.title(id) 
        plt.tight_layout()
        
        # Restrict the xlimit to sort of match Eren's code
        if set_xlim:
            plt.xlim(pd.to_datetime('2018-06-01'), pd.to_datetime('2018-11-01'))
        
        # Change the y info if it is the calculated sapflux
        if flux:
            plt.ylim(0, 20)
            plt.ylabel('Sap flux (cm/hr)')
            
"""       
    

def plot_single_tree():
    pass



if __name__ == '__main__':
    main()