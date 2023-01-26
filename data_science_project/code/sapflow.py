
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import functools
import time
import os, argparse, subprocess

############################## MAIN #####################################
def main():
    args = import_args()
    df = import_raw_data(file_path = os.path.join(args.data_dir, args.data_name), time_col = args.time_col)
    print('\tStart date:', df.index.min())
    print('\tStop date:', df.index.max())
    print('\tShape:', df.shape, '\n')
    
    # Plot raw dT
    plot_trees(df, fig_dir = args.fig_dir, save_dir = 'trees_raw', raw = True)

    # Clean and plot dT
    df = clean_sapflux(df, lowpass_threshold= args.lowpass_threshold, warmup_timestep = args.warmup_timestep, time_col = args.time_col)    
    plot_trees(df, fig_dir = args.fig_dir, save_dir = 'trees_clean', raw = True)

    # Calcuate sapflux and plot dT
    df = calc_sapflux(df, args.rolling_period)
    plot_trees(df, fig_dir = args.fig_dir, save_dir = 'trees', raw = False)


############################## PARSE ARGUMENTS ##############################
def import_args():
    parser = argparse.ArgumentParser('Clean and calculate sapflow from raw dT data and plot results.')
    parser.add_argument('-data_dir', type=str, default='data')
    parser.add_argument('-data_name', type=str, default='RAW_SapTemperatureDifference.csv')
    parser.add_argument('-fig_dir', type=str, default='figs')
    
    # The following arguments are for cleaning & calculating sap flux 
    # They probably shouldn't need to be adjusted.
    parser.add_argument('-lowpass_threshold', type=int, default=1) # Cleaning
    parser.add_argument('-warmup_timestep', type=int, default=12) # Cleaning
    parser.add_argument('-rolling_period', type=int, default=6) # Calculating flux
    parser.add_argument('-time_col', type=str, default='timestamp_local') # Cleaning
    
    print('\nCleaning and parsing arguments for sapflow.py...')
    args = parser.parse_args()

    print('\nChecking and generating figs directory...')
    if os.path.isdir(args.fig_dir) == False:
        subprocess.call('mkdir ' + os.path.join(args.fig_dir), shell=True)
        
    # These directories are made for storing figures. 
    # If the save_dir for plot_trees() is changed, these can change/be removed too.
    if os.path.isdir(args.fig_dir + '/trees') == False:
        subprocess.call('mkdir ' + os.path.join(args.fig_dir, 'trees'), shell=True)
    if os.path.isdir(args.fig_dir + '/trees_raw') == False:
        subprocess.call('mkdir ' + os.path.join(args.fig_dir, 'trees_raw'), shell=True)
    if os.path.isdir(args.fig_dir + '/trees_clean') == False:
        subprocess.call('mkdir ' + os.path.join(args.fig_dir, 'trees_clean'), shell=True)

    return args

############################## ANALYSIS METHODS ##############################

def _timer(func):
    """
    Decorator to print the time it took to run a function.
    Based off of RealPython tutorial.
    """
    @functools.wraps(func) # Ensures metadata is carried through
    def wrapper_timer(*args, **kwargs):
        t0 = time.perf_counter()
        value = func(*args, **kwargs)
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        if elapsed_time > 60:
            print(f"Elapsed time: {elapsed_time/60:0.4f} minutes")
        else: print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer
    
def import_raw_data(file_path, time_col):
    df = pd.read_csv(file_path)
    if type(df.index) != pd.core.indexes.datetimes.DatetimeIndex:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col, drop = True)
    return df


@_timer
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
    def clean_null(col, null_value = 7999):
        col = np.where(col == null_value, np.nan, col)
        return col
    
    def clean_lowpass(col, lowpass_threshold = 1):
        col = np.where(col >= lowpass_threshold, col, np.nan)
        return col
    
    def clean_warmup(col, warmup_timestep = 12):
        counter = 0
        while counter < warmup_timestep:
            col = np.where(np.isnan(col[-1]), np.nan, col)
            counter += 1
        return col
    
    print('\nImporting and cleaning raw data...')

    df = df.apply(clean_null, null_value = 7999)
    df = df.apply(clean_lowpass, axis = 1, raw = True)
    df = df.apply(clean_warmup, axis = 1, raw = True)
    
    return df


@_timer
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
    raise NotImplementedError

############################## PLOTTING METHODS ##############################

def plot_sensors():
    raise NotImplementedError

def plot_trees(df, fig_dir, save_dir, raw = False):
    trees = list(set(([i.split('_')[5] for i in df.columns])))
    for t in trees:
        print('Plotting tree:', t)
        df_tree = df.filter(like=t)
        fig = plt.figure(dpi = 300)
        for col in df_tree:
            plt.plot(df.index, df[col], label = col.split('_')[6], lw = 1)
            level = 'Level ' + col.split('_')[0][-2] + '-' + col.split('_')[0][-1] + ': ' + t
            title = 'L' + col.split('_')[0][-2] + col.split('_')[0][-1] + '_' + t

        plt.xticks(rotation = '45')
        plt.ylabel('Sap flux (cm/hr)')
        if raw: 
            plt.ylabel('dT (deg C)')
            title = title + '_dT'
        plt.title(level) 
        plt.tight_layout()
        plt.legend(ncol = 2, loc = 'upper right')
        path = fig_dir + '/' + save_dir + '/' + title + '.png'
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    main()