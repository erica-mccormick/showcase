from urllib.request import urlopen
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
import helpers

def main():
    #### THINGS THE USER NEEDS TO SET: ####
    use_dendra = True
    path_to_data = '' # if not using dendra
    
    if use_dendra:
        # Import dendra api and authenticate
        # Return the path to the data (equivelant to folder_csvs argument)
        path_to_data = extract_dendra(username = 'rempe@jsg.utexas.edu', measurement = 'SapTemperatureDifference',
                    path_to_code = 'emcode/', path_to_data = 'emdata/verbose/',
                    begins_at = '2017-10-01T00:00:00', ends_before = 'now',
                    time_col = 'timestamp_local')
    

    df = df.clean_sapflux(df)
    df = df.calc_sapflux(df)
    df.to_csv('final_sapflow.csv')
    plot_trees(df)
    print(df)
    
    
########################### ANALYSIS FUNCTIONS ###########################
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
    def clean_null(col, null_value = 7999):
        col = np.where(col == null_value, col, np.nan)
        return col
    
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
    #df = set_time(df, time_col) # Used to be here, now is with dendra
    df = df.apply(clean_null, null_value = 7999)
    df = df.apply(clean_lowpass, axis = 1, raw = True)
    df = df.apply(clean_warmup, axis = 1, raw = True)
    
    return df



### Calculation methods
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
        plt.savefig('emfigs/final_sapflow/' + title + '.png')
        plt.close()


########################### DENDRA FUNCTIONS ###########################

def extract_dendra(username, measurement = 'SapTemperatureDifference',
                   site = 'erczo', path_to_code = 'emcode/', path_to_data = 'emdata/verbose/',
                   begins_at = '2017-10-01T00:00:00', ends_before = 'now', time_col = 'timestamp_local'):

    def set_time(df, time_col):
        #df[time_col] = pd.to_datetime(df[time_col]) # set time column of choice to datetime
        #df = df.set_index(time_col, drop = True).
        df = df.resample('5Min').asfreq() # set timeindex and resample
        df = df.select_dtypes([np.number]) # drop any non-number columns
        #print('\tDuplicated timestamps before resmaple:', df[df.duplicated(subset = 'time')].shape[0]) # Eren mentions fixing this, but I don't see any issues yet
        #print('\tTimesteps with NaN from resampling:', df.shape[0] - df.shape[0])
        #print('Start:', df.index.min())
        #print('Stop:', df.index.max())
        return df
    
    def dendra_import_api():
        def download(url, folder = ''):
            filename = url.split('/')[-1]
            f = urlopen(url)
            data = f.read()
            f.close()
            with open(folder + filename, 'wb') as myfile:
                myfile.write(data)
        raw_url = 'https://raw.githubusercontent.com/DendraScience/dendra-api-client-python/master/dendra_api_client.py'
        download(url = raw_url, folder = path_to_code)

    # Import dendra api by downloading .py file to directory and authenticate
    dendra_import_api()
    import dendra_api_client as dendra
    dendra.authenticate(username)  #rivendell2013
    
    # Set ends_before now that dendra is imported
    if ends_before == 'now': ends_before = dendra.time_format()
        
    def dendra_get_data_list():
        print('\nAccessing available ERCZO data streams for measurement: {}...'.format(measurement))
        query_refinement = { 'is_hidden': False } 
        measurement_list = [] 
        ds_list = dendra.list_datastreams_by_measurement(measurement,'',[],site,query_refinement)
        for ds in ds_list:
            if ds['name'].startswith('Sap dTemp'): measurement_list.append(ds['_id'])
        return measurement_list

    def export_datapoints_individually(measurement_list, begins_at = begins_at, ends_before = ends_before, path_to_data = path_to_data, measurement = measurement, time_col = time_col):
        print(f'Extracting {len(measurement_list)} data streams from {begins_at} to {ends_before}...')
        df = pd.DataFrame()
        t0 = time.time()
        counter = 0
        for i in measurement_list:
            df_temp = dendra.get_datapoints_from_id_list(datastream_id_list = [i], begins_at = begins_at, ends_before = ends_before)        
            #try:
            #    del df_temp['timestamp_utc']
            #    del df_temp['q']
            #except: continue
            df_temp.to_csv(path_to_data + measurement + '_' + i + '.csv')
            df_temp = set_time(df_temp, time_col = time_col)
            df = df.merge(df_temp, how = 'outer', left_index = True, right_index = True)
            counter += 1
            print(f'\tExported {i}, shape {df_temp.shape}')
            print(f'\t{len(measurement_list) - counter} remaining.')
        t1 = time.time()
        print(f'Elapsed time: {t1-t0}')
        df.to_csv('MERGED_' + measurement + '.csv')
        
    # Run functions    
    measurement_list = dendra_get_data_list()
    export_datapoints_individually(measurement_list)
    
    
    
    
        
        
if __name__ == '__main__':
    main()