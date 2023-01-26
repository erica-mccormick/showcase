import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import datetime 
from datetime import datetime, timedelta
from operator import *
from matplotlib.dates import DateFormatter
import glob
import helpers
import warnings

#ignore by message
warnings.filterwarnings("ignore", message='Creating legend with loc="best" can be slow with large amounts of data.')
warnings.filterwarnings("ignore", message='No handles with labels found to put in legend.')


### Import raw sapflow files
# Name of file corresponds to level and what it contains
# Each tree has 4 columns: 1a/b and 2a/b. If 3a/b and 4a/b exist, that is a second tree.

def main():
    plot_intermediates = False
    
    # Import and clean raw sapflow files and merge into one df with 5min timestep
    df = import_files()
    print('Available columns:', df.columns.unique())
    
    # Do a first pass at cleaning obviously wrong data and plot results
    df_clean = clean(df)
    
    df_calc = calculate_sapflow(df_clean)
    
    # Plot raw and cleaned dT
    if plot_intermediates:
        plot_sensors(df, save_folder = 'emfigs/raw_sapflow/', set_xlim = True)
        plot_sensors(df_clean, save_folder = 'emfigs/clean_sapflow/', set_xlim = True)



################# METHODS #################

def import_files(path_to_raw_sapflow = 'emdata/raw_sapflow'):
    # Initialize df for merging
    df = pd.DataFrame()
    
    # Loop through every file in the raw_sapflow folder
    for file in glob.glob(path_to_raw_sapflow + '/*'):
        print('\nImporting and cleaning', file, '...')
        
        # Read in a single file and clean
        temp = pd.read_csv(file)
        temp = temp.reset_index(drop = True, inplace = False)
        
        # Rename the columns: rename_time = False leaves the 'time' column as is
        temp = helpers.rename_cols(temp, file_path = file, rename_time = False)
        
        # Convert to datetimeindex, resample to 5min, and print summary
        temp['time'] = pd.to_datetime(temp['time'])
        print('\tDuplicated timestamps:', temp[temp.duplicated(subset = 'time')].shape[0]) # Eren mentions fixing this, but I don't see any issues yet
        shape_before_resample = temp.shape[0]
        temp = temp.set_index('time', drop = True).resample('5Min').asfreq()
        shape_after_resample = temp.shape[0]
        print('\tTimesteps with NaN from resampling:', shape_after_resample - shape_before_resample)
        print('\tStart:', temp.index.min())
        print('\tStop:', temp.index.max())
    
        # Merge cleaned file with the rest of the sapflow files
        df = df.merge(temp, how = 'outer', left_index = True, right_index = True)
        
    return df  


def plot_sensors(df, save_folder, set_xlim = False):
    """
    Iterate through columns and plot each level/tree combination on a new plot.
    Figures are named with level (as in L32 for Level 3-2) and the sensor (ie 1-4)
    and each plot contains the data for sensor a and b. A single tree has 4 sensors, such as
    1a/b and 2a/b. Sensors denoted with 3 and 4 indicate that 2 trees are on a single multiplexer.
    See the spreadsheet or dendra metadata for the key to the sensor/tree/multiplexer setup.
    
    """
    # Make sure there are no open figures
    plt.close()
    
    # Sort columns lexographically
    df = df.sort_index(axis=1)
    
    # Set up lists to keep track of the previously plotted combos
    identifier = [0]
    
    # Initialize a plot to have something to close on the first round
    #plt.plot(0, 0)
    
    for col in df:
        # Only plot sapflow
        var_type = col.split('_')[1]
        if var_type == 'sap':
            # Use the column name to get the level and tree codes
            level = col.split('_')[0]
            tree = col.split('_')[2][0]
            id = level + '_' + tree
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
                if set_xlim:
                    plt.xlim(pd.to_datetime('2018-06-01'), pd.to_datetime('2018-11-01'))
                
            # Add more sensors to a figure if its the same level/tree configuration
            else: plt.plot(df.index, df[col], label = col, lw = 1)

            # Add identifier to list to keep track of what has been plotted
            identifier.append(id)
                
def clean(df):
    print('\nCleaning sapflow...')
    # Just keep the columns that are truly sapflow
    # Try/except used because some column names are duplicated and drop will fail
    for col in df:
        # Snippet of code that drops columns that aren't sapflow
        if col.split('_')[1] == 'sap' and col.split('_')[2] != 'heatervoltage':
            #print(col, ':\n\tNaNs in raw:', df[col].isna().sum())
            
            # Scenario: Disconnected heaters cause low values
            # Solution: low-pass filter for values <1
            df[col] = df[col].where(df[col] > 1, np.nan)
            #print('\tNaNs after low-pass filter:', df[col].isna().sum())
            
            # Scenario: Heaters are not warmed up after power outage causing unreliable data
            # Solution: Remove 1hr of data after null values are present
            counter = 0
            while counter < 12:
                # If the previous row was NaN, set this row to NaN. Do this 12 times to represent 60 minutes.
                df[col] = np.where(df[col].shift(1).isnull(), np.nan, df[col])
                counter += 1
            #print('\tNaNs after dropping 1hr after NaN:', df[col].isna().sum())


    return df
            

def calculate_sapflow(df):
    """
    Equation from Eren's code and not checked by Erica:
    
    Grainer, 1987:
        u =  a * [(dTMax-dT)/dT]^b
        where:
            u = sap flux density (aK^b)
            dTMax here is the max 5 day dT
            a = 119x10e-6 m/s, 42.84 cm/hr
            b = 1.231
            
    Note from Eren about possible issues:
    
    "Note, this is for a heat field around the top probe such
    that dTMax~10C. In the first row we have a range of dTMax
    of 6.7C to 12.05C. Because we are applying a constant voltage
    to heaters/resistors of known resistance, this range of dTMax 
    is likely due to variations in madrone trunk heat capacity, 
    either due to 1) variations in wood matrix properties, 
    2) variations in volumetric water content of the wood/water matrix, or 
    3) some combination thereof. How this range may affect the accuracy 
    of the empirically-derived constants in Granier's equation is unknown. 
    Any application of this data that wishes to arrive at concrete estimates
    of volumetric water extraction, for example for water budget closures in
    combination with other extensive hydrological data available at this site, 
    will have to grapple with this issue (among others)."
        
    """
  
    df2 = df.shift(periods = -3, freq = "D").rolling('3d', min_periods=1).max()
    df2 = df2.fillna(method="ffill")
   
42.84 * ((dTmax - dT) / dT)   **1.231
def plot_voltages(df, col):
    # Eventually would be good to plot the battery stuff with the sapflow stuff
    # To see if we can get rid of the low values that remain more systematically
    pass
    

    
if __name__ == '__main__':
    main()