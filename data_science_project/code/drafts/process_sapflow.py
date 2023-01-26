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

#ignore warnings by message
warnings.filterwarnings("ignore", message='Creating legend with loc="best" can be slow with large amounts of data.')

### TO DO ###:
# Implement logger decorator instead of printing title, etc in funcs
# Fix helpers.rename_cols() to work with dendra exported data


### Import raw sapflow files
# Name of file corresponds to level and what it contains
# Each tree has 4 columns: 1a/b and 2a/b. If 3a/b and 4a/b exist, that is a second tree.

def main():
    plot_intermediates = False
    
    # Import and clean raw sapflow files and merge into one df with 5min timestep
    # Simultaneously, get a dictionary mapping the column names to the tree name and species
    df, name_dict = import_data()
    
    #print('Available columns:', df.columns.unique())
    
    # Do a first pass at cleaning obviously wrong data and plot results at tree level
    sapflux, df_clean = clean_and_calculate(df)
    plot_trees(sapflux, name_dict, save_folder = 'emfigs/sapflow/')
    
    # Plot raw and cleaned dT
    if plot_intermediates:
        print('\nPlotting intermediate data steps...')
        plot_sensors(df, save_folder = 'emfigs/dt_raw/', plot_tree = False, set_xlim = False, flux = False)
        plot_sensors(df_clean, save_folder = 'emfigs/dt_cleaned/', plot_tree = False, set_xlim = False, flux = False)



################# METHODS #################

def import_data(path, multiple_files = False, **kwargs):
    """
    Import sapflow data from files extracted from dendra, either manually or with the API.
    If multiple_files = True, all of the files in the FOLDER 'path' will be merged on time.
    If multiple_files = False, a single file, specified with the PATH TO THE FILE as 'path' will be imported.
    This code handles all of the column re-naming and also returns a dictionary that has the column:tree information for 
    naming plots and querying tree species and name down the road.
    
    Args:
        path (str): Either the path to the folder (if multiple_files = True) or the csv (multiple_files = False) that contains sapflow data. Other data may be present as well, but only sapflow data will be processed.
        multiple_files (bool, optional): default = False. If False, extract a single file with the path given by 'path'. Else, extract all of the files in the path-to-folder given in 'path'.
        **rename_time ()
    Returns:
        df, dictionary: dataframe with the data merged on time and a dictionary with column_name:tree info. 
    """
    def import_single_file(path):
        # Read in a single file, set index, and rename columns
        # rename_time = False leaves the 'time' column as is
        print('\nImporting and cleaning', path, '...')
        data = pd.read_csv(path)
        data = data.reset_index(drop = True, inplace = False)
        data, name_dict = helpers.rename_cols(data, file_path = path, rename_time = False)
        # Convert to datetimeindex, resample to 5min
        data['time'] = pd.to_datetime(data['time'])
        data_resampled = data.set_index('time', drop = True).resample('5Min').asfreq()
        # Print summary info
        print('\tDuplicated timestamps before resmaple:', data[data.duplicated(subset = 'time')].shape[0]) # Eren mentions fixing this, but I don't see any issues yet
        print('\tTimesteps with NaN from resampling:', data_resampled.shape[0] - data.shape[0])
        print('\tStart:', data_resampled.index.min())
        print('\tStop:', data_resampled.index.max())
        # Return resample df and naming dictionary
        return data_resampled, name_dict
    
    # Loop through all files in path folder
    if multiple_files:
        df = pd.DataFrame()
        name_dict = {}
        for file in glob.glob(path + '/*'):
            temp, name_dict_temp = import_single_file(file)
            # Merge cleaned file with the rest of the sapflow files
            df = df.merge(temp, how = 'outer', left_index = True, right_index = True)
            name_dict.update(name_dict_temp)
    # If just one file, import it
    else: df, name_dict =  import_single_file(path)
    return df, name_dict


def plot_sensors(df, save_folder, plot_tree = False, set_xlim = False, flux = True):
    """
    Iterate through columns and plot each level/tree combination on a new plot.
    Figures are named with level (as in L32 for Level 3-2) and the sensor (ie 1-4)
    and each plot contains the data for sensor a and b. A single tree has 4 sensors, such as
    1a/b and 2a/b. Sensors denoted with 3 and 4 indicate that 2 trees are on a single multiplexer.
    See the spreadsheet or dendra metadata for the key to the sensor/tree/multiplexer setup.
    
    """
    if plot_tree: statement = 'at tree level...'
    else: statement = 'for all sensor pairs...'
    print('\nPlotting in-process data', statement)
   
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
                    
            # Add more sensors to a figure if its the same level/tree configuration
            else: plt.plot(df.index, df[col], label = col, lw = 1)

            # Add identifier to list to keep track of what has been plotted
            identifier.append(id)

           

def plot_trees(df, name_dict, save_folder):
    """
    Iterate through columns and plot each level/tree combination on a new plot.
    Figures are named with level (as in L32 for Level 3-2) and the sensor (ie 1-4)
    and each plot contains the data for sensor a and b. A single tree has 4 sensors, such as
    1a/b and 2a/b. Sensors denoted with 3 and 4 indicate that 2 trees are on a single multiplexer.
    See the spreadsheet or dendra metadata for the key to the sensor/tree/multiplexer setup.
    
    """
    print('\nPlotting sapflux for all trees...')
    # Convert dictionary to set to get unique values
    names_unique = set(val for val in name_dict.values())
    # For unique tree names, plot all of the sensors
    for name in names_unique:
        if name.startswith('madrone') or name.startswith('doug'):
            cols_to_plot = [key for key, val in name_dict.items() if name==val]
            df_plot = df[cols_to_plot]
            df_plot.dropna(axis=0, how='all')
            for col in df_plot: plt.plot(df_plot.index, df_plot[col], label = col, alpha = 0.4)
            plt.ylim(0,20)
            plt.ylabel('sap flux (cm/hr)')
            plt.title(name)
            plt.legend()
            plt.xticks(rotation = 90)
            plt.tight_layout()
            plt.savefig(save_folder + name + '.png')
            plt.close()
        

                
def clean_and_calculate(df):
    
    """
    CLEANING RAW DT:
        Applies a low-pass filter to get rid of values <1
        Removes 1 hour of data after every NaN to account for heater warming up after power outage
        There are still data issues remaining that likely stem from:
            (a) Wiring issues (Eren talks about these in depth; the answer is to remove the data)
            (b) Issues with voltage or power from specific power sources or voltage regulators to specific sensors
        Erica will try to make code that systematically addresses this without outside knowledge of connections at some point in the future.
        
    CALCULATING SAP FLUX:
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
    print('\nCleaning sensor data and calculating sapflux...')
    # Initialize a new dataframe to hold the final sapflow data
    sapflux = pd.DataFrame()
    for col in df:
        if col.split('_')[1] == 'sap' and col.split('_')[2] != 'heatervoltage':
            ### CLEAN RAW DT ###
            # Scenario: Disconnected heaters cause low values
            # Solution: low-pass filter for values <1
            df[col] = df[col].where(df[col] > 1, np.nan)
            
            # Scenario: Heaters are not warmed up after power outage causing unreliable data
            # Solution: Remove 1hr of data after null values are present
            counter = 0
            while counter < 12:
                # If the previous row was NaN, set this row to NaN. Do this 12 times to represent 60 minutes.
                df[col] = np.where(df[col].shift(1).isnull(), np.nan, df[col])
                counter += 1
   
            ### CALCULATE SAP FLUX ###
            # Loop through columns and get the dtMax for 6 day windows and fillna
            new_col_name = col + '_max'
            df[new_col_name] = df[col].shift(periods = -3, freq = "D").rolling('3d', min_periods=1).max()
            df[[new_col_name]] = df[[new_col_name]].fillna(method="ffill")
            
            # Calculate sapflux and add data to new dataframe (with same column names)
            sapflux[col] = 42.84 * ((df[new_col_name] - df[col]) / df[col])**1.231

    return sapflux, df
            



def plot_voltages(df, col):
    # Eventually would be good to plot the battery stuff with the sapflow stuff
    # To see if we can get rid of the low values that remain more systematically
    pass
    

if __name__ == '__main__':
    main()