# THIS CODE EXTRACTS RAW SAP FLUX DATA FROM THE DENDRA API,
# CLEANS IT, CALCULATES SAP FLUX, AND PLOTS RESULTS FOR EACH TREE. 

# Run this code from the base directory by typing the following into the terminal:
### 'sh analyze_sapflow.sh'


# ARGUMENTS FOR DENDRA.PY
######## -skip (str): If True, skip authentication and extraction from Dendra (default False)
######## -export_slow (str): Use the slow (foolproof) method or the fast extraction method (default True)
######## -username (str): Dendra username (MUST SPECIFY THIS)
######## -measurement (str): the measurement to extract. No guarantee the rest of the analysis will work with anything else (default 'SapTemperatureDifference')
######## -site (str): the site of interest. No guarantee the rest of the analysis will work with anything else (default 'erczo')
######## -begins_at (str): the date (inclusive) to begin extracting data (default '2018-01-01T00:00:00')
######## -ends_before (str): the date (exclusive) to stop extracting (default 'now', which is converted to current day)
######## -time_col (str): the name for the column containing time information (default 'timestamp_local')

python3 code/dendra.py -skip False -username rempe@jsg.utexas.edu
  

# ARGUMENTS FOR SAPFLOW.PY
# These may need to be specified if folder set-up changes:
######## -data_dir (str): location of the folder where csvs should be found/stored (default 'data')
######## -data_name (str): filename of the raw dT data (default 'RAW_SapTemperatureDifference.csv')
######## -fig_dir (str): location of the folder where figures should be stored (default 'figs'). Folder is created if doesn't exist.
# These probably don't need to be messed with, but the option exists:
######## -lowpass_threshold (int): cutoff for the lowpass filter (default=1)
######## -warmup_timestep (int): how many data points to remove after NaN (to account for heater warmup) (default=12, equaling 1hr of data)
######## -rolling_period (int): how many data points to roll over for maximum dT in sapflux calculation (default=6)
######## -time_col (str): the name for the column containing time information (default 'timestamp_local')

python3 code/sapflow.py -data_dir data -data_name merged_SapTemperatureDifference.csv


# For saving a new environment.yml:
# $ conda env export --name dendra-env --from-history --file environment.yml
# To 

