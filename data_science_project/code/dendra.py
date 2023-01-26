import argparse
import os
import subprocess
import time
from urllib.request import urlopen

import numpy as np
import pandas as pd

from sapflow import _timer

########################## IMPORT DENDRA_API_CLIENT ##############################
# Note: If you end up downloading a new copy from GitHub, make the following changes:
# Line 431, change merge type to 'outer' to preserve all timesteps.
try:
    import dendra_api_client as dendra
except ModuleNotFoundError:
    print('\nDownloading the dendra_api_client module from:')
    url = 'https://raw.githubusercontent.com/DendraScience/dendra-api-client-python/master/dendra_api_client.py'
    print(url)
    with urlopen(url) as f:
        data = f.read()
    with open('code/' + url.split('/')[-1], 'wb') as myfile:
        myfile.write(data)
    import dendra_api_client as dendra


######################################## MAIN ####################################
def main():

    args = import_args()
    if args.skip == 'False':
        dendra_authenticate(args.username)
        datastream_id_list = get_ids(args.measurement, args.site)

        if args.export_slow == 'False':
            print('\nExtracting from Dendra using fast method...')
            export_as_list(args.data_dir, datastream_id_list,
                           args.measurement, args.begins_at, args.ends_before)

        # Split datastream list into two parts and export
        else:
            print('\nExtracting from Dendra using slow method...')
            id_list_p1 = datastream_id_list[0:30]
            id_list_p2 = datastream_id_list[30:]
            df1 = export_data_long(args.data_dir, 'part1', id_list_p1,
                                   args.measurement, args.begins_at, args.ends_before, args.time_col)
            df2 = export_data_long(args.data_dir, 'part2', id_list_p2,
                                   args.measurement, args.begins_at, args.ends_before, args.time_col)

            # Merge parts together and export
            df = df1.merge(df2, how='outer', left_index=True, right_index=True)
            df.to_csv(args.data_dir + '/merged_SapTemperatureDifference.csv')

    else:
        print('\nSkipping Dendra extraction...')


############################## PARSE ARGUMENTS ##############################
def import_args():
    parser = argparse.ArgumentParser('Download data from Dendra.')
    parser.add_argument('-skip', type=str, default='False')
    parser.add_argument('-export_slow', type=str, default='True')
    parser.add_argument('-data_dir', type=str, default='data')
    parser.add_argument('-username', type=str, default='')
    parser.add_argument('-measurement', type=str,
                        default='SapTemperatureDifference')
    parser.add_argument('-site', type=str, default='erczo')
    parser.add_argument('-begins_at', type=str, default='2018-01-01T00:00:00')
    parser.add_argument('-ends_before', type=str, default='now')
    parser.add_argument('-time_col', type=str,
                        default='timestamp_local')  # Cleaning
    print('\nCleaning and parsing arguments for dendra.py...')
    args = parser.parse_args()
    return args


################################## METHODS ######################################
def dendra_authenticate(username):
    """
    Authenticate dendra with the username. User will be prompted to type in password.
    Args:
        username (str): Dendra username. Password will then be requested.
    """
    try:
        print('Please enter your Dendra password:')
        dendra.authenticate(username)
    except AssertionError:
        print('Dendra password or username is not correct. Please re-start code and try again.')
        exit()


def format_timeindex(df, time_col='timestamp_local', frequency='5Min', drop_non_numeric=True, verbose=False):
    # Check that the df is in datetime index, and if not, set the time_col to date format and make it an index.
    if type(df.index) != pd.core.indexes.datetimes.DatetimeIndex:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col, drop=True)
    # Resample to frequency
    df = df.resample(frequency).asfreq()
    # Optionally drop all non-numeric columns (for example, other date columns)
    if drop_non_numeric:
        df = df.select_dtypes([np.number])
    return df

# DOES STARTSWITH WORK FOR OTHER KINDS OF DATA??


def get_ids(measurement, site):
    print(
        f'\nAccessing data streams at {site.upper()} for measurement = {measurement}...')
    query_refinement = {'is_hidden': False}
    measurement_list = []
    ds_list = dendra.list_datastreams_by_measurement(
        measurement, '', [], site, query_refinement)
    for ds in ds_list:
        if ds['name'].startswith('Sap dTemp'):
            measurement_list.append(ds['_id'])
    return list(measurement_list)


def get_longest_id(measurement, site):
    measurement_list = get_ids(measurement, site)
    for i in measurement_list:
        meta = dendra.get_meta_datastream_by_id(i)
        print('\n', meta['name'])
        print(meta['extent']['begins_at'])
        print(meta['extent']['ends_before'])


@_timer
def export_data_long(path_to_data,  file_prefix, measurement_list, measurement, begins_at='2018-01-01T00:00:00', ends_before='now', time_col='timestamp_local', verbose=False):
    """
    This function gets the entirety of each measurement id between begins_at and ends_before, one at a time.
    It is MUCH slower than the threaded export_as_list(), however it definitely gets every timestep. 
    It is unclear if export_as_list() is still creating gaps in the data, so if you have concerns, use this function.
    Note that you may need to break up the measurement ids (from the measurement_list = get_ids() func) into multiple
    parts to avoid a timeout error. However, all parts are saved as they are extracted so you won't lose anything if a timeout error does occur.
    """
    # Get current date in correct format
    if ends_before == 'now':
        ends_before = dendra.time_format()
    print(
        f'\nExtracting {len(measurement_list)} data streams from {begins_at} to {ends_before}...')
    # Initialize new df and start timer and counter (for printing out progress and total elapsed time of extraction)
    df = pd.DataFrame()
    t0 = time.time()
    counter = 0
    # Loop through IDs and extract data. Export to csvs along the way.
    for i in measurement_list:
        df_temp = dendra.get_datapoints_from_id_list(
            datastream_id_list=[i], begins_at=begins_at, ends_before=ends_before)
        ###df_temp.to_csv(path_to_data + measurement + '_' + i + '.csv')
        df_temp = format_timeindex(df_temp, time_col=time_col, verbose=verbose)
        df = df.merge(df_temp, how='outer', left_index=True, right_index=True)
        counter += 1
        print(f'\tExported {i}, shape {df_temp.shape}')
        print(f'\t{len(measurement_list) - counter} remaining.')
    t1 = time.time()
    print(f'Elapsed time: {t1-t0}')
    # Export final csv with merged and correctly time-indexed data. Each column is a sensor.
    df.to_csv(path_to_data + '/' + file_prefix + '_' + measurement + '.csv')

    return df


@_timer
def export_as_list(path_to_data, datastream_id_list, measurement, begins_at, ends_before):
    t00 = time.time()
    if len(datastream_id_list) > 20:
        datastream_id_list = [datastream_id_list[i::3] for i in range(3)]
    df = pd.DataFrame()
    for ids in datastream_id_list:
        t0 = time.time()
        df_temp = dendra.get_datapoints_from_id_list(
            ids, begins_at, ends_before)
        df_temp = format_timeindex(df_temp)
        t1 = time.time()
        print(f'Chunk elapsed time: {round(t1-t0)} seconds')
        df = df.merge(df_temp, how='outer', left_index=True, right_index=True)
    df.to_csv(path_to_data + '/RAW_' + measurement + '.csv')
    t11 = time.time()
    print('Final csv exported to:', path_to_data + 'RAW_' + measurement + '.csv')
    print(f'TOTAL ELAPSED TIME: {round((t11-t00)/60)} minutes')
    return df

# Potential solution to losing datapoints would be to build the dendra extract off of an existing timeindex df:
#df = pd.DataFrame(index = pd.date_range(start = begins_at, end=ends_before, freq=frequency))


if __name__ == '__main__':
    main()
