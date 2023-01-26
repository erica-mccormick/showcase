from locale import D_FMT
import pandas as pd
import json
import os
import sys
from urllib.request import urlopen
import time
from helpers import timer
import helpers

# Download the dendra_api_client.py from the GitHub repo:
# https://github.com/DendraScience/dendra-api-client-python
# Note: If your code is in a specific folder, specify the folder name plus a '/'
def download(url, folder = ''):
    filename = url.split('/')[-1]
    f = urlopen(url)
    data = f.read()
    f.close()
    with open(folder + filename, 'wb') as myfile:
        myfile.write(data)
raw_url = 'https://raw.githubusercontent.com/DendraScience/dendra-api-client-python/master/dendra_api_client.py'
download(url = raw_url, folder = 'emcode/')

# Now that the file is in the folder, we can import it:
import dendra_api_client as dendra

dendra.authenticate('rempe@jsg.utexas.edu') #rivendell2013

# parameters: start and end time
begins_at = '2017-10-01T00:00:00' 
ends_before = dendra.time_format() # time_format without argument gives current datetime. #'2020-03-01T00:00:00'

measurement = 'SapTemperatureDifference'
print('\nAccessing available ERCZO data streams for measurement: {}...'.format(measurement))
query_refinement = { 'is_hidden': False } 
measurement_list = []   # list of only datastreams that you wish to download data from
ds_list = dendra.list_datastreams_by_measurement(measurement,'',[],'erczo',query_refinement)
for ds in ds_list:
    #print(ds)
    dsm = dendra.get_meta_datastream_by_id(ds['_id'])  # This will pull full datastream metadata in JSON format
    station_name = dsm['station_lookup']['name']
    #print(station_name,ds['name'],ds['_id'])
    if ds['name'].startswith('Sap dTemp'):
        #print(ds)
        measurement_list.append(ds['_id'])
    
'''
#print('ds_list\n\n', measurement_list, '\n')
print('Extracting data for {} sensors...'.format(len(measurement_list)))

@helpers.timer
def export_datapoints_list(measurement_list = measurement_list, begins_at = begins_at, ends_before = dendra.time_format(), file_name = measurement, savedir = 'emdata/', print_out = False):
    for data in measurement_list:
        #df_temp = dendra.get_datapoints()
        print(data)

@helpers.timer
def export_datapoints(measurement_list = measurement_list, begins_at = begins_at, ends_before = dendra.time_format(), file_name = measurement, savedir = 'emdata/', print_out = False):
    df = dendra.get_datapoints_from_id_list(measurement_list, begins_at, ends_before)
    if print_out:
        print(df)
        print('Available columns:')
        for col in df.columns:
            print('\t', col)
    df.to_csv(savedir + file_name + '.csv')
'''  
@helpers.timer
def export_datapoints_individually(measurement_list = measurement_list, begins_at = begins_at, ends_before = dendra.time_format(), file_name = measurement, savedir = 'export_verbose/', print_out = True, time_name = 'timestamp_local'):
    for i in measurement_list:
        df_temp = dendra.get_datapoints_from_id_list(datastream_id_list = [i], begins_at = begins_at, ends_before = ends_before)        
    
        print('\nGetting' + i)
        print(df_temp.head())
        print('Available columns:')
        for col in df_temp.columns:
            print('\t', col)
        print('\tStart date:', df_temp.iloc[:, 0].min())
        print('\tEnd date:', df_temp.iloc[:, 0].max())
        df_temp.to_csv(savedir + file_name + '_' + i + '.csv')
    

export_datapoints_individually()
