import pstats
import pandas as pd
import numpy as np
import functools
import time
    # Old column format: sap-d-temp-tdp-80-madrone-36-whiskey-2-b-c-avg-degc  
    # If you have a dictionary 'name' that has key=column, value=tree
    # You could loop through to get the columns to keep for plotting individual trees:  
    # cols = [key for key, val in names.items() if "MadroneA"==val]


def tracer(func):
    @functools.wraps(func)
    def wrapper_tracer(*args, **kwargs):
        print(f'\nRunning {func.__name__}...')
        return func(*args, **kwargs)
    return wrapper_tracer        

def timer(func):
    """
    Give the time it took to run a function.
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


def parse_tree_name(col):
    """
    
    Uses knowns pecies of Madrone and DougFir to parse the name and species of the tree types
    associated with sapflow columns in order to make a dictionary for plotting names later.
    Note: This works with Rivendell sapflow data naming conventions, but has not been tested for other sites (like Rancho Venada).
    This code is called in rename_cols().
    
    """
    if col.split('-')[5] == 'madrone':
        # If the tree number comes before the name:
        if col.split('-')[6].isnumeric():
            tree_id = col.split('-')[5] + '_' + col.split('-')[7]
        # If the tree name comes before the number:
        else: tree_id = col.split('-')[5] + '_' + col.split('-')[6]
            
    # Split off doug because doug fir is 2 words
    elif col.split('-')[5] == 'doug':
        # If the tree number comes before the name:
        if col.split('-')[7].isnumeric():
            # There is one doug fir named 'x ray' which takes 2 places,
            # So if the name is one character long, include the next place in the name too:
            if len(col.split('-')[8]) == 1:
                tree_id = col.split('-')[5] + col.split('-')[6] + '_' + col.split('-')[8] + col.split('-')[9]
            else: tree_id = col.split('-')[5] + col.split('-')[6] + '_' + col.split('-')[8]
        # If the tree name comes before the number:
        else: tree_id = col.split('-')[5] + col.split('-')[6] + '_' + col.split('-')[7]
    # Only columns that start with 'sap' go through this, but some are still heatervoltage:
    else: tree_id = 'not_a_tree'
    return tree_id


def rename_cols_dendra(df, **kwargs):
    pass
    
def rename_cols(df, **kwargs):
    """
    Rename the columns from dendra file downloads by taking the first word of the column name,
    which corresponds to the variable, and then appending the tree information for sapflow only.
    Then, either add a prefix to the beginning of the column, or use the file path to add the level, if files
    have been named accordingly. If no file_path or prefix kwargs are given, there will be no prefix added to the column names.
    Returns a dataframe with the renamed columns.
    
    Args:
        df: a dataframe from a dendra csv file with the original column names
        **file_path (str, optional): the file path string for automatically generating a level prefix for the columns
        **prefix (str, optional): a custom prefix to the column names
        **rename_time (bool, optional, default = True): rename the 'time' column following the above. If false, leave 'time' as is.
        **time_name (str, optional, default = 'time'): the name of the original 'time' column
    Returns:
        df
    
    """
    default_kwargs = {
        'file_path': '',
        'prefix': '',
        'rename_time': True,
        'time_name': 'time'}
    kwargs = {**default_kwargs, **kwargs}
    
    
    new_cols = []
    old_cols = []
    tree_ids = []
    for col in df.columns.unique():
        old_cols.append(col)
        # If column begins with 'sap', make the new name 'sap_' + 1a, 1b, etc
        if col.split('-')[0] == 'sap': 
            name = 'sap_' + col.split('-')[-5] + col.split('-')[-4]
            tree_id = parse_tree_name(col)
        # Otherwise, make the new name the first word of the column (ie temperature, voltage, etc)
        else:
            name = col.split('-')[0]
            tree_id = 'not_a_tree'
        new_cols.append(name)
        tree_ids.append(tree_id)

    # At the beginning of each column name
    # Either: add the level identifier from the beginning of the file name (L32, etc)
    # Or: Add a generic prefix
    if kwargs['file_path']:
        new_cols = [kwargs['file_path'].split('/')[-1][0:3] + '_' + i for i in new_cols] # add level prefix
    elif kwargs['prefix']:
        new_cols = [kwargs['prefix'] + '_' + i for i in new_cols] # add level prefix

    # Convert old and new column lists to a dictionary and rename
    col_rename = {old_cols[i]: new_cols[i] for i in range(len(old_cols))}
    df = df.rename(columns = col_rename, inplace = False)
    
    # Dictionary of tree names to column names
    name_dict = dict(zip(new_cols, tree_ids))
    
    if kwargs['rename_time'] == False:
        if kwargs['file_path']:
            df = df.rename(columns = {kwargs['file_path'].split('/')[-1][0:3] + '_' + kwargs['time_name']: kwargs['time_name']}, inplace = False)
        elif kwargs['prefix']:
            df = df.rename(columns = {kwargs['prefix'] + '_' + kwargs['time_name']: kwargs['time_name']}, inplace = False)
    
    return df, name_dict


def drop_cols(df):
    """ 
    NOT BEING USED / NOT A REAL FUNCTION
    Chunk of code that might be helpful in the future for isolating specific columns by name.
    """
    for col in df:
        # Snippet of code that drops columns that aren't sapflow
        if col.split('_')[1] != 'sap' or col.split('_')[2] == 'heatervoltage':
            try: df.drop(col, axis = 1, inplace = True)
            except: KeyError
        

def find_anomaly(df):
    """
    NOT BEING USED / NOT A REAL FUNCTION
    Chunk of code that might be helpful in the future when you want to 
    zoom in to a particular spot on a dataframe and print it out.
    """
    df2 = df[df.index > pd.to_datetime('2018-06-13')]
    df2 = df2[df2.index < pd.to_datetime('2018-06-14')]
    df2 = df2[['L51_sap_3b']]
    
    for i in np.arange(0, 600, step = 50):
        print(df2[i:i+50])

    low_data = df2['L51_sap_3b'].min()
    print('Min L51_sap_3b data:', low_data)
    
    diagnose = df2[df2['L51_sap_3b'] == low_data]
    print(diagnose)
    print(diagnose.index)