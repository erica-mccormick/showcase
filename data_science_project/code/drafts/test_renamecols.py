
# testing tracer decorator
from turtle import tracer
import pandas as pd
df = pd.read_csv('emdata/SapTemperatureDifference.csv')
#print(df.columns.unique())

def parse_columns(df, order = 'level_sensor_species_tree'):
    for col in df:
        try:
            split = col.split('_')
            level = split[0]#[-2:]
            sensor = split[-3]
            tree = split[-4]
            species = split[-5]    
            
            name_to_var_dict = {
                'level': level,
                'sensor': sensor,
                'tree': tree,
                'species': species,
            }
            
            col_name = [str(name_to_var_dict[i]) + '_' for i in order.split('_')]
            print(col_name)
        except:
            pass
        
#parse_columns(df)
        
test = ['1', '2', '3']
        

##### PLAYING WITH DECORATORS
import functools
import time


timer_on = False # Print out how long the function took 
tracer_on = False # Print out the name of functions as they run
tracer_verbose = False # Print out the name, args, kwargs, and returns of the functions as they run


def tracer(func, tracer_on):
    if tracer_on:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f'Running {func.__name__}()...')
            #if tracer_verbose:
            #    print(f'tArgs: {args}, Kwargs: {kwargs}')
            #    output = func(*args, **kwargs)
            #    print(f'Returns: {output}\n')
        return wrapper
    else:
        return func

def timer(func, timer_on):
    """
    Decorator to time functions. See RealPython tutorial for more info.
    """
    if timer_on:
        @functools.wraps(func) # Ensures metadata is carried through
        def wrapper_timer(*args, **kwargs):
            tic = time.perf_counter()
            value = func(*args, **kwargs)
            toc = time.perf_counter()
            elapsed_time = toc - tic
            if elapsed_time > 60:
                print(f"Elapsed time: {elapsed_time/60:0.4f} minutes")
            else: print(f"Elapsed time: {elapsed_time:0.4f} seconds")
            return value
        return wrapper_timer
    else:
        return func

#@timer
@tracer(tracer_on)
def say_things(a, b):
    """
    Say things in a row.
    """
    print(f'{a}, {b}')


say_things('hi', 10)

