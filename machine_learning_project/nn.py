import numpy as np
import time
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from utils import fluxnet_tools
import os
from utils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import subprocess
import argparse
import json

"""
THINGS I NEED TO UPDATE:
- starting # of nodes for nn == # of features
- make sure all of the data is scaled
- make sure all of the sites have the same # of features
- check out how I did the column parsing for x and y in RF
- play with how many layers and nodes to have
- why does activation function (eg ReLu) go in the forward method? 
- ^ I've seen some people put it in the layers?? Where should it be?
- set learning rate, batch_size, and epochs
- see what a representative seed is and set that
- I think I need an "accuracy" thing that gives wiggle room for what is establisehd as "correct =" in test_loop
"""
    
def main():
    args = import_args()

    site_df = pd.read_csv(os.path.join(args.data_dir, args.event_file))
    sites = site_df['SITE_ID'].unique()
    #sites = ['DE-Hai', 'CH-Dav', 'CA-Gro', 'CA-Oas', 'RU-Fyo', 'FR-LBr', 'US-Blo', 'US-SRC', 'CN-Cha']
    #sites = ['US-Blo']

    for site_id in sites:

        print(f"\n\nPROCESSING SITE: {site_id}")

        # Load in data for a site and process
        #site_id = args.site #'DE-Hai'
        target = args.target #'gpp_dt_dayavg'

        # Make output directory for the specific site and save args
        output_dir = os.path.join(args.output_dir)#, site_id)
        if os.path.exists(output_dir) == False:
                subprocess.call('mkdir -p ' + output_dir, shell=True)
                #print(f'\nSaving neural net results to {output_dir}')
        with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)


        train_data, test_data, train_metadata, test_metadata, scaler = load_and_prep_data(
                                                                                dir = args.data_dir,
                                                                                file_name = args.train_file,
                                                                                site_id = site_id,
                                                                                target = target,
                                                                                split_train = args.split_train,
                                                                                test_size = args.test_size)

        event_data, _, event_metadata, _, _ = load_and_prep_data(
                                                            dir = args.data_dir,
                                                            file_name = args.event_file,
                                                            site_id = site_id,
                                                            target = target,
                                                            split_train = False)

        #print("\n\n")
        #print(train_data.head())
        #print(test_data.head())

        train_metadata.to_csv(os.path.join(output_dir, 'train_metadata_' + site_id + '.csv'))
        test_metadata.to_csv(os.path.join(output_dir, 'test_metadata_' + site_id + '.csv'))
        event_metadata.to_csv(os.path.join(output_dir, 'event_metadata_' + site_id + '.csv'))

        # Begin nn analysis
        torch.manual_seed(5)
        np.random.seed(5)
        
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using {device} device")
        #If untraceable error messages, just use cpu:
        #device = "cpu"
        
        print("Creating dataset object")
        train_dataset = FluxDataset(dataframe = train_data, target_col_name = target, device = device)
        test_dataset = FluxDataset(dataframe = test_data, target_col_name = target, device = device)
        
        batch_size = args.batch_size

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True)

        # Initialize net
        net = Net().to(device)
        
        # Set up hyperparameters
        epochs = args.epochs
        learning_rate = args.learning_rate
        loss_fn = nn.MSELoss()
        log_interval = args.log_interval
        optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    
        # Train and test model
        for t in range(epochs):
            print(f"Epoch {t+1}\n---------------------------------------------------")
            train_loop(
                dataloader = train_dataloader,
                model = net,
                loss_fn = loss_fn,
                optimizer = optimizer,
                log_interval = log_interval)
            test_loop(test_dataloader, net, loss_fn)
        print("Done training!")
        
        print("\nAssessing performance...")
        results = predict_and_save(test_data, target, net, device)
        test_inverse_transform = pd.DataFrame(scaler.inverse_transform(test_data.loc[:, test_data.columns != target]), columns = scaler.feature_names_in_)
        results = pd.concat([results, test_inverse_transform], axis = 1)
        results = deseason_target(results, 'test', output_dir, target, site_id)
        results.to_csv(os.path.join(output_dir, 'results_testing_' + site_id + '.csv'))

        print("\nPredicting on event days...")
        event_results = predict_and_save(event_data, target, net, device)
        event_inverse_transform = pd.DataFrame(scaler.inverse_transform(event_data.loc[:, event_data.columns != target]), columns = scaler.feature_names_in_)
        event_results = pd.concat([event_results, event_inverse_transform], axis = 1)
        event_results = deseason_target(event_results, 'event', output_dir, target, site_id)
        event_results.to_csv(os.path.join(output_dir, 'results_event_' + site_id + '.csv'))

        # Save the model
        model_file_path = os.path.join(output_dir, "nn_" + site_id + ".pth")
        torch.save(net.state_dict(), model_file_path)


def import_args():
    parser = argparse.ArgumentParser('Go from hh_processing.py to a ready-to-go training and testing csv for neural net.')
    parser.add_argument('-output_dir', type=str, default='runs/NN/run1', help='directory with training and event data')
    parser.add_argument('-data_dir', type=str, default='runs/output_v3', help='directory with training and event data')
    parser.add_argument('-site', type=str, default='DE-Hai', help='directory with training and event data')
    parser.add_argument('-train_file', type=str, default='allsites_training.csv', help='file with training data')
    parser.add_argument('-event_file', type=str, default='allsites_event.csv', help='file with event data')
    parser.add_argument('-target', type=str, default='gpp_dt_dayavg', help='column in training and event data to use as y (target)')
    parser.add_argument('-use_swc_as_feature', type=bool, default=False, help='binary; whether to use soil water content as feature')
    parser.add_argument('-use_doy_as_feature', type=bool, default=False, help='binary; whether to use soil water content as feature')
    parser.add_argument('-split_train', type=bool, default=True, help='binary whether to split training data and predict GPP over testing data that is not during events')
    parser.add_argument('-test_size', type=float, default=0.25, help='The fraction of training data to be witheld for testing if split_train = True')
    parser.add_argument('-batch_size', type=int, default=70, help='')
    parser.add_argument('-epochs', type=int, default=20, help='')
    parser.add_argument('-learning_rate', type=float, default=0.005, help='')
    parser.add_argument('-log_interval', type=int, default=1, help='')
    args = parser.parse_args()
    return args


# Dataset class --------------------------------------------------------------------------------------------------

class FluxDataset(torch.utils.data.Dataset):
    # define init method
    def __init__(self, dataframe, target_col_name, device):

        # Separate into x and y
        data = dataframe.copy()
        temp_y = np.array(data[target_col_name]).reshape(-1,1)
        del data[target_col_name]
        temp_x = np.array(data)

        print(temp_x.shape, temp_y.shape)
        
        # Convert x and y to tensors and store as attributes
        self.x_data = torch.tensor(temp_x, dtype = torch.float32).to(device)
        self.y_data = torch.tensor(temp_y, dtype = torch.float32).to(device)
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        predictors = self.x_data[idx,:]
        label = self.y_data[idx,:]
        return (predictors, label) # tuple of 2 matrices
    
# Set up neural net ------------------------------------------------------------------------------------------------

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define layers here
        self.hidden1 = nn.Linear(11, 40)
        self.hidden2 = nn.Linear(40, 40)
        self.output = nn.Linear(40,1)
        
        # Initialize weights and biases (_ after method = "in place")
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.zeros_(self.hidden2.bias)
        #nn.init.xavier_uniform_(self.hidden3.weight)
        #nn.init.zeros_(self.hidden3.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)   

    def forward(self, x):
        z = torch.relu(self.hidden1(x))
        z = torch.relu(self.hidden2(z))
        #z = torch.relu(self.hidden3(z))
        z = self.output(z) # no activation on the output bc we want the real value
        return z
    
    
def train_loop(dataloader, model, loss_fn, optimizer, log_interval):
    size = len(dataloader.dataset)

    for (batch, (X,y)) in enumerate(dataloader): 
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backprop
        optimizer.zero_grad() # need to reset each loop
        loss.backward() # do backprop
        optimizer.step() # update parameters based on optimization algorithm

        if batch % log_interval == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} (batch: {batch:>2d}, len(x):{len(X):>3d}) [{current:>5d}/{size:>5d}]")
          
            
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0 # re-initialize each time
    
    with torch.no_grad(): # tell it not to track gradients for backprop
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f}\n") #Accuracy: {100*correct:>0.1f}, 
    

def predict_and_save(dataframe, target_col_name, model, device):
    # Separate into x and y
    actual = dataframe[target_col_name].values
    nn_output = []
    del dataframe[target_col_name]

    for i in range(dataframe.shape[0]):
        # Get a single row of features
        features = np.array(dataframe.loc[i])

        # Convert to tensor
        features = torch.tensor(features, dtype = torch.float32).to(device)
    
        # Predict values
        with torch.no_grad():
            predicted = model(features)
            predicted_value = predicted.item()

            nn_output.append(predicted_value)
    
    results = pd.DataFrame({"actual": actual, "predicted":nn_output})
    return results



            
        

# Prepare Data ---------------------------------------------------------------------------------------------------         

def load_and_prep_data(
    dir, file_name, site_id, target, split_train = True, test_size = 0.25, 
    use_swc_as_feature = False, use_doy_as_feature = False):
    
    print(f"Prepping {dir}/{file_name} with target {target}.\
        \n\t\tuse_swc={use_swc_as_feature}\n\t\tuse_doy={use_doy_as_feature} \
        \n\t\tsplit_train={split_train}\n\t\ttest_size={test_size}")

    # Import csvs to start processingå
    data = pd.read_csv(os.path.join(dir, file_name), parse_dates = ['day'])
    data['doy'] = data['day'].dt.dayofyear

    # Narrow down just to site
    site_data = data[data['SITE_ID'] == site_id]

    # Choose which column to use as target (ie GPP)
    gpp = target
    if gpp not in site_data.columns:
        raise ValueError(f"Predictor column {gpp} not found in features dataframe.")

    # Add lagged precipitation columns and april GPP
    plags = [2, 7, 30, 90, 120]
    site_data = add_lag_columns_and_april_gpp(site_data, plags, site_id)

    # Choose only feature and target columns
    feature_list = [
        #'Unnamed: 0',
        #'day',
        #'start',
        #'end',
        #'swc_dayavg',
        #'gpp_dt_dayavg',
        #'gpp_nt_dayavg',
        'sw_in_dayavg',
        'ta_dayavg', 
        'le_dayavg', 
        'vpd_dayavg',
        'rh_dayavg', 
        #'doy', 
        #'swc_dayavg_mean_doy', 
        #'gpp_dt_dayavg_mean_doy',
        #'gpp_nt_dayavg_mean_doy', 
        #'sw_in_dayavg_mean_doy', 
        #'ta_dayavg_mean_doy',
        #'le_dayavg_mean_doy', 
        #'vpd_dayavg_mean_doy', 
        #'rh_dayavg_mean_doy',
        #'SITE_ID', 
        #'sw_dif_dayavg', 
        #'ppfd_in_dayavg'
        ]
    if use_swc_as_feature == True:
        feature_list.append('swc_dayavg')
    if use_doy_as_feature == True:
        feature_list.append('doy')
    pcols = [col for col in site_data if col.startswith('P_F')]
    feature_list.extend(pcols)
    print(f"Using {len(feature_list)} features ({feature_list})")

    # Only include columns for features and target
    metadata_cols = [target + '_mean_doy', 'day', 'SITE_ID', 'doy']
    keep_cols = feature_list + [target] + metadata_cols
    site_data = site_data[keep_cols]

    # Split data into training and testing
    if split_train == True:
        train_data, test_data = train_test_split(site_data, test_size = test_size, shuffle = True)

    else:
        train_data = site_data

    # Separate target from features
    train_target = train_data[target].values
    # Separate seasonally adjusted and date from training and testing
    train_metadata = train_data[metadata_cols]
    train_features = train_data[feature_list]
    # Fit MinMax Scaler on training data
    scaler = MinMaxScaler()
    scaler.fit(train_features)
    # Scale training and test features
    train_scaled = pd.DataFrame(scaler.transform(train_features), columns = scaler.feature_names_in_)
    # Add back on target column (unscaled)
    train_scaled[target] = train_target

    if split_train == True:
        # Separate target from features
        test_target = test_data[target].values
        # Separate seasonally adjusted and date from training and testing
        test_metadata = test_data[metadata_cols]
        test_features = test_data[feature_list]
        test_scaled = pd.DataFrame(scaler.transform(test_features), columns = scaler.feature_names_in_)
        test_scaled[target] = test_target
    else:
        test_scaled = pd.DataFrame()
        test_metadata = pd.DataFrame()

    return train_scaled, test_scaled, train_metadata, test_metadata, scaler


def april_gpp(df_dd, features, doy_start = 92, doy_end = 100, p_thresh = 1):
    df = fluxnet_tools.add_time_columns_to_df(df_dd, timestamp_col = 'TIMESTAMP')
    df = df[df['P_F'] < p_thresh]
    df = df[df['DOY'] > doy_start]
    df = df[df['DOY'] < doy_end]
    print(f"April GPP is using DOY {df['DOY'][0]} to {df['DOY'][-1]}")
    gpps = df.groupby('Year')['GPP_DT_VUT_REF'].mean() 
    gpps = gpps.to_frame().rename(columns={'GPP_DT_VUT_REF':'GPP_April'}).reset_index()
    features['Year'] = features['day'].dt.year
    features = features.merge(gpps, how = 'left', on = 'Year')
    del features['Year']
    return features

def calculate_lagged_df(df, feat_col_name, num_days = 7, agg = 'mean'):
    if isinstance(num_days, int) == False:
        raise TypeError(f"num_days must be int. Received {type(num_days)}.")
    if feat_col_name not in df:
        raise ValueError(f"feat_col_name '{feat_col_name}' not in df.")
    new_col = feat_col_name + '_lag' + str(num_days)  
    offset = str(num_days) + 'd'
    if agg == 'mean':
        new_col = new_col + 'mean'
        df[new_col] = df[feat_col_name].rolling(offset).mean() 
    elif agg == 'sum':
        new_col = new_col + 'sum'
        df[new_col] = df[feat_col_name].rolling(offset).sum() 
    else:
        raise ValueError(f"agg must be mean or sum. Received {agg}.")
    return df
    
def add_lag_columns_and_april_gpp(event_or_training_df, plags, site_id):
    df_dd_plus_lag = pd.DataFrame()
    day_file = paths.FLUXNET_DD_DIR + '/' + site_id + '.csv'
    df_dd = pd.read_csv(day_file, parse_dates = ['TIMESTAMP'], usecols = ['TIMESTAMP', 'P_F', 'GPP_DT_VUT_REF'])#, index_col = 'TIMESTAMP') 
    df_dd.index = df_dd['TIMESTAMP']
    df_dd['day'] = df_dd['TIMESTAMP'].dt.floor('d')
    for lag in plags:
        df_dd = calculate_lagged_df(df_dd, feat_col_name = 'P_F', num_days = lag, agg = 'sum')
    df_dd_plus_lag = pd.concat([df_dd_plus_lag, df_dd])   
    event_or_training_df = event_or_training_df.merge(df_dd_plus_lag, how = 'left', on = 'day')
    event_or_training_df = april_gpp(df_dd, event_or_training_df)
    return event_or_training_df


def deseason_target(predicted_df, test_or_event, output_dir, target_col, site_id):
    filename = test_or_event + '_metadata_' + site_id + '.csv'
    growseason = pd.read_csv(os.path.join(output_dir, filename))
    col_name = target_col + '_mean_doy'
    predicted_df = pd.concat([predicted_df, growseason], axis = 1)
    predicted_df['actual_seasonal'] = predicted_df['actual'] + predicted_df[col_name]
    predicted_df['predicted_seasonal'] = predicted_df['predicted'] + predicted_df[col_name]
    return predicted_df

if __name__ == '__main__':
    main()