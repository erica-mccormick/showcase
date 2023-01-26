

import glob
import os
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

from utils import paths
from utils import event_identification_tools
from utils import fluxnet_tools
import time
import pickle
import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint
import sklearn
from sklearn import preprocessing


# python3 randomforest_current.py  -output_dir 'runs/RF/gpp_5'


def main():

    # Import and save args to txt file
    args = import_args()
    print(args.output_dir)
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    dir = args.input_dir #'runs/output_test'
    dir_to_save_results = args.output_dir #'runs/RF/gpp_3'
    print(f"Saving output to {dir_to_save_results}...\n")

    # Open features
    features = pd.read_csv(os.path.join(dir,'allsites_training.csv'), parse_dates = ['day'])
    features['doy'] = features['day'].dt.dayofyear

    # Open event file
    events = pd.read_csv(os.path.join(dir,'allsites_event.csv'), parse_dates = ['day'])
    events['doy'] = events['day'].dt.dayofyear

    # Choose which column to use as target (ie GPP)
    gpp = args.target # 'gpp_dt_dayavg
    if gpp not in features.columns:
        raise ValueError(f"Predictor column {gpp} not found in features dataframe.")

    # Choose whether to include SWC as feature
    remove_feats = ['gpp_dt_dayavg', 'gpp_nt_dayavg', 'GPP_DT_VUT_REF', 'Unnamed: 0']
    if str.lower(args.use_swc_as_feature) == 'false': remove_feats.append('swc_dayavg')
   
    # CHOOSE SITES: CAN CHANGE THIS IN THE CODE
    sites = features['SITE_ID'].unique()
  
    # PRECIPITATION LAGS: CAN CHANGE THIS IN THE CODE
    plags = [2, 7, 30, 90, 120]

    # HYPER-PARAMETERS TO SEARCH OVER: CAN CHANGE THIS IN THE CODE
    estimator = RandomForestRegressor()
    params = {
        'max_depth': randint(args.max_depth_min, args.max_depth_max),
        'min_samples_split': randint(args.min_samples_split_min, args.min_samples_split_max),
        'max_features': uniform(args.max_features_loc, args.max_features_scale),
        'n_estimators':[args.n_estimators]
    }

    save_df = pd.DataFrame()
    if str.lower(args.split_train) == 'true': save_df_test = pd.DataFrame()

    for site in sites:

        t0 = time.time()
        feat_site = features[features['SITE_ID'] == site]
        event_site = events[events['SITE_ID'] == site]

        if event_site.shape[0] > 0:
            print(site)

            ### ADD LAGGED PRECIPITATION
            df_dd_plus_lag = pd.DataFrame()
            day_file = paths.FLUXNET_DD_DIR + '/' + site + '.csv'
            df_dd = pd.read_csv(day_file, parse_dates = ['TIMESTAMP'], usecols = ['TIMESTAMP', 'P_F', 'GPP_DT_VUT_REF'])#, index_col = 'TIMESTAMP') 
            df_dd.index = df_dd['TIMESTAMP']
            df_dd['day'] = df_dd['TIMESTAMP'].dt.floor('d')
            for lag in plags:
                df_dd = add_avg_lagged_feature(df_dd, feat_col_name = 'P_F', num_days = lag, agg = 'sum')
            df_dd_plus_lag = pd.concat([df_dd_plus_lag, df_dd])

            feat_site = april_gpp(df_dd, feat_site)
            event_site = april_gpp(df_dd, event_site)


            # Merge lagged precip columns with features df
            feat_site = feat_site.merge(df_dd_plus_lag, how = 'left', on = 'day')
            event_site = event_site.merge(df_dd_plus_lag, how = 'left', on = 'day')


            pcols = [col for col in feat_site if col.startswith('P_F')]
            avgcols = [col for col in feat_site if col.endswith('_dayavg')]
            feature_list = ['doy', 'GPP_April'] + pcols + avgcols
            for i in remove_feats:
                if i in feature_list:
                    feature_list.remove(i)


            X_dataframe = feat_site[feature_list].dropna(axis = 1)
            X = np.array(X_dataframe)
            y = np.array(feat_site[gpp])

            # Scale X
            X_scaled = preprocessing.StandardScaler().fit_transform(X, y)

            if str.lower(args.split_train) == 'true':
                X, X_test, y, y_test = train_test_split(X_scaled, y, test_size = args.test_size)
                y_test_doy = X_test[:,0]

            X_event = np.array(event_site[X_dataframe.columns].dropna(axis = 1))
            y_event = np.array(event_site[gpp])
            y_doy = np.array(event_site['doy'])

            print(f"\tX shape: {X.shape}, Y shape: {y.shape}")
            print(f"\tX_event shape: {X_event.shape}, Y_event shape: {y_event.shape}")

            # Inner and outer loop cross-validation
            inner_cv = KFold(n_splits = 6, shuffle = True)
            outer_cv = KFold(n_splits = 6, shuffle = True)

            # Parameter optimization
            regr = RandomizedSearchCV(estimator, params, cv = inner_cv, n_jobs = -1, n_iter = args.niter)
            score_cv = cross_val_score(estimator, X, y, cv = outer_cv)

            regr.fit(X, y)
            gpp_predicted = regr.predict(X_event)
            r2 = regr.score(X_event, y_event)


            print(f"\tScore mean: {score_cv.mean()}")

            temp_df = pd.DataFrame({
                "gpp_predicted": gpp_predicted,
                "gpp_actual": y_event,
                "SITE_ID": site,
                "score": r2,
                "training_score_mean": score_cv.mean(),
                "n_training": X.shape[0],
                "n_event": X_event.shape[0],
                "doy": y_doy
                })

            save_df = pd.concat([save_df, temp_df])

            results = pd.DataFrame(regr.cv_results_)
            results.to_csv(os.path.join(dir_to_save_results, site + '_regrsearch.csv'))

            # Don't need to save all these intermediate files because things are running smoothly now
            #temp_df.to_csv(os.path.join(dir_to_save_results, site + '_gpp_temp.csv'))

            if str.lower(args.split_train) == 'true':
                gpp_predicted_test = regr.predict(X_test)
                r2_test = regr.score(X_test, y_test)

                temp_save_df_test = pd.DataFrame({
                    "gpp_predicted": gpp_predicted_test,
                    "gpp_actual": y_test,
                    "SITE_ID": site,
                    "score": r2_test,
                    "training_score_mean": score_cv.mean(),
                    "training_score_std": score_cv.std(),
                    "n_training": X.shape[0],
                    "n_event": X_event.shape[0],
                    "n_testing": X_test.shape[0],
                    "doy": y_test_doy
                })
                save_df_test = pd.concat([save_df_test, temp_save_df_test])

            t1 = time.time()
            print(f"\tTime: {round(t1-t0)} secs")
        
        # Save the predictions during event days
        save_df.to_csv(os.path.join(dir_to_save_results, 'ALL_GPP_PREDICTIONS_EVENTS.csv'))

        # Save the predictions on training days witheld for testing, if -split_train = True
        save_df_test.to_csv(os.path.join(dir_to_save_results, 'ALL_GPP_PREDICTIONS_TESTING.csv'))


def import_args():
    parser = argparse.ArgumentParser('Predict GPP on event days using random forest regressor on training days.')
    parser.add_argument('-input_dir', type=str, default='runs/output_v1', help='directory with training and event data')
    parser.add_argument('-output_dir', type=str, default='', help='directory to save output files')
    parser.add_argument('-target', type=str, default='gpp_dt_dayavg', help='column in training and event data to use as y (target)')
    parser.add_argument('-use_swc_as_feature', type=str, default='False', help='binary; whether to use soil water content as feature')
    parser.add_argument('-niter', type=int, default=60, help='hyperparameter tuning: n_iter; how many times to sample from param_grid distribution')
    parser.add_argument('-max_depth_min', type=int, default=10, help='hyperparameter tuning: the minimum for max_depth (searches randint of min and max)')
    parser.add_argument('-max_depth_max', type=int, default=110, help='hyperparameter tuning: the maximum for max_depth (searches randint of min and max)')
    parser.add_argument('-n_estimators', type=int, default=1000, help='hyperparameter tuning: n_estimators. Takes one value only.')
    parser.add_argument('-min_samples_split_min', type=int, default=2, help = 'hyperparameter tuning: the minimum of min_samples_split (searches randint(min, max))')
    parser.add_argument('-min_samples_split_max', type=int, default=30, help = 'hyperparameter tuning: the maximum of min_samples_split (searches randint(min, max))')
    parser.add_argument('-max_features_loc', type=float, default=0, help='hyperparameter tuning: the loc param of uniform(loc, scale) for max_features (dist is [loc, loc+scale])')
    parser.add_argument('-max_features_scale', type=float, default=1, help='hyperparameter tuning: the scale param of uniform(loc, scale) for max_features (dist is [loc, loc+scale])')
    parser.add_argument('-split_train', type=str, default='False', help='binary whether to split training data and predict GPP over testing data that is not during events')
    parser.add_argument('-test_size', type=float, default=0.25, help='The fraction of training data to be witheld for testing if split_train = True')
    args = parser.parse_args()

    if os.path.exists(args.output_dir) == False:
        os.makedirs(args.output_dir)
    else:
        print(f"Warning: output_dir {args.output_dir} already exists and will be overwritten!")
    return args


def save_log(args, ):
        # Save args to txt file too
    with open(os.path.join(args.dir_output, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    
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

def add_avg_lagged_feature(df, feat_col_name, num_days = 7, agg = 'mean'):
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



if __name__ == '__main__':
    main()