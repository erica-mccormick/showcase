import os

# General file dirs
USER = 'ericamcc' # Stanford Desktop
#USER = 'ericamccormick' # Laptop

GDRIVE_PATH = '/Users/' + USER + '/Library/CloudStorage/GoogleDrive-ericamcc@stanford.edu/My Drive/'

METADATA_PATH = os.path.join(GDRIVE_PATH, 'DATASETS/FLUXNET15/FLX_AA-Flx_BIF_ALL_20200501/FLX_AA-Flx_BIF_DD_20200501.csv')
FLUXNET_RAW_DIR = os.path.join(GDRIVE_PATH, 'DATASETS/FLUXNET15')

OUT_CSVS_DIR = os.path.join(GDRIVE_PATH, 'PROJECTS/22_extreme_wet_events/output')
FLUXNET_HH_DIR = os.path.join(GDRIVE_PATH, 'DATASETS/FLUXNET_HH_SELECTED')
FLUXNET_DD_DIR = os.path.join(GDRIVE_PATH, 'DATASETS/FLUXNET_DD_SELECTED')

# Names for specific output files 
PATH_TO_SWC_SITES = os.path.join(OUT_CSVS_DIR, 'selected_sites_SWC.txt')
EVENTS_CSV_DIR = os.path.join(OUT_CSVS_DIR, 'events_lists')
PATH_TO_EVENTS_WITH_TIMESERIES_RAW = os.path.join(OUT_CSVS_DIR, 'events_with_timeseries_raw.csv')
BAD_SWC_TIMES = os.path.join(OUT_CSVS_DIR, 'fluxnet_bad_swc.csv')


# Names for specific figure directories
FIG_DIR = os.path.join(GDRIVE_PATH, 'PROJECTS/22_extreme_wet_events/figs')
FIG_DIR_FOR_EVENT_TIMESERIES = os.path.join(FIG_DIR, 'figs_timeseries_by_thresh')
