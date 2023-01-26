# Sapflow Processing with Dendra Extraction

## Pull the most recent raw data from Dendra and process sap flux. 

All of the extraction, cleaning, and processing can be done by typing the following into the terminal while in the base directory:

``sh analyze_sapflow.sh``

Details on available arguments are available as comments in that file. This has been tested with the naming conventions used at the ERCZO site.

## Notes about dendra.py:

* To skip the Dendra extraction (and authentication), specify -skip True in the bash script. Make sure to also supply the path and name of the raw data file if it is different than the default.

* A valid username and password is required. The username must be specified as an argument in the bash script and the password will be prompted for.

* dendra.py uses functions from the dendra_api_client by Collin Bode et al, which is downloaded and imported here. See here for the original repository, including a tutorial.

* There are 2 methods of extraction provided: export_data_long(), which takes about 3 hours (and saves data at the halfway point), and export_as_list(), which takes about 25 minutes. At present, export_data_long() is used by default because I am testing to make sure data is not missed with export_as_list() due to known issues with merging each data stream given NaN values in initial stream.

## Notes about sapflow.py:

* Calculating error is not yet implemented

* Cleaning takes ~35 seconds.

* Calculating sap flux from raw, cleaned data takes ~7 seconds.

* Raw (dT, deg C) and final (flux, cm/hr) timeseries for each tree are saved in the figs/ directory, which is created if it does not already exist.

## To Do Next:
* See how code runs for other measurements (not just sap temperature)
* Connect sapflux.py to Hydroshare csv to expedite processsing iteration

## Files:
```
├── analyze_sapflow.sh
├── code
│   ├── __pycache__
│   │   ├── dendra.cpython-39.pyc
│   │   ├── dendra_api_client.cpython-310.pyc
│   │   ├── dendra_api_client.cpython-39.pyc
│   │   └── sapflow.cpython-310.pyc
│   ├── dendra.py
│   ├── dendra_api_client.py
│   ├── drafts
│   │   ├── ... many draft .py files ...
├── data
├── figs (the following 3 directories are created automatically)
│   ├── trees (final sapflux figs for each tree)
│   ├── trees_raw (raw dT figs for each tree)
│   ├── trees_clean (cleaned dT figs for each tree)
├── environment.yml
├── README.md
├── .gitignore
```