import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from siphon.catalog import TDSCatalog
import xarray as xr
from clisops.core import subset
from xclim import ensembles as xens
from pathlib import Path
import sys


### DATA

# Locations
#locations = "../04_maps/maps_01_maps_plantations.tsv"
locations_file = sys.argv[1]

# url with the climate datasets (from https://pavics.ouranos.ca/datasets.html#a, for a different url select the dataset and click on "THREDDS catalog")
url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/simulations/bias_adjusted/cmip6/ouranos/ESPO-G/ESPO-G6-R2v1.0.0/catalog.xml"

# Scenario
#scenario_id = "ssp245"
# ssp370
# ssp585

scenario_id = sys.argv[2]

# Climate models to subset
model_ids = ["ACCESS-ESM1-5", "BCC-CSM2-MR", "CNRM-ESM2-1", "CanESM5", "EC-Earth3_",
             "GFDL-ESM4", "GISS-E2-1-G", "INM-CM5-0", "IPSL-CM6A-LR", "MIROC6",
             "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]

# Getting data for a decade
#year_start = 2021
year_start = int(sys.argv[3])

#year_stop = year_start+9
year_stop = 2100

# Subsetting models of interest for a given scenario: selecting 13, but only 12 are present in ouranus
def subset_models(catalog, scenario_id_, model_ids_):
    
    models_sub = []
    for mid in model_ids_:
        model_sub = [catalog.datasets[x] for x in catalog.datasets if (mid in x) and (scenario_id_ in x)]
        models_sub+=model_sub

    return(models_sub)

def get_future_ensembl_daily_for_location(latitude, longitude, models, year_start_, year_stop_):

    # Get all models for given coordinates
    M_ssp = []
    for model in models:    
        ds_url = model.access_urls["OpenDAP"]
        # open xarray.Dataset 
        ds = xr.open_dataset(ds_url, chunks=dict(time=256 * 2, lon=32, lat=32))
        # subsetting location
        ds_gridpoint = subset.subset_gridpoint(ds, lon=longitude, lat=latitude)
        # Subsetting time
        ds_sub = subset.subset_time(ds_gridpoint, start_date=str(year_start_), end_date=str(year_stop_))
        M_ssp.append(ds_sub)
    
    #print("Getting ensembl of models")
    ens_ssp = xens.create_ensemble(M_ssp)
    ens_stats = xens.ensemble_mean_std_max_min(ens_ssp)
    return(ens_stats)

def save_as_dataframe(dataset_array, population_id, scenario_id_, year_start_):
    
    #print("Getting dataframe")
    df = dataset_array[['tasmin_mean', 'tasmax_mean', 'pr_mean']].to_dataframe()
    
    #print("Formating and saving df")
    df['date'] = pd.to_datetime([str(ele) for ele in df.index.values])
    df['year'], df['month'], df['day'] = df['date'].dt.year, df['date'].dt.month, df['date'].dt.day
    df['pop'] = population_id
    df_ = df[['pop','lon','lat','date','year','month','day','tasmin_mean', 'tasmax_mean','pr_mean']].reset_index(drop=True)
    df_.to_csv("02_extract_daily_from_PAVICS_" + str(population_id) + "_"+scenario_id_+"_"+str(year_start_)+".csv", sep=",",header=True, index=False)


if __name__ == "__main__":

    # Reading table with locations
    df_locs = pd.read_csv(locations_file, sep="\t", header=0, dtype={'pop':str})
    tab_locs = df_locs.values.tolist()
    print(f"Number of locations: {len(tab_locs)}")

    # Create Catalog
    cat = TDSCatalog(url)
    # Access mechanisms - here we are interested in OpenDAP, a data streaming protocol
    cds = cat.datasets[0]
    print(f"Access URLs: {tuple(cds.access_urls.keys())}")
    
    # Subset all models from which to get ensembl values
    all_models = subset_models(cat, scenario_id, model_ids)
    
    # Loop by location
    for (pop, lat, lon) in tab_locs:
        print(pop)

        my_file = Path("./02_extract_daily_from_PAVICS_"+str(pop)+"_"+scenario_id+"_"+str(year_start)+".csv")
        if my_file.is_file():
            print("File exists")
        else:
                    
            # Get ensembl of daily future values
            dx_ensembl = get_future_ensembl_daily_for_location(lat, lon, all_models, year_start, year_stop)
    
            # Save as dataframe
            save_as_dataframe(dx_ensembl, pop, scenario_id, year_start)


