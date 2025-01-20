#!/bin/bash

# activate conda environment

POPS=locations.txt
SCENARIO=ssp245
YEAR=2081

python download_climate_for_list_of_locations.py $POPS $SCENARIO $YEAR
