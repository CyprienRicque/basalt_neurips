# DO NOT MODIFY THIS FILE
# This is the entry point for your submission.
# Changing this file will probably fail your submissions.

import train
import run_BuildVillageHouse
import run_CreateVillageAnimalPen
import run_FindCave
import run_MakeWaterfall

import os

# By default, only do testing
EVALUATION_STAGE = os.getenv('EVALUATION_STAGE', 'testing')

# Training Phase
if EVALUATION_STAGE in ['all', 'training']:
    train.main()

# Testing Phase
if EVALUATION_STAGE in ['all', 'testing']:
    test_BuildVillageHouse.main()
    test_CreateVillageAnimalPen.main()
    test_FindCave.main()
    test_MakeWaterfall.main()
