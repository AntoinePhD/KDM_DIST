import subprocess
import os
import parameters
import pandas as pd
import glob
import numpy as np
from time import sleep
import warnings

# just to make the terminal clearer can be commented if needed
warnings.filterwarnings("ignore")

print('---------  Creating the output folders ---------')
## Create Output folder
# if the dir exit than write over
if os.path.isdir(parameters.folder_output):
    yn = input('the output folder exist and will be erased are you sure ? [y,n] ')
    if yn == 'y':
        os.popen('rm -d -r -f '+parameters.folder_output)
        sleep(0.2)
    elif yn=='m' :
        1
    else:
        os._exit(0)

os.mkdir(parameters.folder_output)
os.mkdir(parameters.folder_output+'/graph')

## Run prep_cat
print('---------  Reading input file and formating it for the next routines ---------')
subprocess.call(['python',"prep_cat.py"])

## Find T,D
# create fortran param file
f = open(parameters.folder_output+"/fvar.f", "w")
f.write("INTEGER, PARAMETER :: taille_cat ="+str(len(pd.read_csv(parameters.folder_output + '/' + 'cat_formated.csv'))))
f.close()

# Compile the fortran file
os.popen('gfortran time_distance_test.f90 -o time_distance_test')

print('---------  Compute the quartile of time and space distance --------- ')
# Find T,D
proc = os.popen("./time_distance_test").read()
TD = proc.split()
T = round(float(TD[3]),4)
D = round(float(TD[7]),4)

# save the CTD factor for python
f = open(parameters.folder_output+"/CTD.py", "w")
f.write("T="+str(T)+'\n'+"D="+str(D))
f.close()

# save the CTD factor for fortran
f = open(parameters.folder_output+"/fvar.f", "a")
f.write('\n'+"REAL, PARAMETER :: Space_norme =" + str(round(D,4)))
f.write('\n'+"REAL, PARAMETER :: Time_norme =" + str(round(T,4) * 24 * 60 * 60))
f.close()

## Compute Declustering parameters
# compile the fortran routine
os.popen('gfortran tennearest.f90 -o tennearest')

# start the calculation of neighbours distance
print('---------  Compute declustering features --------- ')
subprocess.call(['python',"declustering_npar.py"])

# calculate magnitude intensity
subprocess.call(['python',"intensity.py"])

## Train SOM
print(' ---------  Trains the models --------- ')
subprocess.call(['python',"Create_SOM.py"])

## Do classification
print('--------- Apply the models --------- ')
subprocess.call(['python',"classification++.py"])
