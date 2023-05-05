import subprocess
import os
import parameters
import pandas as pd
import glob
import numpy as np
from time import sleep

## Compile the fortran file
os.popen('gfortran time_distance_test.f90 -o time_distance_test')
#os.popen('gfortran time_distance_test.f90 -o time_distance_test')

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
subprocess.call(['python',"prep_cat.py"])

## Find T,D

# Find T,D
proc = os.popen("./time_distance_test").read()
TD = proc.split()
T = round(float(TD[3]),4)
D = round(float(TD[7]),4)

# save the CTD factor
f = open(parameters.folder_output+"/CTD.py", "w")
f.write("T="+str(T)+'\n'+"D="+str(D))
f.close()

#give the user the time to change the value in f90
input('press enter after changing the space and time norme in tennearest.f90')

## Compute Declustering parameters
# compile the fortran routine
os.popen('gfortran tennearest.f90 -o tennearest')

# start the calculation of neighbours distance
subprocess.call(['python',"declustering_npar.py"])

# calculate magnitude intensity
subprocess.call(['python',"intensity.py"])

## Train SOM
subprocess.call(['python',"Create_SOM.py"])

## Do classification
subprocess.call(['python',"classification++.py"])
