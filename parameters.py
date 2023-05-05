## Use absolute path !!
#cat_raw = '/home/septier/PhdFolder/Cat_mixer/GOC_etas_mixer_2km_spread.csv' # the catalog to decluster
cat_raw = '/home/septier/PhdFolder/Tanzani/cat_filtered.csv' # the catalog to decluster
#cat_raw = '/home/septier/PhdFolder/etas/cat_synth_sw.csv'
cat_formated = 'cat_formated.csv'# the name of the produced formatted catalog
waveform = '/home/septier/PhdFolder/Taiwan/waveform' # path to the waveform folder (Must be Year/Event origin time/staName.mseed)
folder_output = 'output' # /!\ do not change it /!\ input the path of the wanted output folder (it will be created or erased and then created)

## Signal processing argument
startshift = 75 # Number of point to shift the start of the signal
endcut = 150 # Number of point to take for the signal
# TODO : add frequency parameter

## CTD Arg
N_Ctest = 1 # Number of random sample to make the correlation

## Declustering parameter computation arg
step_save = 1 #2000 # save each step_save the result (if to small may greatly slow down the computation)
declustering_output = 'declustering_par.csv'

## Fortran parameters
taille_cat = 33119 # Taille exact du catalog must be set manualy in the f90, file for now and recompile them
#cat_formated and output folder must also be manualy writed

## SOM map parameters
model_name = 'som_model.p' # name of the output SOM model

n_neurons = 150 # height of the map
m_neurons = 150 # width of the map

iteration = 5000 #95550 #38000 # number of iteration
NSample = 2000 #136500 #28000 #35000  # number of sample used in the training
"""
Features = ['Rj', 'Tj', 'maxmean','Cj1', 'Cj2', 'Cj3', 'Cj4', 'Cj5', 'Cj6', 'Cj7', 'Cj8', 'Cj9',
             'Tj1', 'Tj2', 'Tj3', 'Tj4', 'Tj5', 'Tj6', 'Tj7', 'Tj8', 'Tj9',
             'Rj1', 'Rj2', 'Rj3', 'Rj4', 'Rj5', 'Rj6', 'Rj7', 'Rj8', 'Rj9','intensity','intensity_norm','Nnear'] # the 30 first must stay the same for the p_back graph
"""
"""
Features = ['Rj', 'Tj',
             'Tj1', 'Tj2', 'Tj3', 'Tj4', 'Tj5', 'Tj6', 'Tj7', 'Tj8', 'Tj9',
             'Rj1', 'Rj2', 'Rj3', 'Rj4', 'Rj5', 'Rj6', 'Rj7', 'Rj8', 'Rj9','intensity','intensity_norm','Nnear','bval','r2_Tj']
"""
Features = ['Rj', 'Tj',
             'Tj1', 'Tj2', 'Tj3', 'Tj4', 'Tj5', 'Tj6', 'Tj7', 'Tj8', 'Tj9',
             'Rj1', 'Rj2', 'Rj3', 'Rj4', 'Rj5', 'Rj6', 'Rj7', 'Rj8', 'Rj9','bval','intensity_norm','Nnear','r2_Tj']