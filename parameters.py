## Use absolute path !!
cat_raw = '/home/septier/PhdFolder/GOC/waveform_cat_raw.csv' # the catalog to decluster
cat_formated = 'cat_formated.csv'# the name of the produced formatted catalog
folder_output = 'output' # /!\ do not change it /!\ (it will be created or erased and then created)
cpu = 8 # Number of cpu usable

## Declustering parameter computation arg
declustering_output = 'declustering_par.csv'

## SOM map parameters
model_name = 'som_model.p' # name of the output SOM model

n_neurons = 4 # height of the map
m_neurons = 4 # width of the map

iteration = 5000 # number of iteration
NSample = 2000 # number of sample used in the training

# definne the name of the features you want to use
Features = ['Rj', 'Tj',
             'Tj1', 'Tj2', 'Tj3', 'Tj4', 'Tj5', 'Tj6', 'Tj7', 'Tj8', 'Tj9',
             'Rj1', 'Rj2', 'Rj3', 'Rj4', 'Rj5', 'Rj6', 'Rj7', 'Rj8', 'Rj9','bval','intensity_norm']
