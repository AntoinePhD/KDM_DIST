from minisom import MiniSom
import pickle
import pandas as pd
import numpy as np
import parameters
from tqdm import tqdm
import warnings

# just to make the terminal clearer can be commented if needed
warnings.filterwarnings("ignore")

## PARAMETERS
model_name = parameters.folder_output+'/'+parameters.model_name

n_neurons = parameters.n_neurons
m_neurons = parameters.m_neurons
catalogue_par_file = parameters.folder_output+'/'+parameters.declustering_output

iteration = parameters.iteration
NSample = parameters.NSample

## LOAD DATA
data = pd.read_csv(catalogue_par_file)
data = data[parameters.Features]#, 'kurto']]

data = data[:NSample]
# data normalization
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

# Dropping nan lines
nanid = np.isnan(data).any(axis=1)
data2 = data[~np.isnan(data).any(axis=1), :]
print('data usable / catalogue entry ', len(data2), '/', len(data))
data = data2

## RUNNING SOM ALGO

som = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5,
              neighborhood_function='gaussian', random_seed=0)
"""
som = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5, 
              neighborhood_function='triangle', random_seed=0)
"""

som.pca_weights_init(data)
tqdm(range(NSample))
som.train(data, iteration, verbose=False)  # random training

with open(model_name, 'wb') as outfile:
    pickle.dump(som, outfile)
