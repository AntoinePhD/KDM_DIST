import parameters
import pandas as pd
import numpy as np
import pickle
import collections
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering

## LOAD CATALOGUE
catalogue_par_file = parameters.folder_output + '/' + parameters.declustering_output

Features = parameters.Features

data = pd.read_csv(catalogue_par_file)
data = data[Features]
data = data.replace(99999, np.nan)
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

# Dropping nan lines
nanid = np.isnan(data).any(axis=1)
data2 = data[~np.isnan(data).any(axis=1), :]
print('data usable / catalogue entry ->', len(data2), '/', len(data))
data = data2

## LOAD SOM ALGO
model_name = parameters.folder_output + '/' + parameters.model_name

with open(model_name, 'rb') as infile:
    som = pickle.load(infile)
print(som)

## APPLY SOM (MODEL) TO DATA
print('applying the model to the catalog')
w = [som.winner(d) for d in tqdm(data)]
w_x = [c[0] for c in w]
w_y = [c[1] for c in w]

w_x = np.array(w_x)
w_y = np.array(w_y)

## SELECT CLUSTER AUTO
print('Grouping the data on the map')
fre = collections.Counter(w_x + 1j * w_y)
nodes = dict(sorted(dict(fre).items()))
xx = []
yy = []
ww = []
zz = []
for k in nodes.keys():
    if nodes[k] > 1:
        xx.append(k.real)
        yy.append(k.imag)
        ww.append((k.real, k.imag))
        zz.append(k)

clustering = AgglomerativeClustering(linkage='average', n_clusters=None, distance_threshold=1).fit(ww)  # 35
result_auto_nodes = clustering.labels_ + 1

result_auto = [-999] * len(w)
for i in range(len(w)):
    c = w_x[i] + 1j * w_y[i]
    if c in zz:
        result_auto[i] = result_auto_nodes[zz.index(c)]
    else:
        result_auto[i] = 0

result = result_auto.copy()
# frequency count
frequency = collections.Counter(result_auto)
print(dict(sorted(dict(frequency).items())))

## Save Result in a stand alone dataframe
par = pd.read_csv(catalogue_par_file)
par['cluster'] = [np.nan for _ in range(len(par.datetime))]
par['cluster'][~nanid] = result_auto
par["cluster_str"] = par["cluster"].astype(str)

par['classe'] = [np.nan for _ in range(len(par.datetime))]
time = pd.to_datetime(par.datetime, format='%Y-%m-%dT%H:%M:%S')
par['datetime'] = time

# create a copy without the NaN value
cross = par.copy()
cross = cross.dropna(subset=['cluster'], how="all")
cross.loc[:, 'Delta_days'] = cross.loc[:, 'Tj'] / (60 * 60 * 24)
cross['w_x'] = w_x
cross['w_y'] = w_y

## Create Probability matrix
print('Create a probability matrix')
clust_id = [x for x in set(result_auto) if x != 0]
xCluster = np.array([np.mean(cross.query('cluster==' + str(x))['w_x']) for x in clust_id])
yCluster = np.array([np.mean(cross.query('cluster==' + str(x))['w_y']) for x in clust_id])

## Measure all feat

cross = cross.rename(columns={"maxmean": "Cj"})

idy = ['', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# idy=['']

RPerCluster = [np.mean([np.mean(cross.query('cluster==' + str(x))['Rj' + y]) for y in idy]) for x in clust_id]
TPerCluster = [np.mean([np.mean(cross.query('cluster==' + str(x))['Tj' + y]) for y in idy]) for x in clust_id]
NearPerCluster = [np.mean([np.mean(cross.query('cluster==' + str(x))['Nnear']) for y in ['']]) for x in clust_id]

if 'intensity_norm' in Features:
    MPerCluster = [np.mean([np.mean(cross.query('cluster==' + str(x))['intensity_norm']) for y in ['']]) for x in
                   clust_id]
else:
    MPerCluster = [1] * len(clust_id)

if 'bval' in Features:
    BPerCluster = [np.mean([np.mean(cross.query('cluster==' + str(x))['bval']) for y in ['']]) for x in clust_id]
else:
    BPerCluster = [1] * len(clust_id)

# P with Near and Intensity capé
def ECmax(L, x):
    return (np.percentile(L, 90) - L[x]) / np.percentile(L, 90)

def ECmin(L, x):
    return (L[x] - np.percentile(L, 10)) / np.percentile(L, 10)

def EC1(L, x):
    return abs((L[x] - 1) / 1)

P = [(ECmax(RPerCluster, x) + ECmax(TPerCluster, x) + ECmin(NearPerCluster, x) + EC1(MPerCluster, x) + EC1(BPerCluster, x),
      ECmin(RPerCluster, x) + ECmin(TPerCluster, x) + ECmax(NearPerCluster, x) - EC1(MPerCluster, x) - EC1(BPerCluster, x))
     for x in range(len(TPerCluster))
     ]

Probability = [(np.exp(P[x][0]) / np.sum(np.exp(P[x])), np.exp(P[x][1]) / np.sum(np.exp(P[x]))) for x in range(len(P))]
p_r = [(x[0], 1 - x[1])[np.isnan(x[0])] for x in Probability]
p_b = [(x[1], 1 - x[0])[np.isnan(x[1])] for x in Probability]
p_a = [(1 - abs(0.5 - np.max(x)) / 0.5, 0)[np.isnan(np.max(x))] for x in Probability]

# define interpolation processe based on number of point
inter_pro = 'nearest'

# plot probability map

def map_value(M, xCluster=xCluster, yCluster=yCluster, n=parameters.n_neurons):
    # R = np.zeros((int(max(xCluster))+1,int(max(yCluster))+1))
    R = np.empty((n, n)) * np.nan
    k = 0
    for x, y in zip(xCluster, yCluster):
        R[int(x)][int(y)] = M[k]
        k += 1
    return R.T

plt.figure(figsize=(30, 9))
plt.subplot(1, 3, 1)
plt.pcolor(map_value(p_b))

cbar = plt.colorbar(alpha=1)
plt.clim(0, 1)
plt.axis('equal')
plt.ylim([0, parameters.n_neurons])

cbar.set_label('Probability for an event to be a Non Crisis Event', rotation=270, labelpad=20)
plt.xlabel('x coordinate of the nodes')
plt.ylabel('y coordinate of the nodes')

plt.subplot(1, 3, 2)
plt.pcolor(map_value(p_r))
cbar = plt.colorbar(alpha=1)
plt.clim(0, 1)
cbar.set_label('Probability for an event to be a Crisis Event', rotation=270, labelpad=20)
plt.xlabel('x coordinate of the nodes')
plt.ylabel('y coordinate of the nodes')
plt.axis('equal')
plt.ylim([0, parameters.n_neurons])

plt.subplot(1, 3, 3)
plt.pcolor(map_value(p_a))
cbar = plt.colorbar(alpha=1)
plt.clim(0, 1)
plt.axis('equal')
plt.ylim([0, parameters.n_neurons])
cbar.set_label('Map of uncertainty', rotation=270, labelpad=20)
plt.xlabel('x coordinate of the nodes')
plt.ylabel('y coordinate of the nodes')
plt.savefig(parameters.folder_output + '/graph/class++3.png')

# adding result to dataframe
Proba_grid_back = map_value(p_b)
incertitude_grid = map_value(p_a)
ev_incertitude = [-999] * len(data)
ev_Probability = [-999] * len(data)
ev_result = [-999] * len(data)
for k in tqdm(range(len(cross))):
    if Proba_grid_back[w_y[k]][w_x[k]] > 0.5:
        ev_result[k] = 1
        ev_incertitude[k] = incertitude_grid[w_y[k]][w_x[k]]
        ev_Probability[k] = Proba_grid_back[w_y[k]][w_x[k]]
        # result_auto[k]=-1
    else:
        ev_result[k] = 0
        ev_incertitude[k] = incertitude_grid[w_y[k]][w_x[k]]
        ev_Probability[k] = 1 - Proba_grid_back[w_y[k]][w_x[k]]

cross['probability'] = ev_Probability
cross['incertitude'] = ev_incertitude
cross['classe_bin'] = ev_result
cross['classe'] = [('rep', 'back')[k] for k in ev_result]

# dic_rev={'0':'back','1':'back','2':'back','3':'back','4':'back','5':'rep','6':'rep','7':'back','8':'back','9':'back'}
cond = np.array(p_b) > 0.5
dic = {'rep': [x for x in list(np.array(clust_id)[~cond[:]]) if x != 0] + [-2],
       'back': [x for x in list(np.array(clust_id)[cond[:]]) if x != 0] + [-1], '?': [0]}
dic_rev = {}
for k in dic.keys():
    for i in dic[k]:
        dic_rev[str(i)] = k
dic_c = {'rep': 'r', 'back': 'g', '?': 'b'}

a_file = open(parameters.folder_output + "/dic.pkl", "wb")
pickle.dump(dic_rev, a_file)
a_file.close()

cross.to_csv(parameters.folder_output + '/Clustering.csv')
print(' {} Non Crisis and {} Crisis'.format(sum(cross.classe_bin), len(cross.classe_bin) - sum(cross.classe_bin)))

## SAVE SOME PLOT
repback_rename = {'rep': "Crisis_event", 'back': "Non_Crisis_event", '?': '?'}

# PLOT THE MAP
plt.figure(figsize=(20, 9))
plt.subplot(1, 2, 1)
color = sb.color_palette("icefire", len(set(result_auto)))
# color=['magenta','pink','cyan','orange','red','blue','green','lime','yellow','black','deepskyblue','darkgoldenrod','aquamarine','indigo','hotpink','khaki','gray','mediumorchid','steelblue','olive','tomato','navy','lavender']
for c in set(result_auto):
    idx_target = result_auto == c
    if not (type(True) == type(idx_target)):
        plt.scatter(w_x[idx_target] + (np.random.rand(len(w_x[idx_target])) - .5),
                    w_y[idx_target] + (np.random.rand(len(w_x[idx_target])) - .5),
                    s=50, label='cluster ' + str(c), color=color[list(set(result_auto)).index(c)])

plt.legend(loc='upper right')
plt.xlabel('x coordinate of the nodes')
plt.ylabel('y coordinate of the nodes')
plt.grid()

result = np.array(result)

plt.subplot(1, 2, 2)
dic_c = {'rep': 'r', 'back': 'g', '?': 'b'}
for categorie in list(dic):
    idx_target = [i in dic[categorie] for i in cross.cluster]
    plt.scatter(w_x[idx_target] + (np.random.rand(len(w_x[idx_target])) - .5),
                w_y[idx_target] + (np.random.rand(len(w_x[idx_target])) - .5),
                s=50, label=repback_rename[str(categorie)], c=dic_c[str(categorie)])

plt.legend(loc='upper right')
plt.xlabel('x coordinate of the nodes')
plt.ylabel('y coordinate of the nodes')
plt.grid()

plt.savefig(parameters.folder_output + '/graph/SOMmap.png')