import parameters
import pandas as pd
import numpy as np
import pickle
import collections
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
import os
from tqdm import tqdm
import output.CTD as CTD
from sklearn.metrics import r2_score
import warnings

# just to make the terminal clearer can be commented if needed
warnings.filterwarnings("ignore")

def bval_regress(mag, plot=False):
    box1 = np.arange(1, 6, 1)
    rep = np.histogram(mag, bins=box1)
    vbin = np.array([ rep[0][i] for i in range(len(rep[0])) if rep[0][i] > 0 ])
    bornbin = np.array([ rep[1][i] + 0.5 for i in range(len(rep[0])) if rep[0][i] > 0 ])
    N = vbin / len(mag)

    box = bornbin
    G = np.array([np.ones(np.size(np.transpose(box))), -np.array(box)])

    [x, r, rank, s] = np.linalg.lstsq(G.T, np.log10(N), rcond=None)
    # res= np.dot( np.dot(np.linalg.inv( np.dot(G.T,N)), G.T)  ,np.log10(N) )
    if plot:
        plt.plot(box, 10 ** (x[0] - x[1] * box))
        plt.plot(box, N)
        plt.legend(['regression', 'data'])
        plt.show()
    return x[1]

def tri_reg(Z):
    Y = np.array(list(range(1,11)))
    Z = np.array(Z)
    G = np.array([ Y , np.ones(np.size(Y))])
    [x, r, rank, s] = np.linalg.lstsq(G.T, Z, rcond=None)
    a=r2_score(x[0]*Y+x[1],Z)
    return [a,x[0],x[1]]

# Z = [ cat['Tj'+str(k)].iloc[5000] for k in ['',1,2,3,4,5,6,7,8,9] ]

## LOAD RESULT FILE
data = pd.read_csv(parameters.folder_output + '/' + parameters.declustering_output)
time = pd.to_datetime(data.time, format='%Y-%m-%dT%H:%M:%S')
data['time'] = time

data['Nnear'] = [-999]*len(data)
data['intensity'] = [-999]*len(data)
data['intensity_norm'] = [-999]*len(data)
data['bval'] = [-999]*len(data)

fd = max(CTD.T,2) # days
fk = max(CTD.D,2) # km


for ibg in tqdm(range(len(data))):
    #if ibg%500==0:
        #print(round(ibg/len(data)*100),'%')
    bg=data.iloc[ibg:ibg+1]
    t = bg.years.iloc[0]
    lat= bg.lat.iloc[0]
    lon=bg.lon.iloc[0]
    filt = (np.sqrt((data.lat-lat)**2 + (data.lon-lon)**2)*100 <= fk) * ((data.years - t) < 0) * (abs(data.years-t) <= fd/365)

    filt3 = (np.sqrt((data.lat - lat) ** 2 + (data.lon - lon) ** 2) * 100 <= 5*fk) * (abs(data.years - t) > 0) * (
                abs(data.years - t) <= fd*5 / 365)
    filt2 = (np.sqrt((data.lat-lat)**2 + (data.lon-lon)**2)*100 <= fk/2) * ((data.years - t) < 0) * (abs(data.years-t) <= fd/(2*365))

    if np.sum(filt)==0 or np.sum(filt2)==0 or np.mean(data.mag[filt])==0:
        data.Nnear[ibg] = 1
        data.intensity[ibg] = 1
        data.intensity_norm[ibg] = 1
        data.bval[ibg] = 1
        if np.sum(filt)>1:
            data.bval[ibg] = bval_regress(data.mag[filt3])
            data.intensity[ibg] = np.mean(data.mag[filt])
            data.Nnear[ibg] = np.sum(filt)/np.sum(filt3)
    else:
        data.bval[ibg] = bval_regress(data.mag[filt3])
        data.Nnear[ibg] = np.sum(filt2)/np.sum(filt)
        data.intensity[ibg] = np.mean(data.mag[filt])
        data.intensity_norm[ibg] = np.mean(data.mag[filt2])/np.mean(data.mag[filt])


plt.figure()
plt.subplot(2,1,1)
sb.lineplot(x=data.query('intensity!=0')['time'], y=data.query('intensity!=0')['intensity'], dashes=True,color='yellow')
#sb.scatterplot(y=yr,x=xr)
sb.scatterplot(x=data.query('intensity!=0')['time'],y=data.query('intensity!=0')['mag'])
plt.ylabel('intensité')
plt.xlabel('time')

plt.subplot(2,1,2)
sb.lineplot(x=data.query('intensity_norm!=0')['time'], y=data.query('intensity_norm!=0')['intensity_norm'], dashes=True,color='yellow')
#sb.scatterplot(y=yr,x=xr)
sb.scatterplot(x=data.query('intensity_norm!=0')['time'],y=data.query('intensity_norm!=0')['mag'])
plt.ylabel('intensité normé')
plt.xlabel('time')
plt.show(block=False)


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('bval', color=color)
ax1.plot(data.query('bval!=0')['time'], data.query('bval!=0')['bval'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('mag', color=color)  # we already handled the x-label with ax1
ax2.scatter(data.query('bval!=0')['time'], data.query('bval!=0')['mag'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show(block=False)



import plotly.graph_objects as go
from plotly.subplots import make_subplots

def ECDF2(x):
    x = np.sort(x)
    def result(v):
        return np.searchsorted(x, v, side='right') / x.size
    return result
    
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=np.unique(data.years), y=ECDF2(data.years)(np.unique(data.years)), name="Cumulative curve"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=data.query('bval!=0').years, y=data.query('bval!=0').bval, name="B value"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="time plot of productivity"
)

# Set x-axis title
fig.update_xaxes(title_text="years")

# Set y-axes titles
fig.update_yaxes(title_text="<b>b-value</b> ", secondary_y=False)
fig.update_yaxes(title_text="<b>percent of event</b>", secondary_y=True)

fig.write_image(parameters.folder_output+'/graph/bval.png',width=2000)

data.to_csv(parameters.folder_output + '/'+parameters.declustering_output)
