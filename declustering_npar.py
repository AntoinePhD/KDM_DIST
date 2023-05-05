# import lib
import pandas as pd
from scipy.stats import kurtosis
import os
import numpy as np
import parameters
from output.CTD import T,D
from tqdm import tqdm

# open cat
cat = pd.read_csv(parameters.folder_output+'/'+parameters.cat_formated)

dtime = pd.to_datetime(cat.time, format='%Y-%m-%dT%H:%M:%S')
cat['datetime'] = dtime
cat['time'] = dtime

# some parameter to set
st_id = 11  # the min is 11
ed_id = len(cat.time)

from multiprocessing import Pool


def fp(j, cat=cat):
    fp_result = pd.DataFrame()
    corr_temp = []
    corr_temp2 = []
    kurto_sta = []
    std_sta = []

    #if j % 100 == 0:
    #    print(round(j, 1))
    # find the ten nearest event and get Tij Rij Dist and father id
    ten_near = [x.split() for x in os.popen('./tennearest {}'.format(j + 1)).read().split('\n')][:-1]
    tn_id = [int(float(x[0]) - 1) for x in ten_near if float(x[3]) != 9999]
    distRT = {int(float(x[0]) - 1): float(x[3]) for x in ten_near if float(x[3]) != 9999}
    Rj = {int(float(x[0]) - 1): float(x[1]) for x in ten_near if float(x[3]) != 9999}
    Tj = {int(float(x[0]) - 1): float(x[2]) for x in ten_near if float(x[3]) != 9999}

    x=list(Rj.keys())[0]
    return ['not_used', round(Rj[x], 2), round(Tj[x], 2), cat.datetime[j], 1,
            1, 1, 1, 1, ] + [Rj[k] for k in Rj.keys()] + [Tj[k] for k in Tj.keys()] + [1 for h in tn_id] + tn_id



if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    with Pool(8) as p:
        pool = p.map(fp, list(range(st_id, ed_id)))

    etj = [x[0] for x in pool]
    Rj = [x[1] for x in pool]
    Tj = [x[2] for x in pool]
    time = [x[3] for x in pool]
    std = [x[4] for x in pool]
    kurto = [x[5] for x in pool]
    corrme = [x[6] for x in pool]
    corrma = [x[7] for x in pool]

    Rj0 = [x[9] for x in pool]
    Rj1 = [x[10] for x in pool]
    Rj2 = [x[11] for x in pool]
    Rj3 = [x[12] for x in pool]
    Rj4 = [x[13] for x in pool]
    Rj5 = [x[14] for x in pool]
    Rj6 = [x[15] for x in pool]
    Rj7 = [x[16] for x in pool]
    Rj8 = [x[17] for x in pool]
    Rj9 = [x[18] for x in pool]

    Tj0 = [x[19] for x in pool]
    Tj1 = [x[20] for x in pool]
    Tj2 = [x[21] for x in pool]
    Tj3 = [x[22] for x in pool]
    Tj4 = [x[23] for x in pool]
    Tj5 = [x[24] for x in pool]
    Tj6 = [x[25] for x in pool]
    Tj7 = [x[26] for x in pool]
    Tj8 = [x[27] for x in pool]
    Tj9 = [x[28] for x in pool]

    Cj0 = [x[29] for x in pool]
    Cj1 = [x[30] for x in pool]
    Cj2 = [x[31] for x in pool]
    Cj3 = [x[32] for x in pool]
    Cj4 = [x[33] for x in pool]
    Cj5 = [x[34] for x in pool]
    Cj6 = [x[35] for x in pool]
    Cj7 = [x[36] for x in pool]
    Cj8 = [x[37] for x in pool]
    Cj9 = [x[38] for x in pool]

    tn0 = [x[39] for x in pool]
    tn1 = [x[40] for x in pool]
    tn2 = [x[41] for x in pool]
    tn3 = [x[42] for x in pool]
    tn4 = [x[43] for x in pool]
    tn5 = [x[44] for x in pool]
    tn6 = [x[45] for x in pool]
    tn7 = [x[46] for x in pool]
    tn8 = [x[47] for x in pool]
    tn9 = [x[48] for x in pool]

    # saving
    save = {
        'datetime': time, 'etj': etj, 'Rj': Rj, 'Tj': Tj, 'stdd': std, 'maxmean': corrme, 'maxmax': corrma,
        'kurto': kurto, 'Cj0': Cj0, 'Cj1': Cj1, 'Cj2': Cj2, 'Cj3': Cj3, 'Cj4': Cj4, 'Cj5': Cj5, 'Cj6': Cj6,
        'Cj7': Cj7, 'Cj8': Cj8, 'Cj9': Cj9,
        'Tj0': Tj0, 'Tj1': Tj1, 'Tj2': Tj2, 'Tj3': Tj3, 'Tj4': Tj4, 'Tj5': Tj5, 'Tj6': Tj6, 'Tj7': Tj7,
        'Tj8': Tj8, 'Tj9': Tj9,
        'Rj0': Rj0, 'Rj1': Rj1, 'Rj2': Rj2, 'Rj3': Rj3, 'Rj4': Rj4, 'Rj5': Rj5, 'Rj6': Rj6, 'Rj7': Rj7,
        'Rj8': Rj8, 'Rj9': Rj9,'tn0':tn0,'tn1':tn1,'tn2':tn2,'tn3':tn3,'tn4':tn4,'tn5':tn5,'tn6':tn6,'tn7':tn7,'tn8':tn8,'tn9':tn9
    }

    save = pd.DataFrame(save)
    result = pd.merge(cat, save, on="datetime")

    result[
        ['lon', 'lat', 'years', 'mag', 'depth', 'time', 'datetime', 'etj', 'Rj', 'Tj', 'stdd', 'maxmean', 'maxmax',
         'kurto','Cj1', 'Cj2', 'Cj3', 'Cj4', 'Cj5', 'Cj6', 'Cj7', 'Cj8', 'Cj9',
         'Tj1', 'Tj2', 'Tj3', 'Tj4', 'Tj5', 'Tj6', 'Tj7', 'Tj8', 'Tj9',
         'Rj1', 'Rj2', 'Rj3', 'Rj4', 'Rj5', 'Rj6', 'Rj7', 'Rj8', 'Rj9',
         'tn0', 'tn1', 'tn2', 'tn3', 'tn4', 'tn5', 'tn6', 'tn7', 'tn8', 'tn9']].to_csv(parameters.folder_output+'/'+parameters.declustering_output)
