import pandas as pd
import parameters
from datetime import datetime

#reading the catalogue
cat = pd.read_csv(parameters.cat_raw,sep=None, engine='python')
#cat = pd.read_csv(parameters.cat_raw,delim_whitespace=True)

#create a datetime colum
if 'Time' in cat.columns and 'Date' in cat.columns:
	time = pd.to_datetime(cat.Date+'T'+cat.Time.replace(":60.",":59.", regex=True), format='%Y-%m-%dT%H:%M:%S')
	cat['time'] = time
elif 'time' in cat.columns :
	time = pd.to_datetime(cat.time)
	cat['time'] = time
elif '% Year' in cat.columns :
	Tframe = cat[['% Year','Day','Month','Hour','Minute','Sec']].rename(columns = {'Minute':'minute','% Year':'year','Day':'day','Hour':'hours','Month':'month','Sec':'second'})
	time = pd.to_datetime(Tframe[['year','day','month','hours','minute','second']])
	cat['time'] = time
elif 'HRMM' in cat.columns :
	cat['hours'] = [ ('0'*(4-len(str(x))) + str(x))[2:] for x in cat['HRMM'] ]
	cat['minute'] = [ ('0'*(4-len(str(x))) + str(x))[:2] for x in cat['HRMM']]
	Tframe = cat[['Year','Day','Month','hours','minute','Sec']].rename(columns = {'Day':'day','Month':'month','Sec':'second','Year':'year'})
	time = pd.to_datetime(Tframe[['year','day','month','hours','minute','second']])
	cat['time'] = time
else:
	Tframe = cat[['year','day','month','hours','mins','sec']].rename(columns = {'mins':'minute', 'sec':'second'})
	time = pd.to_datetime(Tframe[['year','day','month','hours','minute','second']])
	cat['time'] = time

if 'Magnitude' in cat.columns:
	cat['mag'] = cat['Magnitude']
elif 'mag' in cat.columns:
	cat['mag'] = cat['mag']
elif 'ML' in cat.columns:
	cat['mag'] = cat['ML']
elif 'ML_s' in cat.columns:
	cat['mag'] = cat['ML_s']
elif 'Mc' in cat.columns:
	cat['mag'] = cat['Mc']

if 'Lon' in cat.columns:
	cat['lon'] = cat['Lon']
elif 'Longitude' in cat.columns:
	cat['lon'] = cat['Longitude']
elif 'lon' in cat.columns:
	pass
else:
	cat['lon'] = cat['longitude']

if 'Lat' in cat.columns:
	cat['lat'] = cat['Lat']
elif 'Latitude' in cat.columns:
	cat['lat'] = cat['Latitude']
elif 'lat' in cat.columns:
	pass
else:
	cat['lat'] = cat['latitude']

if 'Depth' in cat.columns:
	cat['depth'] = cat['Depth']

#create date in years
L_new_time = []
for i in range(len(cat.time)):
    # New_time = time[i].year + time[i].month/13 + time[i].day/(31*13) + time[i].hour/(31*13*24) + time[i].minute/(31*13*24*60)
    New_time = time[i].year + time[i].timetuple().tm_yday / 367 + time[i].hour / (367 * 24) + time[i].minute / (367 * 24 * 60) + time[i].second / (367 * 24 * 60 * 60)
    L_new_time.append(float(str(New_time)[:9 + 5]))

cat['years'] = L_new_time

# Change the number of decimal so all the event have the same accuracy (make it readable for Fortran)
cat['lon'] = round(cat['lon'],4)
cat['lat'] = round(cat['lat'],4)
cat['mag'] = round(cat['mag'],1)

# Sorting event by date (it's important as fathers will only by considered as event before in the catalog)
#cat = cat.query('years>2021.08 and lon>-104.8 and lat <31.85 and lat >31.34')
cat = cat.sort_values(by=['time'])
cat.index = range(len(cat.time))

#writing the catalogue
cat[['lon', 'lat', 'years', 'mag', 'depth', 'time']].to_csv(parameters.folder_output+'/'+parameters.cat_formated)
