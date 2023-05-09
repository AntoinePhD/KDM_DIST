# KDM_DIST
This is a declustering algorythm produced by Septier&al
Please cite any usage of this code in your publication

## Usage
### Requirement
This code is based on Python3 and  Fortran 90 
The following python librairie are needed : 
```
subbprocess
pandas
glob
numpy 
pickle
scipy
tqdm
sklearn
minisom
```
For the Graphical output: 
```
matplotlib
seaborn
```
You can install all the package with 
```
pip install -r requirement.txt
```

For fortran the compiler `gfortran` is needed

### Quick start
After cloning the repository

0. Install requirement
1. Set the information in the paramaters.py (read the comment as some parameters have to be set in the .f90 files)
2. Run the main.py in the KDM folder
