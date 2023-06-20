# KDM_DIST
This is a declustering algorythm produced by [Septier&al](https://www.authorea.com/doi/full/10.22541/essoar.168167340.09761738)
Please cite any usage of this code in your publication

SOM [Vettigli&al](https://github.com/JustGlowing/minisom) is an unsupervised neural network-based dimensionality reduction algorithm used to represent a high-dimensional dataset as a low-dimensional (usually 2D) discretised pattern. The dimensionality reduction is performed while maintaining the topological structure of the input data.
The neural network is trained by competitive learning, as opposed to error-correction learning (e.g. back-propagation with gradient descent). After dimensionality reduction by SOM, each dataset used, defined as vectors of p features measured in n observations, is visualised on a 2D SOM map by clusters of observations. 
Observations in the proximal clusters have more similar feature values than observations in the distal clusters.

 We train the SOM with a 22-dimensional training dataset. Each seismic event is described by an input vector containing the values of the 22 features described above. The SOM learning process leads to the creation of a reduced 2D space representing the high-dimensional dataset.
All the events are agglomerated on the node that best represents them, creating a cluster of points that should be defined by shorter feature distances than all the other nodes.  

Once the clusters have been identified, we classify each SOM cluster. This interpretation of the SOM output gives a new representation of the studied catalogue by assigning each event to a class: crisis class or non-crisis class.

To obtain a relevant classification of each event class, we develop a centroid-based probabilistic approach.

Add the end of the process we obtained an average of 89.5% of balanced accuracy on synthetic catalogues. 
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
