# KDM_DIST
KDM (Kohonen map based Declustering Methode) is a declustering algorythm produced by [Septier&al](https://www.authorea.com/doi/full/10.22541/essoar.168167340.09761738) .

Please cite any usage of this code in your publication
## Abstract
Declustering is in our case providing a classification between non-crisis and crisis earthquakes of a sismological catalague.
Non crisis event beging event that can't be link to any event prior to it (exemple of possible non_crisis: Lone event, mainshock, foreshock ...) and Crisis event earthquake that can be link to a preveious event (exemple : Aftershocks, Swarms, Geothermally induced event)

To do that we use an Self Organized map (SOM). SOM [(Vettigli&al)](https://github.com/JustGlowing/minisom) is an unsupervised neural network-based algorithm used to represent a high-dimensional dataset as a low-dimensional (usually 2D) discretised pattern. The dimensionality reduction is performed while maintaining the topological structure of the input data.
The neural network is trained by competitive learning, as opposed to error-correction learning (e.g. back-propagation with gradient descent). After dimensionality reduction by SOM, each dataset used, defined as vectors of p features measured in n observations, is visualised on a 2D SOM map by clusters of observations. 
Observations in the proximal clusters have more similar feature values than observations in the distal clusters.

We train the SOM with a 22-dimensional training dataset. Each seismic event is described by an input vector containing the values of the 22 features described above. The SOM learning process leads to the creation of a reduced 2D space representing the high-dimensional dataset.
All the events are agglomerated on the node that best represents them, creating a cluster of points that should be defined by shorter feature distances than all the other nodes.  

Once the clusters have been identified, we classify each SOM cluster. This interpretation of the SOM output gives a new representation of the studied catalogue by assigning each event to a class: crisis class or non-crisis class.

To obtain a relevant classification of each event class, we develop a centroid-based probabilistic approach (see [Septier&al](https://www.authorea.com/doi/full/10.22541/essoar.168167340.09761738)
) .

At the end of the process we obtain an average of 89.5% of balanced accuracy on synthetic catalogues. 
## Usage
### Requirement
This code use Python (> version 3) and Fortran 90 
The following python library are needed : 
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
matplotlib
seaborn
```
You can install all the package after pull from the GitHub with the command :
```
pip install -r requirement.txt
```
I recommend you to use conda to create a new environment for this code if you use other python programs. 

For fortran the compiler `gfortran` is needed (see [the installation guide](https://fortran-lang.org/en/learn/os_setup/install_gfortran/) if you are not used to fortran)

### Quick start
After cloning the repository

0. Install requirement
1. Set the information in the paramaters.py (read the comment as some parameters have to be set in the .f90 files)
2. Run the main.py in the KDM folder
