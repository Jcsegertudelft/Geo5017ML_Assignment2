# Geo5017ML_Assignment2
Group 10: Job Segers, Timber Groeneveld, Akhil Veeranki 

# Running the code
To run the code: First run 1_Data_Prep.py, this will produce two csv files with the features calculated, and only the chosen features exported.
Afterwards 2_RF_Classifier.py and 3_SVM.py can be ran in any order. These will produce several figures for the learning curve and confusion matrix.

# Needed packages
numpy: for calculations

pandas: for DataFrames

matplotlib: for plots

seaborn: used in creation confusion matrix plot

sklearn: for the classifiers and confusion matrix

scipy: for convexhull and KDtree functionality

All packages can be installed through pip install *name of package*

# Data location
Data is expected to be stored in a folder Data\Data\xxx.xyz from the current working directory, where xxx are three digits. (Intermediate) Results are deposited in the current working directory.
