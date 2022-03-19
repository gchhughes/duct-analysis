# %% Modules
import pandas as pd
import matplotlib as plt
import numpy as np
# %% Obtain paths to statistics
pathDir = r'C:\Users\griff\Box\CASIT\Files for Derrick\duct-analysis'
trackerDir = pathDir + '\Tracker.xlsx'

# Find completed cases using Tracker spreadsheet
tracker = pd.read_excel(trackerDir,sheet_name=0)
logical = np.zeros(tracker.shape[0])
for i in range(tracker.shape[0]):
    if tracker.iloc[i,2] == 1:
        logical[i] = 1
logical = logical.astype('bool')
cases = pd.DataFrame(tracker.loc[logical,'ID'])
cases = cases.reset_index(drop=True)

# Create path to each case's statistics
cases['statPath'] = pd.Series(dtype='string')
for i in range(cases.shape[0]):
    cases.iloc[i,0] = str(cases.iloc[i,0])
    cases.iloc[i,0] = cases.iloc[i,0].zfill(3)

    cases.iloc[i,1] = ('{}\\{}\\results_{}\\results_{}.xlsx'.format(pathDir,cases.iloc[i,0],cases.iloc[i,0],cases.iloc[i,0]))
# %% Import all data into dataframe
cases = cases.assign(Area = np.nan,EquivDiam = np.nan, MajorAxisLength = np.nan, Cancer = np.nan, Slice = np.nan)
#for i in range(cases.shape[0]):

# %% Create separate df for each Gleason score
healthy = pd.DataFrame(columns = ['Area','EquivDiam','MajorAxisLength'])
g33 = pd.DataFrame(columns = ['Area','EquivDiam','MajorAxisLength'])
g34 = pd.DataFrame(columns = ['Area','EquivDiam','MajorAxisLength'])
g43 = pd.DataFrame(columns = ['Area','EquivDiam','MajorAxisLength'])
g44 = pd.DataFrame(columns = ['Area','EquivDiam','MajorAxisLength'])
g45 = pd.DataFrame(columns = ['Area','EquivDiam','MajorAxisLength'])
g55 = pd.DataFrame(columns = ['Area','EquivDiam','MajorAxisLength'])

# %% Plot Histograms