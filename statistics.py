# %% Modules
import pandas as pd
import matplotlib.pyplot as plt
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

    cases.iloc[i,1] = ('{}\\{}\\results_{}\\data_{}.xlsx'.format(pathDir,cases.iloc[i,0],cases.iloc[i,0],cases.iloc[i,0]))

# %% Create separate array for each Gleason grade
healthy = np.zeros((1,4))
g33 = np.zeros((1,4))
g34 = np.zeros((1,4))
g43 = np.zeros((1,4))
g44 = np.zeros((1,4))
g45 = np.zeros((1,4))
g55 = np.zeros((1,4))

# %% Import data and append it to arrays
max = np.zeros((cases.shape[0],3)) # Area, Equivalent Diamater, Major Axis Length

for i in range(cases.shape[0]):
    temp = pd.read_excel(cases.loc[i,'statPath'],usecols='A:D').values
    max[i,0] = np.max(temp[:,0])
    max[i,1] = np.max(temp[:,1])
    max[i,2] = np.max(temp[:,2])

    # Assign values
    for i in range(temp.shape[0]):
        if temp[i,3] == 0:
            healthy = np.vstack((healthy,temp[i,:]))
        elif temp[i,3] == 33:
            g33 = np.vstack((healthy,temp[i,:]))
        elif temp[i,3] == 34:
            g34 = np.vstack((healthy,temp[i,:]))
        elif temp[i,3] == 44:
            g44 = np.vstack((healthy,temp[i,:]))
        elif temp[i,3] == 45:
            g45 = np.vstack((healthy,temp[i,:]))
        elif temp[i,3] == 55:
            g55 = np.vstack((healthy,temp[i,:]))

# %% Plot Histograms
fig, ax = plt.subplots(2, 1, sharex=True)
cancer = [g33[1:g33.shape[0],0],g34[1:g34.shape[0],0],g44[1:g44.shape[0],0],g45[1:g45.shape[0],0]]
# bins = np.linspace(0,np.max(max[:,0]),1000)
bins = np.linspace(0,0.125,1000)

colors = ['b','g','y','orange']
labels = ['3+3','3+4','4+4','4+5']
ax[0].hist(healthy[1:healthy.shape[0],0],bins=bins,label='Healthy')
ax[0].legend()
ax[1].hist(cancer,bins=bins,color=colors,label=labels)
ax[1].legend()
fig.show()
# %%
