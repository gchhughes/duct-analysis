# %% Modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, zscore

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

print(cases.head())

# %% Create separate array for each Gleason grade
healthy = np.zeros((1,4))
g33 = np.zeros((1,4))
g34 = np.zeros((1,4))
g43 = np.zeros((1,4))
g44 = np.zeros((1,4))
g45 = np.zeros((1,4))
g54 = np.zeros((1,4))
g55 = np.zeros((1,4))
totCancer = np.zeros((1,4))

# %% Import data and append it to arrays
max = np.zeros((cases.shape[0],3)) # Area, Equivalent Diamater, Major Axis Length

for i in range(cases.shape[0]):
    temp = pd.read_excel(cases.loc[i,'statPath'],usecols='A:D').values
    max[i,0] = np.max(temp[:,0])
    max[i,1] = np.max(temp[:,1])
    max[i,2] = np.max(temp[:,2])

    # Assign values
    for ii in range(temp.shape[0]):
        if temp[ii,3] == 0:
            healthy = np.vstack((healthy,temp[ii,:]))
        elif temp[ii,3] == 33:
            g33 = np.vstack((g33,temp[ii,:]))
            totCancer = np.vstack((totCancer,temp[ii,:]))
        elif temp[ii,3] == 34:
            g34 = np.vstack((g34,temp[ii,:]))
            totCancer = np.vstack((totCancer,temp[ii,:]))
        elif temp[ii,3] == 43:
            g43 = np.vstack((g43,temp[ii,:]))
            totCancer = np.vstack((totCancer,temp[ii,:]))
        elif temp[ii,3] == 44:
            g44 = np.vstack((g44,temp[ii,:]))
            totCancer = np.vstack((totCancer,temp[ii,:]))
        elif temp[ii,3] == 45:
            g45 = np.vstack((g45,temp[ii,:]))
            totCancer = np.vstack((totCancer,temp[ii,:]))
        elif temp[ii,3] == 54:
            g54 = np.vstack((g54,temp[ii,:]))
            totCancer = np.vstack((totCancer,temp[ii,:]))
        elif temp[ii,3] == 55:
            g55 = np.vstack((g55,temp[ii,:]))
            totCancer = np.vstack((totCancer,temp[ii,:]))
    print(i)

# %% Check arrays
print(g33.shape)
print(g34.shape)
print(g43.shape)
print(g44.shape)
print(g45.shape)
print(g54.shape)
print(g55.shape)

# %% Remove first row
healthy = healthy[1:healthy.shape[0],:]
g33 = g33[1:g33.shape[0],:]
g34 = g34[1:g34.shape[0],:]
g43 = g43[1:g43.shape[0],:]
g44 = g44[1:g44.shape[0],:]
g45 = g45[1:g45.shape[0],:]
g54 = g54[1:g54.shape[0],:]
g55 = g55[1:g55.shape[0],:]
totCancer = totCancer[1:totCancer.shape[0],:]

# %% Plot Histograms
i = 0 # 0: Area; 1: Equivalent Diameter; 2: Major Axis Length
fig, ax = plt.subplots(2, 1, sharex=True)
plt1 = [healthy[:,i],totCancer[:,i]]
# plt2 = [g33[:,i],g34[:,i],g43[:,i],g44[:,i],g45[:,i],g54[:,i],g55[:,i]]
# plt2 = [g33[:,i],g34[:,i],g43[:,i],g44[:,i]]
# bins = np.linspace(0,np.max(max[:,0]),1000)
bins = np.linspace(0,0.25,100)
# colors = ['b','g','y','orange','red','magenta','purple']
# labels = ['3+3','3+4','4+3','4+4','4+5','5+4','5+5']
# colors = ['b','g','y','orange']
# labels = ['3+3','3+4','4+3','4+4']

ax[0].hist(plt1,bins=bins,color=['b','r'],label=['Healthy','Total Cancer'])
ax[0].legend()
ax[1].hist(plt2,bins=bins,color=colors,label=labels)
# ax[1].hist([g34[:,i]],bins=bins)
ax[1].legend(title='Gleason Grade')
fig.show()

# %% Import data using append vs vstack
healthy = []
g33 = []
g34 = []
g43 = []
g44 = []
g45 = []
g54 = []
g55 = []
totCancer = []

max = np.zeros((cases.shape[0],3)) # Area, Equivalent Diamater, Major Axis Length

for i in range(cases.shape[0]):
    temp = pd.read_excel(cases.loc[i,'statPath'],usecols='A:D').values
    max[i,0] = np.max(temp[:,0])
    max[i,1] = np.max(temp[:,1])
    max[i,2] = np.max(temp[:,2])

    print(temp.shape)
    # Assign values
    for ii in range(temp.shape[0]):
        if temp[ii,3] == 0:
            healthy = np.append(healthy,[temp[ii,:]],axis=0)
        elif temp[ii,3] == 33:
            g33 = np.append(g33,temp[ii,:],axis=0)
            totCancer = np.append(totCancer,temp[ii,:],axis=0)
        elif temp[ii,3] == 34:
            g34 = np.append(g34,temp[ii,:],axis=0)
            totCancer = np.append(totCancer,temp[ii,:],axis=0)
        elif temp[ii,3] == 43:
            g43 = np.append(g43,temp[ii,:],axis=0)
            totCancer = np.append(totCancer,temp[ii,:],axis=0)
        elif temp[ii,3] == 44:
            g44 = np.append(g44,temp[ii,:],axis=0)
            totCancer = np.append(totCancer,temp[ii,:],axis=0)
        elif temp[ii,3] == 45:
            g45 = np.append(g45,temp[ii,:],axis=0)
            totCancer = np.append(totCancer,temp[ii,:],axis=0)
        elif temp[ii,3] == 54:
            g54 = np.append(g54,temp[ii,:],axis=0)
            totCancer = np.append(totCancer,temp[ii,:],axis=0)
        elif temp[ii,3] == 55:
            g55 = np.append(g55,temp[ii,:],axis=0)
            totCancer = np.append(totCancer,temp[ii,:],axis=0)
    print(i)

# %% Import data using concatenate
# %% Import data using append vs vstack
healthy = []
g33 = []
g34 = []
g43 = []
g44 = []
g45 = []
g54 = []
g55 = []
totCancer = []

max = np.zeros((cases.shape[0],3)) # Area, Equivalent Diamater, Major Axis Length

for i in range(cases.shape[0]):
    temp = pd.read_excel(cases.loc[i,'statPath'],usecols='A:D').values
    max[i,0] = np.max(temp[:,0])
    max[i,1] = np.max(temp[:,1])
    max[i,2] = np.max(temp[:,2])

    print(temp.shape)
    # Assign values
    for ii in range(temp.shape[0]):
        if temp[ii,3] == 0:
            healthy = np.concatenate(([healthy],[temp[ii,:]]),axis=0)
        elif temp[ii,3] == 33:
            g33 = np.concatenate((g33,[temp[ii,:]]),axis=0)
            totCancer = np.concatenate((totCancer,[temp[ii,:]]),axis=0)
        elif temp[ii,3] == 34:
            g34 = np.concatenate((g34,[temp[ii,:]]),axis=0)
            totCancer = np.concatenate((totCancer,[temp[ii,:]]),axis=0)
        elif temp[ii,3] == 43:
            g43 = np.concatenate((g43,[temp[ii,:]]),axis=0)
            totCancer = np.concatenate((totCancer,[temp[ii,:]]),axis=0)
        elif temp[ii,3] == 44:
            g44 = np.concatenate((g44,[temp[ii,:]]),axis=0)
            totCancer = np.concatenate((totCancer,[temp[ii,:]]),axis=0)
        elif temp[ii,3] == 45:
            g45 = np.concatenate((g45,[temp[ii,:]]),axis=0)
            totCancer = np.concatenate((totCancer,[temp[ii,:]]),axis=0)
        elif temp[ii,3] == 54:
            g54 = np.concatenate((g54,[temp[ii,:]]),axis=0)
            totCancer = np.concatenate((totCancer,[temp[ii,:]]),axis=0)
        elif temp[ii,3] == 55:
            g55 = np.concatenate((g55,[temp[ii,:]]),axis=0)
            totCancer = np.concatenate((totCancer,[temp[ii,:]]),axis=0)
    print(i)

# %% Test np.r_
# %% Import data using append vs vstack
healthy = np.zeros((1,4))
g33 = np.zeros((1,4))
g34 = np.zeros((1,4))
g43 = np.zeros((1,4))
g44 = np.zeros((1,4))
g45 = np.zeros((1,4))
g54 = np.zeros((1,4))
g55 = np.zeros((1,4))
totCancer = np.zeros((1,4))

max = np.zeros((cases.shape[0],3)) # Area, Equivalent Diamater, Major Axis Length

for i in range(cases.shape[0]):
    temp = pd.read_excel(cases.loc[i,'statPath'],usecols='A:D').values
    max[i,0] = np.max(temp[:,0])
    max[i,1] = np.max(temp[:,1])
    max[i,2] = np.max(temp[:,2])

    print(temp.shape)
    # Assign values
    for ii in range(temp.shape[0]):
        if temp[ii,3] == 0:
            healthy = np.r_['0,2',healthy,temp[ii,:]]
        elif temp[ii,3] == 33:
            g33 = np.r_['0,2',g33,temp[ii,:]]
            totCancer = np.r_['0,2',totCancer,temp[ii,:]]
        elif temp[ii,3] == 34:
            g34 = np.r_['0,2',g34,temp[ii,:]]
            totCancer = np.r_['0,2',totCancer,temp[ii,:]]
        elif temp[ii,3] == 43:
            g43 = np.r_['0,2',g43,temp[ii,:]]
            totCancer = np.r_['0,2',totCancer,temp[ii,:]]
        elif temp[ii,3] == 44:
            g44 = np.r_['0,2',g44,temp[ii,:]]
            totCancer = np.r_['0,2',totCancer,temp[ii,:]]
        elif temp[ii,3] == 45:
            g45 = np.r_['0,2',g45,temp[ii,:]]
            totCancer = np.r_['0,2',totCancer,temp[ii,:]]
        elif temp[ii,3] == 54:
            g54 = np.r_['0,2',g54,temp[ii,:]]
            totCancer = np.r_['0,2',totCancer,temp[ii,:]]
        elif temp[ii,3] == 55:
            g55 = np.r_['0,2',g55,temp[ii,:]]
            totCancer = np.r_['0,2',totCancer,temp[ii,:]]

# %% Z-Score the data
zhealthy = zscore(healthy,ddof=1)
zg33 = zscore(g33,ddof=1)
zg34 = zscore(g34,ddof=1)
zg43 = zscore(g43,ddof=1)
zg44 = zscore(g44,ddof=1)
ztotCancer = zscore(totCancer,ddof=1)

# %% KS Test
stat = np.zeros((5,3))
pval = np.zeros((5,3))

for i in range(stat.shape[1]):
    stat[0,i],pval[0,i] = ks_2samp(zhealthy[:,i],ztotCancer[:,i])
    stat[1,i],pval[1,i] = ks_2samp(zhealthy[:,i],zg33[:,i])
    stat[2,i],pval[2,i] = ks_2samp(zhealthy[:,i],zg34[:,i])
    stat[3,i],pval[3,i] = ks_2samp(zhealthy[:,i],zg43[:,i])
    stat[4,i],pval[4,i] = ks_2samp(zhealthy[:,i],zg44[:,i])

"""
How to interpret KS Statistic:
- The null hypothesis is that the two distributions are identical
- "If the KS statistic is small or the p-value is high, then we cannot reject the null hypothesis in favor of the alternative."
"""
# %% Print KS Test Results;
for i in range(stat.shape[0]):
    for ii in range(stat.shape[1]):
        print('MEAN stat: {}; pval: {}'.format(stat[i,ii],pval[i,ii]))
        print('Equiv Diam stat: {}; pval: {}'.format(stat[i,ii],pval[i,ii]))
        print('MAL stat: {}; pval: {}'.format(stat[i,ii],pval[i,ii]))
# %% Mean values for each measurement
avg = np.zeros((5,3))
avg[0,0] = np.mean(healthy[:,0],axis=0)
avg[0,1] = np.mean(healthy[:,1],axis=0)
avg[0,2] = np.mean(healthy[:,2],axis=0)
avg[1,0] = np.mean(totCancer[:,0],axis=0)
avg[1,1] = np.mean(totCancer[:,1],axis=0)
avg[1,2] = np.mean(totCancer[:,2],axis=0)
# %% 
