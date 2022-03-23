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

# %% Import data using np.r_
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

# %% Check arrays
print(g33.shape)
print(g34.shape)
print(g43.shape)
print(g44.shape)
print(g45.shape)
print(g54.shape)
print(g55.shape)

# %% Plot Histograms
i = 0 # 0: Area; 1: Equivalent Diameter; 2: Major Axis Length

bins = np.linspace(0,1,100)
# bins = np.linspace(0,np.max(max[:,i]),50)
# logbins = np.logspace(np.log10(0.0001),np.log10(bins.max()),len(bins))
colors = ['b','g','r','k']
labels = ['3+3','3+4','4+3','4+4']
xLbl = ['Area (mm^2)','Equivalent Diameter (mm)','Major Axis Length (mm)']

for i in range(3):
    fig, ax = plt.subplots(2, 1, sharex=True)
    plt1 = [healthy[:,i],totCancer[:,i]]
    plt2 = [g33[:,i],g34[:,i],g43[:,i],g44[:,i]]

    ax[0].hist(plt1,bins=bins,color=['b','r'],label=['Healthy','Total Cancer'],density=True)
    ax[0].legend()
    ax[1].hist(plt2,bins=bins,color=colors,label=labels,density=True)
    ax[1].legend(title='Gleason Grade')


    fig.supxlabel(xLbl[i])
    fig.supylabel('Frequency')
    fig.suptitle('Comparison of Cancerous and Healthy Duct Sizes')

    fig.show()

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
    ii = 0
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
