# %% Modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, zscore, ttest_rel

# %% Messing around with dictionaries
pathDir = r'C:\Users\griff\Box\CASIT\Files for Derrick\duct-analysis'
trackerDir = pathDir + '\Tracker.xlsx'
data = {} # Going to use nested dictionaries

# Find completed cases using Tracker spreadsheet
tracker = pd.read_excel(trackerDir,sheet_name=0)
logical = np.zeros(tracker.shape[0]).astype('bool')
for i in range(tracker.shape[0]):
    if tracker.iloc[i,2] == 1:
        logical[i] = True

temp = tracker.loc[logical,'ID'].values
for i in range(len(temp)):
    data[i] = {}
    data[i]['id'] = str(temp[i]).zfill(3)
    data[i]['dataPath'] = '{}\\data\\data_{}.xlsx'.format(pathDir,data[i]['id'])

# %% Import data using np.r_
grades = [0,33,34,43,44,45,54,55]
totDucts = {}

# Arrays to store total ducts for each grade
for grade in grades:
    totDucts[grade] = np.zeros((1,5)) # Update size depending on spreadsheet

totDucts['cancerNo33'] = np.zeros((1,5))
totDucts['cancer33'] = np.zeros((1,5))
totDucts['density'] = np.zeros((1,2))

maxVal = np.zeros((len(data),3)) # Area, Equivalent Diamater, Major Axis Length

for i in range(len(data)):
    data[i]['rawData'] = pd.read_excel(data[i]['dataPath'],usecols='A:E').values
    data[i]['density'] = np.zeros((1,2))
    
    # Record max values in case they're needed later
    maxVal[i,0] = np.max(data[i]['rawData'][:,0])
    maxVal[i,1] = np.max(data[i]['rawData'][:,1])
    maxVal[i,2] = np.max(data[i]['rawData'][:,2])

    # Temporary arrays to store patient specific data
    for grade in grades:
        data[i][grade] = np.zeros((1,5))

    print('{}: {}'.format(data[i]['id'],data[i]['rawData'].shape))

    # Assign values
    for ii in range(data[i]['rawData'].shape[0]):

        # Patient specific duct sizes (Need to update if column headers change)
        if data[i]['rawData'][ii,4] == 0:
            data[i][0] = np.r_['0,2',data[i][0],data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,4] == 33:
            data[i][33] = np.r_['0,2',data[i][33],data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,4] == 34:
            data[i][34] = np.r_['0,2',data[i][34],data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,4] == 43:
            data[i][43] = np.r_['0,2',data[i][43],data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,4] == 44:
            data[i][44] = np.r_['0,2',data[i][44],data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,4] == 45:
            data[i][45] = np.r_['0,2',t45,data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,4] == 54:
            data[i][54] = np.r_['0,2',data[i][54],data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,4] == 55:
            data[i][55] = np.r_['0,2',data[i][55],data[i]['rawData'][ii,:]]

        # Density (Need to update if column headers change)
        if ii == 0:
            tempDensity = data[i]['rawData'][ii,3:5]
            totDucts['density'] = np.r_['0,2',totDucts['density'],tempDensity]
            data[i]['density'] = np.r_['0,2',data[i]['density'],tempDensity]
        elif tempDensity[0] != data[i]['rawData'][ii,3]:
            tempDensity = data[i]['rawData'][ii,3:5]
            totDucts['density'] = np.r_['0,2',totDucts['density'],tempDensity]
            data[i]['density'] = np.r_['0,2',data[i]['density'],tempDensity]

    # Update total arrays in totDucts dictionary
    for grade in grades:
        # Remove first row that is all zeros
        data[i][grade] = data[i][grade][1:data[i][grade].shape[0],:]

        # Add to appropriate totDucts array
        totDucts[grade] = np.append(totDucts[grade],data[i][grade],axis=0)

        if grade != 0 and grade != 33:
            totDucts['cancerNo33'] = np.append(totDucts['cancerNo33'],data[i][grade],axis=0)
        
        if grade != 0:
            totDucts['cancer33'] = np.append(totDucts['cancer33'],data[i][grade],axis=0)

print('\nNumber of elements in')
# Remove first row that is all zeros
for key in totDucts:
    totDucts[key] = totDucts[key][1:totDucts[key].shape[0],:]
    print('{}: {}'.format(key,totDucts[key].shape))

# %% Density Healthy Logical
logical = np.zeros(density.shape[0]).astype('bool')
lgcl33 = np.zeros(density.shape[0]).astype('bool')
for i in range(density.shape[0]):
    if density[i,1] == 0:
        logical[i] = 1
    if density[i,1] == 33:
        lgcl33[i] = 1
logical = logical.astype('bool')
lgcl33 = lgcl33.astype('bool')

# %% Mean/StD values for each measurement
avg = np.zeros((2,3))
med = np.zeros((2,3))
sd = np.zeros((2,3))
avgDensity = np.zeros(2)
sdDensity = np.zeros(2)

avg[0,0] = np.mean(healthy[:,0],axis=0)
avg[0,1] = np.mean(healthy[:,1],axis=0)
avg[0,2] = np.mean(healthy[:,2],axis=0)
avg[1,0] = np.mean(totCancer[:,0],axis=0)
avg[1,1] = np.mean(totCancer[:,1],axis=0)
avg[1,2] = np.mean(totCancer[:,2],axis=0)

med[0,0] = np.median(healthy[:,0],axis=0)
med[0,1] = np.median(healthy[:,1],axis=0)
med[0,2] = np.median(healthy[:,2],axis=0)
med[1,0] = np.median(totCancer[:,0],axis=0)
med[1,1] = np.median(totCancer[:,1],axis=0)
med[1,2] = np.median(totCancer[:,2],axis=0)

sd[0,0] = np.std(healthy[:,0],axis=0)
sd[0,1] = np.std(healthy[:,1],axis=0)
sd[0,2] = np.std(healthy[:,2],axis=0)
sd[1,0] = np.std(totCancer[:,0],axis=0)
sd[1,1] = np.std(totCancer[:,1],axis=0)
sd[1,2] = np.std(totCancer[:,2],axis=0)

avgDensity[0] = np.mean(density[logical,0])
avgDensity[1] = np.mean(density[~logical,0])
sdDensity[0] = np.std(density[logical,0])
sdDensity[1] = np.std(density[~logical,0])

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

# %% Boxplots
i = 0
plt.boxplot([healthy[:,i],totCancer[:,i],g33[:,i],g34[:,i],g43[:,i],g44[:,i]])
plt.ylim((0,0.1))
plt.show()

# %%
plt.boxplot(healthy[:,i])
plt.title('Healthy')
plt.show()
plt.boxplot(totCancer[:,i])
plt.title('totCancer')
plt.show()
plt.boxplot(g33[:,i])
plt.title('G33')
plt.show()
plt.boxplot(g34[:,i])
plt.title('G34')
plt.show()
plt.boxplot(g43[:,i])
plt.title('G43')
plt.show()
plt.boxplot(g44[:,i])
plt.title('G44')
plt.show()


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
ii = 2 # 0: Area; 1: Equivalent Diameter; 2: Major Axis Length
dataType = ['Area', 'Equivalent Diameter', 'Major Axis Length']

for i in range(stat.shape[0]):
    print('{} stat: {}; pval: {}'.format(dataType[ii],stat[i,ii],pval[i,ii]))



# %% Density Boxplot
plt.boxplot([density[logical,0],density[log33,0]])
plt.show()

# %% 200 um Dataset
