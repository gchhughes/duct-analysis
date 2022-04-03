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
measurements = ['Area (mm^2)','Equivalent Diameter (mm)','Major Axis Length (mm)']
grades = [0,33,34,43,44,45,54,55]
totDucts = {}

# Arrays to store total ducts for each grade
for grade in grades:
    totDucts[grade] = np.zeros((1,5)) # Update size depending on spreadsheet

totDucts['cancerNo33'] = np.zeros((1,5))
totDucts['cancer33'] = np.zeros((1,5))
totDucts['ratio'] = np.zeros((1,2))

maxVal = np.zeros((len(data),3)) # Area, Equivalent Diamater, Major Axis Length

for i in range(len(data)):
    data[i]['rawData'] = pd.read_excel(data[i]['dataPath'],usecols='A:E').values
    data[i]['rawData'][:,0:3] = data[i]['rawData'][:,0:3]*1000
    
    # Record max values in case they're needed later
    maxVal[i,0] = np.max(data[i]['rawData'][:,0])
    maxVal[i,1] = np.max(data[i]['rawData'][:,1])
    maxVal[i,2] = np.max(data[i]['rawData'][:,2])

    # Temporary arrays to store patient specific data
    for grade in grades:
        data[i][grade] = np.zeros((1,5))

    data[i]['cancerNo33'] = np.zeros((1,5))
    data[i]['cancer33'] = np.zeros((1,5))
    data[i]['ratio'] = np.zeros((1,2))

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

        # Ductal Ratio (Need to update if column headers change)
        if ii == 0: # Save first ratio we see
            tempRatio = data[i]['rawData'][ii,3:5]
            totDucts['ratio'] = np.r_['0,2',totDucts['ratio'],tempRatio]
            data[i]['ratio'] = np.r_['0,2',data[i]['ratio'],tempRatio]
        elif tempRatio[0] != data[i]['rawData'][ii,3]: # Don't update unless it changes
            tempRatio = data[i]['rawData'][ii,3:5]
            totDucts['ratio'] = np.r_['0,2',totDucts['ratio'],tempRatio]
            data[i]['ratio'] = np.r_['0,2',data[i]['ratio'],tempRatio]

    # Update total arrays in totDucts dictionary
    for grade in grades:
        # Remove first row that is all zeros
        data[i][grade] = data[i][grade][1:data[i][grade].shape[0],:]

        # Add to appropriate totDucts array
        totDucts[grade] = np.append(totDucts[grade],data[i][grade],axis=0)

        if grade != 0 and grade != 33:
            totDucts['cancerNo33'] = np.append(totDucts['cancerNo33'],data[i][grade],axis=0)
            data[i]['cancerNo33'] = np.append(data[i]['cancerNo33'],data[i][grade],axis=0)

        if grade != 0:
            totDucts['cancer33'] = np.append(totDucts['cancer33'],data[i][grade],axis=0)
            data[i]['cancer33'] = np.append(data[i]['cancer33'],data[i][grade],axis=0)

print('\nSize of array')
# Remove first row that is all zeros
for key in totDucts:
    totDucts[key] = totDucts[key][1:totDucts[key].shape[0],:]
    print('{}: {}'.format(key,totDucts[key].shape))

# %% Separate density by healthy vs cancer (and no 3+3)
ratioBools = ['ratio0Bool','ratio33Bool','ratio34Bool','ratio43Bool','ratio44Bool']

# Totals
for ratioBool in ratioBools:
    totDucts[ratioBool] = np.zeros(totDucts['ratio'].shape[0]).astype('bool')

for i in range(totDucts['ratio'].shape[0]):
    if totDucts['ratio'][i,1] == 0:
        totDucts['ratio0Bool'][i] = True
    elif totDucts['ratio'][i,1] == 33:
        totDucts['ratio33Bool'][i] = True
    elif totDucts['ratio'][i,1] == 34:
        totDucts['ratio34Bool'][i] = True
    elif totDucts['ratio'][i,1] == 43:
        totDucts['ratio43Bool'][i] = True
    elif totDucts['ratio'][i,1] == 44:
        totDucts['ratio44Bool'][i] = True

# Do the same for each patient
for pt in data:
    for ratioBool in ratioBools:
        data[pt][ratioBool] = np.zeros(data[pt]['ratio'].shape[0]).astype('bool')
    
    for i in range(data[pt]['ratio'].shape[0]):
        if data[pt]['ratio'][i,1] == 0:
            data[pt]['ratio0Bool'][i] = True
        elif data[pt]['ratio'][i,1] == 33:
            data[pt]['ratio33Bool'][i] = True
        elif data[pt]['ratio'][i,1] == 34:
            data[pt]['ratio34Bool'][i] = True
        elif data[pt]['ratio'][i,1] == 43:
            data[pt]['ratio43Bool'][i] = True
        elif data[pt]['ratio'][i,1] == 44:
            data[pt]['ratio44Bool'][i] = True

# %% Total Ductal Ratio Box & Whisker + Mean & StD
# Boxplots
x = [totDucts['ratio'][totDucts['ratio0Bool'],0],
    totDucts['ratio'][totDucts['ratio33Bool'],0],
    totDucts['ratio'][~(totDucts['ratio0Bool']+totDucts['ratio33Bool']),0]]
labels = ['Healthy (N={})'.format(totDucts['ratio'][totDucts['ratio0Bool']].shape[0]),
    'Insignificant Cancer (N={})'.format(totDucts['ratio'][totDucts['ratio33Bool']].shape[0]),
    'Significant Cancer (N={})'.format(totDucts['ratio'][~(totDucts['ratio0Bool']+totDucts['ratio33Bool'])].shape[0])]

plt.boxplot(x,labels=labels)
plt.xticks(rotation=-20)
plt.title('Total Ductal Ratio (Median and IQR)')
plt.ylabel('Ductal Ratio')
plt.show()

x = [totDucts['ratio'][totDucts['ratio0Bool'],0],
    totDucts['ratio'][totDucts['ratio33Bool'],0],
    totDucts['ratio'][totDucts['ratio34Bool'],0],
    totDucts['ratio'][totDucts['ratio43Bool'],0],
    totDucts['ratio'][totDucts['ratio44Bool'],0]]
labels = ['Healthy','3+3','3+4','4+3','4+4']
plt.boxplot(x,labels=labels)
plt.title('Total Ductal Ratio (Median and IQR)')
plt.ylabel('Ductal Ratio')
plt.show()

# Calculate Mean & StD
ratioMeanStD = np.zeros((2,7))
xAxis = np.arange(0,7,1)

for i in range(len(ratioBools)):
    ratioMeanStD[0,i] = np.mean(totDucts['ratio'][totDucts[ratioBools[i]],0])
    ratioMeanStD[1,i] = np.std(totDucts['ratio'][totDucts[ratioBools[i]],0])

# Cancer (w/ 3+3)
ratioMeanStD[0,5] = np.mean(totDucts['ratio'][~totDucts['ratio0Bool'],0])
ratioMeanStD[1,5] = np.std(totDucts['ratio'][~totDucts['ratio0Bool'],0])

# Cancer (No 3+3)
ratioMeanStD[0,6] = np.mean(totDucts['ratio'][~(totDucts['ratio0Bool']+totDucts['ratio33Bool']),0])
ratioMeanStD[1,6] = np.std(totDucts['ratio'][~(totDucts['ratio0Bool']+totDucts['ratio33Bool']),0])

plt.scatter(xAxis,ratioMeanStD[0,:])
plt.errorbar(xAxis,ratioMeanStD[0,:],yerr=ratioMeanStD[1,:],fmt='o')
plt.title('Ductal Ratio (Mean and StD)')
plt.ylabel('Ductal Ratio')
plt.xlabel('Healthy, 33, 34, 43, 44, Cancer (w/ 33), Cancer (No 33)')
plt.show()

# %% Patient Specific Box & Whisker + Mean & StD: Ductal Ratio
xAxis = np.arange(0,7,1)

for pt in data:
    fig,ax = plt.subplots(2,2)
    plt.subplots_adjust(hspace = 0.3,wspace = 0.3)
    
    fig.suptitle('Patient {}: Ductal Ratio Boxplot, Mean & StD'.format(data[pt]['id']))
    fig.supylabel('Ductal Ratio')

    # Boxplots
    x0 = [data[pt]['ratio'][data[pt]['ratio0Bool'],0],
        data[pt]['ratio'][~data[pt]['ratio0Bool'],0],
        data[pt]['ratio'][~(data[pt]['ratio0Bool']+data[pt]['ratio33Bool']),0]]
    labels0 = ['H','C (w/ 33)','C (No 33)']

    ax[0,0].boxplot(x0,labels=labels0)
    ax[0,0].set_title('Median and IQR')

    x1 = [data[pt]['ratio'][data[pt]['ratio0Bool'],0],
        data[pt]['ratio'][data[pt]['ratio33Bool'],0],
        data[pt]['ratio'][data[pt]['ratio34Bool'],0],
        data[pt]['ratio'][data[pt]['ratio43Bool'],0],
        data[pt]['ratio'][data[pt]['ratio44Bool'],0]]
    labels1 = ['H','33','34','43','44']
    ax[1,0].boxplot(x1,labels=labels1)

    # Calculate Mean & StD
    data[pt]['ratioMeanStD'] = np.zeros((2,7))

    for i in range(len(ratioBools)):
        data[pt]['ratioMeanStD'][0,i] = np.mean(data[pt]['ratio'][data[pt][ratioBools[i]],0])
        data[pt]['ratioMeanStD'][1,i] = np.std(data[pt]['ratio'][data[pt][ratioBools[i]],0])

    # Cancer (w/ 3+3)
    data[pt]['ratioMeanStD'][0,5] = np.mean(data[pt]['ratio'][~data[pt]['ratio0Bool'],0])
    data[pt]['ratioMeanStD'][1,5] = np.std(data[pt]['ratio'][~data[pt]['ratio0Bool'],0])

    # Cancer (No 3+3)
    data[pt]['ratioMeanStD'][0,6] = np.mean(data[pt]['ratio'][~(data[pt]['ratio0Bool']+data[pt]['ratio33Bool']),0])
    data[pt]['ratioMeanStD'][1,6] = np.std(data[pt]['ratio'][~(data[pt]['ratio0Bool']+data[pt]['ratio33Bool']),0])

    ax[0,1].scatter(xAxis,data[pt]['ratioMeanStD'][0,:])
    ax[0,1].errorbar(xAxis,data[pt]['ratioMeanStD'][0,:],yerr=data[pt]['ratioMeanStD'][1,:],fmt='o')
    ax[0,1].set_title('Mean & StD')
    ax[0,1].set_xlabel('H, 33, 34, 43, 44, C (w/ 33), C (No 33)')

    fig.delaxes(ax[1,1])

    data[pt]['ratioFig'] = fig
    plt.savefig('ratioFig{}.png'.format(data[pt]['id']),dpi=1000)

# %% Equivalent Diameter Histograms
i = 1

# Create bins for data
n = 10 # bin size
bins = np.arange(70,500+n,n)
# bins = np.insert(bins,0,0)
# bins = np.insert(bins,n+1,np.max(maxVal[:,i]))

# Histograms
fig,ax = plt.subplots(figsize=(9,5))

ax[0].hist(np.clip(totDucts[0][:,i],bins[0],bins[-1]),bins=bins)
xlabels = bins[1:].astype(str)
xlabels[-1] += '+'
ax[0].set_xticklabels(xlabels)

fig.show()

# %% Measurements Box & Whisker + Mean & StD
xAxis = np.arange(0,6,1)
i = 1 # 0: Area; 1: Equivalent Diameter; 2: Major Axis Length

for pt in data:
    fig,ax = plt.subplots(2,2)
    plt.subplots_adjust(hspace = 0.7,wspace = 0.3)
    
    fig.suptitle('Patient {}: Equivalent Diameter Boxplot, Mean & StD'.format(data[pt]['id']))
    fig.supylabel('Equivalent Diameter (µm^2)')

    sigCancer = np.concatenate((data[pt][34][:,i],data[pt][43][:,i],data[pt][44][:,i],data[pt][45][:,i],data[pt][54][:,i],data[pt][55][:,i]),axis=0)

    # Boxplots
    x0 = [data[pt][0][:,i],data[pt][33][:,i],sigCancer]
    labels0 = ['H (N={})'.format(data[pt][0].shape[0]),'IC (N={})'.format(data[pt][33].shape[0]),'SC (N={})'.format(sigCancer.shape[0])]

    ax[0,0].boxplot(x0,labels=labels0)
    ax[0,0].tick_params(axis='x', rotation=-20)
    ax[0,0].axhline(y=135, color='r', linestyle='-')
    ax[0,0].set_title('Median and IQR')

    # Update if cases have 4+5, 5+4, and 5+5
    x1 = [data[pt][0][:,i],data[pt][33][:,i],data[pt][34][:,i],
        data[pt][43][:,i],data[pt][44][:,i]]
    labels1 = ['H','33','34','43','44']
    ax[1,0].boxplot(x1,labels=labels1)
    ax[1,0].axhline(y=135, color='r', linestyle='-')

    # Calculate Mean & StD
    data[pt]['measurementsMeanStD'] = np.zeros((2,6)) # Change length if all GGs are present

    for iii in range(len(grades[0:5])): # Change length if all GGs are present
        data[pt]['measurementsMeanStD'][0,iii] = np.mean(data[pt][grades[iii]][:,i])
        data[pt]['measurementsMeanStD'][1,iii] = np.std(data[pt][grades[iii]][:,i])

    # Significant cancer
    data[pt]['measurementsMeanStD'][0,5] = np.mean(sigCancer)
    data[pt]['measurementsMeanStD'][1,5] = np.std(sigCancer)

    ax[0,1].scatter(xAxis,data[pt]['measurementsMeanStD'][0,:])
    ax[0,1].errorbar(xAxis,data[pt]['measurementsMeanStD'][0,:],yerr=data[pt]['measurementsMeanStD'][1,:],fmt='o')
    ax[0,1].set_title('Mean & StD')
    ax[0,1].axhline(y=135, color='r', linestyle='-')
    ax[0,1].set_xticks(range(6),labels=['H','33','34','43','44','SC'])

    fig.delaxes(ax[1,1])

    data[pt]['measurement{}Fig'.format(i)] = fig
    plt.savefig('measurement{}Fig{}.png'.format(i,data[pt]['id']),dpi=1000)

# %% Plot OLD Histograms
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
ii = 2 # 0: Area; 1: Equivalent Diameter; 2: Major Axis Length
dataType = ['Area', 'Equivalent Diameter', 'Major Axis Length']

for i in range(stat.shape[0]):
    print('{} stat: {}; pval: {}'.format(dataType[ii],stat[i,ii],pval[i,ii]))

# %% 200 um Dataset
