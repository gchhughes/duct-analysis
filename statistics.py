# %% Modules
from pickle import FALSE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, zscore, ttest_rel, ttest_ind



# %% Messing around with dictionaries
pathDir = r'C:\Users\griff\Box\CASIT\Files for Derrick\duct-analysis'
trackerPath = pathDir + '\Tracker.xlsx'
rawPath = pathDir + '\\raw.xlsx'
data = {} # Going to use nested dictionaries

# Find completed cases using Tracker spreadsheet
tracker = pd.read_excel(trackerPath,sheet_name=0)
logical = np.zeros(tracker.shape[0]).astype('bool')
for i in range(tracker.shape[0]):
    if tracker.iloc[i,2] == 1:
        logical[i] = True

temp = tracker.loc[logical,'ID'].values
numPts = len(temp)
for i in range(numPts):
    data[i] = {}
    data[i]['id'] = str(temp[i]).zfill(3)



# %% Import data using np.r_
measurements = ['Area (mm^2)','Equivalent Diameter (mm)','Major Axis Length (mm)']
grades = [0,33,34,43,44,45,54,55]
data['total'] = {}

# Arrays to store total ducts for each grade
for grade in grades:
    data['total'][grade] = np.zeros((1,10)) # Update size depending on spreadsheet

data['total']['sigCancer'] = np.zeros((1,10))
data['total']['ratio'] = np.zeros((1,2))

maxVal = np.zeros((numPts,3)) # Area, Equivalent Diamater, Major Axis Length

for i in range(numPts):
    data[i]['rawData'] = pd.read_excel(rawPath,usecols='A:J',
        sheet_name='{}'.format(data[i]['id'])).values
    data[i]['rawData'][:,7:10] = data[i]['rawData'][:,7:10]*1000
    
    # Record max values in case they're needed later
    maxVal[i,0] = np.max(data[i]['rawData'][:,7])
    maxVal[i,1] = np.max(data[i]['rawData'][:,8])
    maxVal[i,2] = np.max(data[i]['rawData'][:,9])

    # Arrays to store patient specific data
    for grade in grades:
        data[i][grade] = np.zeros((1,10))

    data[i]['sigCancer'] = np.zeros((1,10))
    data[i]['ratio'] = np.zeros((1,2))

    print('{}: {}'.format(data[i]['id'],data[i]['rawData'].shape))

    # Assign values
    for ii in range(data[i]['rawData'].shape[0]):

        # Patient specific duct sizes (Need to update if column headers change)
        if data[i]['rawData'][ii,2] == 0:
            data[i][0] = np.r_['0,2',data[i][0],data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,2] == 33:
            data[i][33] = np.r_['0,2',data[i][33],data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,2] == 34:
            data[i][34] = np.r_['0,2',data[i][34],data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,2] == 43:
            data[i][43] = np.r_['0,2',data[i][43],data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,2] == 44:
            data[i][44] = np.r_['0,2',data[i][44],data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,2] == 45:
            data[i][45] = np.r_['0,2',data[i][45],data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,2] == 54:
            data[i][54] = np.r_['0,2',data[i][54],data[i]['rawData'][ii,:]]
        elif data[i]['rawData'][ii,2] == 55:
            data[i][55] = np.r_['0,2',data[i][55],data[i]['rawData'][ii,:]]

        # Ductal Ratio (Need to update if column headers change)
        if ii == 0: # Save first ratio we see
            tempRatio = data[i]['rawData'][ii,[2,6]]
            data['total']['ratio'] = np.r_['0,2',data['total']['ratio'],tempRatio]
            data[i]['ratio'] = np.r_['0,2',data[i]['ratio'],tempRatio]
        elif tempRatio[1] != data[i]['rawData'][ii,6]: # Don't update unless it changes
            tempRatio = data[i]['rawData'][ii,[2,6]]
            data['total']['ratio'] = np.r_['0,2',data['total']['ratio'],tempRatio]
            data[i]['ratio'] = np.r_['0,2',data[i]['ratio'],tempRatio]

    # Update total arrays in data['total'] dictionary and append sigCancer
    for grade in grades:
        # Remove first row that is all zeros
        data[i][grade] = data[i][grade][1:data[i][grade].shape[0],:]

        # Add to appropriate data['total'] array
        data['total'][grade] = np.append(data['total'][grade],data[i][grade],axis=0)

        if grade != 0 and grade != 33:
            data['total']['sigCancer'] = np.append(data['total']['sigCancer'],data[i][grade],axis=0)
            data[i]['sigCancer'] = np.append(data[i]['sigCancer'],data[i][grade],axis=0)

print('\nSize of array')
# Remove first row that is all zeros
for key in data['total']:
    data['total'][key] = data['total'][key][1:data['total'][key].shape[0],:]
    print('{}: {}'.format(key,data['total'][key].shape))



# %% Separate density by healthy vs cancer (and no 3+3)
rBools = ['r0Bool','r33Bool','r34Bool','r43Bool','r44Bool']

# Totals
for key in data:
    for rBool in rBools:
        data[key][rBool] = np.zeros(data[key]['ratio'].shape[0]).astype('bool')
    
    for i in range(data[key]['ratio'].shape[0]):
        if data[key]['ratio'][i,0] == 0:
            data[key]['r0Bool'][i] = True
        elif data[key]['ratio'][i,0] == 33:
            data[key]['r33Bool'][i] = True
        elif data[key]['ratio'][i,0] == 34:
            data[key]['r34Bool'][i] = True
        elif data[key]['ratio'][i,0] == 43:
            data[key]['r43Bool'][i] = True
        elif data[key]['ratio'][i,0] == 44:
            data[key]['r44Bool'][i] = True

# %%
data['total']['ratio'][data['total']['r33Bool'],0],
'Insignificant Cancer (N={})'.format(data['total']['ratio'][data['total']['ratio33Bool']].shape[0]),



# %% Total Ductal Ratio Box & Whisker + Mean & StD
# Boxplots
x = [data['total']['ratio'][data['total']['r0Bool'],1],
    data['total']['ratio'][~(data['total']['r0Bool']+data['total']['r33Bool']),1]]
labels = ['Healthy (N={})'.format(data['total']['ratio'][data['total']['r0Bool']].shape[0]),
    'Significant Cancer (N={})'.format(data['total']['ratio'][~(data['total']['r0Bool']+data['total']['r33Bool'])].shape[0])]

plt.figure(figsize=(4,6),dpi=1000)
plt.boxplot(x,labels=labels)
plt.xticks(rotation=-20)
plt.title('Total Ductal Ratio (Median and IQR)')
plt.ylabel('Ductal Ratio')
plt.show()

x = [data['total']['ratio'][data['total']['r0Bool'],1],
    data['total']['ratio'][data['total']['r33Bool'],1],
    data['total']['ratio'][data['total']['r34Bool'],1],
    data['total']['ratio'][data['total']['r43Bool'],1],
    data['total']['ratio'][data['total']['r44Bool'],1]]
labels = ['Healthy','3+3','3+4','4+3','4+4']
plt.boxplot(x,labels=labels)
plt.title('Total Ductal Ratio (Median and IQR)')
plt.ylabel('Ductal Ratio')
plt.show()

# Calculate Mean & StD
data['total']['ratioMeanStD'] = np.zeros((2,7))
xAxis = np.arange(0,7,1)

for i in range(len(rBools)):
    data['total']['ratioMeanStD'][0,i] = np.mean(data['total']['ratio'][data['total'][rBools[i]],1])
    data['total']['ratioMeanStD'][1,i] = np.std(data['total']['ratio'][data['total'][rBools[i]],1])

# Cancer (w/ 3+3)
data['total']['ratioMeanStD'][0,5] = np.mean(data['total']['ratio'][~data['total']['r0Bool'],1])
data['total']['ratioMeanStD'][1,5] = np.std(data['total']['ratio'][~data['total']['r0Bool'],1])

# Cancer (No 3+3)
data['total']['ratioMeanStD'][0,6] = np.mean(data['total']['ratio'][~(data['total']['r0Bool']+data['total']['r33Bool']),1])
data['total']['ratioMeanStD'][1,6] = np.std(data['total']['ratio'][~(data['total']['r0Bool']+data['total']['r33Bool']),1])

plt.figure(figsize=(4,6),dpi=1000)
plt.scatter([0,1],data['total']['ratioMeanStD'][0,[0,6]])
plt.errorbar([0,1],data['total']['ratioMeanStD'][0,[0,6]],yerr=data['total']['ratioMeanStD'][1,[0,6]],fmt='o',capsize=3)
#plt.scatter(1,ratioMeanStD[0,6])
#plt.errorbar(1,ratioMeanStD[0,6],yerr=ratioMeanStD[1,6],fmt='o',capsize=3)
plt.title('Ductal Ratio (Mean and StD)')
plt.ylabel('Ductal Ratio')
# plt.axes().set_xticks([0,1],labels=['Healthy','Significant Cancer'])
plt.show()



# %% Patient Specific Ductal Ratio Box & Whisker + Mean & StD
xAxis = np.arange(0,7,1)

for pt in data:
    fig,ax = plt.subplots(2,2)
    plt.subplots_adjust(hspace = 0.3,wspace = 0.3)
    
    fig.suptitle('Patient {}: Ductal Ratio Boxplot, Mean & StD'.format(data[pt]['id']))
    fig.supylabel('Ductal Ratio')

    # Boxplots
    x0 = [data[pt]['ratio'][data[pt]['r0Bool'],0],
        data[pt]['ratio'][~data[pt]['r0Bool'],0],
        data[pt]['ratio'][~(data[pt]['r0Bool']+data[pt]['r33Bool']),0]]
    labels0 = ['H','C (w/ 33)','C (No 33)']

    ax[0,0].boxplot(x0,labels=labels0)
    ax[0,0].set_title('Median and IQR')

    x1 = [data[pt]['ratio'][data[pt]['r0Bool'],0],
        data[pt]['ratio'][data[pt]['r33Bool'],0],
        data[pt]['ratio'][data[pt]['r34Bool'],0],
        data[pt]['ratio'][data[pt]['r43Bool'],0],
        data[pt]['ratio'][data[pt]['r44Bool'],0]]
    labels1 = ['H','33','34','43','44']
    ax[1,0].boxplot(x1,labels=labels1)

    # Calculate Mean & StD
    data[pt]['ratioMeanStD'] = np.zeros((2,7))

    for i in range(len(rBools)):
        data[pt]['ratioMeanStD'][0,i] = np.mean(data[pt]['ratio'][data[pt][rBools[i]],0])
        data[pt]['ratioMeanStD'][1,i] = np.std(data[pt]['ratio'][data[pt][rBools[i]],0])

    # Cancer (w/ 3+3)
    data[pt]['ratioMeanStD'][0,5] = np.mean(data[pt]['ratio'][~data[pt]['r0Bool'],0])
    data[pt]['ratioMeanStD'][1,5] = np.std(data[pt]['ratio'][~data[pt]['r0Bool'],0])

    # Cancer (No 3+3)
    data[pt]['ratioMeanStD'][0,6] = np.mean(data[pt]['ratio'][~(data[pt]['r0Bool']+data[pt]['r33Bool']),0])
    data[pt]['ratioMeanStD'][1,6] = np.std(data[pt]['ratio'][~(data[pt]['r0Bool']+data[pt]['r33Bool']),0])

    ax[0,1].scatter(xAxis,data[pt]['ratioMeanStD'][0,:])
    ax[0,1].errorbar(xAxis,data[pt]['ratioMeanStD'][0,:],yerr=data[pt]['ratioMeanStD'][1,:],fmt='o',marker='^', capsize=3)
    ax[0,1].set_title('Mean & StD')
    ax[0,1].set_xlabel('H, 33, 34, 43, 44, C (w/ 33), C (No 33)')

    fig.delaxes(ax[1,1])

    data[pt]['ratioFig'] = fig
    plt.savefig('ratioFig{}.png'.format(data[pt]['id']),dpi=1000)



# %% Equivalent Diameter Histograms
i = 8

# Create bins for data
n = 10 # bin size
bins = np.arange(70,500+n,n)
# bins = np.insert(bins,0,0)
# bins = np.insert(bins,n+1,np.max(maxVal[:,i]))

# Histograms
fig,ax = plt.subplots(figsize=(9,5))
x = [np.clip(data['total']['sigCancer'][:,i],bins[0],bins[-1]),np.clip(data['total'][0][:,i],bins[0],bins[-1])]

ax.hist(x,bins=bins,density=True,color=['b','r'])
xlabels = bins[1:].astype(str)
xlabels[-1] += '+'
ax.set_xticklabels(xlabels)

fig.show()



# %% Equivalent Diameter Percentage
i = 8
j = 200 # ED we're screening for
k = 1000

data['total']['percent'] = np.zeros(6) # Update if 45,54,55 are included

for ii in range(5): # Update if 45,54,55 are included
    data['total']['percent'][ii] = sum(val > j for val in data['total'][grades[ii]][:,i])/(data['total'][grades[ii]].shape[0])

data['total']['percent'][5] = sum(val > j for val in data['total']['sigCancer'][:,i])/(data['total']['sigCancer'].shape[0])



# %% Total Measurements Box & Whisker + Mean & StD
xAxis = np.arange(0,6,1)
i = 8 # 7: Area; 8: Equivalent Diameter; 9: Major Axis Length

fig,ax = plt.subplots(2,2)
plt.subplots_adjust(hspace = 0.7,wspace = 0.3)
    
fig.suptitle('Total Equivalent Diameter Boxplot, Mean & StD')
fig.supylabel('Equivalent Diameter (??m)')

# Boxplots
x0 = [data['total'][0][:,i],data['total'][33][:,i],data['total']['sigCancer'][:,i]]
labels0 = ['H (N={})'.format(data['total'][0].shape[0]),'IC (N={})'.format(data['total'][33].shape[0]),'SC (N={})'.format(data['total']['sigCancer'].shape[0])]

ax[0,0].boxplot(x0,labels=labels0,sym='')
ax[0,0].tick_params(axis='x', rotation=-20)
ax[0,0].axhline(y=200, color='r', linestyle='-')
ax[0,0].set_title('Median and IQR')

# Update if cases have 4+5, 5+4, and 5+5
x1 = [data['total'][0][:,i],data['total'][33][:,i],data['total'][34][:,i],
    data['total'][43][:,i],data['total'][44][:,i]]
labels1 = ['H','33','34','43','44']
ax[1,0].boxplot(x1,labels=labels1,sym='')
ax[1,0].axhline(y=200, color='r', linestyle='-')

# Calculate Mean & StD
data['total']['measurement{}MeanStD'.format(i)] = np.zeros((2,6)) # Change length if all GGs are present

for iii in range(len(grades[0:5])): # Change length if all GGs are present
    data['total']['measurement{}MeanStD'.format(i)][0,iii] = np.mean(data['total'][grades[iii]][:,i])
    data['total']['measurement{}MeanStD'.format(i)][1,iii] = np.std(data['total'][grades[iii]][:,i])

# Significant cancer
data['total']['measurement{}MeanStD'.format(i)][0,5] = np.mean(data['total']['sigCancer'][:,i])
data['total']['measurement{}MeanStD'.format(i)][1,5] = np.std(data['total']['sigCancer'][:,i])

ax[0,1].scatter(xAxis,data['total']['measurement{}MeanStD'.format(i)][0,:])
ax[0,1].errorbar(xAxis,data['total']['measurement{}MeanStD'.format(i)][0,:],yerr=data['total']['measurement{}MeanStD'.format(i)][1,:],fmt='o')
ax[0,1].set_title('Mean & StD')
ax[0,1].axhline(y=200, color='r', linestyle='-')
ax[0,1].set_xticks(range(6),labels=['H','33','34','43','44','SC'])

fig.delaxes(ax[1,1])

data['total']['totalMeasurement{}'.format(i)] = fig
plt.savefig('totalMeasurement{}.png'.format(i),dpi=1000)



# %% Abstract Figure
fig,ax = plt.subplots(1,2)
plt.subplots_adjust(hspace = 0.7,wspace = 0.5)

# Ductal Ratio
ax[0].scatter([0,1],data['total']['ratioMeanStD'][0,[0,6]],c=['b','r'])
ax[0].errorbar(0,data['total']['ratioMeanStD'][0,0],yerr=data['total']['ratioMeanStD'][1,0],fmt='o',capsize=4,ecolor='b')
ax[0].errorbar(1,data['total']['ratioMeanStD'][0,6],yerr=data['total']['ratioMeanStD'][1,6],fmt='o',capsize=4,ecolor='r')
ax[0].set_xticks([0,1])
ax[0].set_xticklabels(['Benign','Cancerous'])
ax[0].set_xlim([-0.5,1.5])
ax[0].set_title('Ductal Ratio')
ax[0].set_ylabel('Ductal Ratio')

# Equivalent Diameter
ax[1].scatter([0,1],data['total']['measurement8MeanStD'][0,[0,5]],c=['b','r'])
ax[1].errorbar(0,data['total']['measurement8MeanStD'][0,0],yerr=data['total']['measurement8MeanStD'][1,0],fmt='o',capsize=3,ecolor='b')
ax[1].errorbar(1,data['total']['measurement8MeanStD'][0,5],yerr=data['total']['measurement8MeanStD'][1,5],fmt='o',capsize=3,ecolor='r')
ax[1].axhline(y=200, color='k', linestyle='--')
ax[1].axhline(y=70, color='k', linestyle='--')
ax[1].set_xticks([0,1])
ax[1].set_xticklabels(['Benign','Cancerous'])
ax[1].set_xlim([-0.5,1.5])
ax[1].set_title('Equivalent Diameter')
ax[1].set_ylabel('Equivalent Diameter (??m)')

plt.savefig('abstract-figure.png',dpi=1000)



# %% T-test between Benign and Cancerous Ductal Ratio
print(ttest_ind(data['total']['ratio'][data['total']['r0Bool'],1],data['total']['ratio'][~(data['total']['r0Bool']+data['total']['r33Bool']),1],equal_var=False))
print(ttest_ind(data['total']['ratio'][data['total']['r0Bool'],1],data['total']['ratio'][~(data['total']['r0Bool']+data['total']['r33Bool']),1]))



# %% Patient Specific Measurements Box & Whisker + Mean & StD
xAxis = np.arange(0,6,1)
i = 1 # 0: Area; 1: Equivalent Diameter; 2: Major Axis Length

for pt in data:
    fig,ax = plt.subplots(2,2)
    plt.subplots_adjust(hspace = 0.7,wspace = 0.3)
    
    fig.suptitle('Patient {}: Equivalent Diameter Boxplot, Mean & StD'.format(data[pt]['id']))
    fig.supylabel('Equivalent Diameter (??m^2)')

    sigCancer = np.concatenate((data[pt][34][:,i],data[pt][43][:,i],data[pt][44][:,i],data[pt][45][:,i],data[pt][54][:,i],data[pt][55][:,i]),axis=0)

    # Boxplots
    x0 = [data[pt][0][:,i],data[pt][33][:,i],sigCancer]
    labels0 = ['H (N={})'.format(data[pt][0].shape[0]),'IC (N={})'.format(data[pt][33].shape[0]),'SC (N={})'.format(sigCancer.shape[0])]

    ax[0,0].boxplot(x0,labels=labels0,sym='')
    ax[0,0].tick_params(axis='x', rotation=-20)
    ax[0,0].axhline(y=200, color='r', linestyle='-')
    ax[0,0].set_title('Median and IQR')

    # Update if cases have 4+5, 5+4, and 5+5
    x1 = [data[pt][0][:,i],data[pt][33][:,i],data[pt][34][:,i],
        data[pt][43][:,i],data[pt][44][:,i]]
    labels1 = ['H','33','34','43','44']
    ax[1,0].boxplot(x1,labels=labels1,sym='')
    ax[1,0].axhline(y=200, color='r', linestyle='-')

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
    ax[0,1].axhline(y=200, color='r', linestyle='-')
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
