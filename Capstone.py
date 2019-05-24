import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

tox21 = pd.read_csv("C:/Users/jubin/Dropbox/capstone_project/Stream_2_Tox21ToxCast in vitro mechanistic data/medNAS_dili_tt.csv",encoding='cp1252')
tox21.describe(include=['object'])
# function to extract specific columns

col = ['chnm','CmaxStand','testname','EC','testtarget']

co= ['casn','code','CmaxStand','EC','Conorm2_Class']

def sel_colmns(d_frame, col):
    new = d_frame.loc[:, col]
    return new

tox21_new = sel_colmns(tox21, col)

tox21_new['chnm'].nunique()
print("shape: {}".format(tox21_new.shape))

def cal(x, y):
    return x/y

tox21_new['Cratio'] = cal(tox21_new['CmaxStand'],tox21_new['EC'])


def classification(col):
    if col>= 1:
        return "MostDili"
    elif 0.2 <= col < 1:
        return "LessDili"
    elif 0 < col <= 0.2:
        return "NoDili"
    else:
        return "AmbiDili"

tox21_new['Conorm_Class'] = tox21_new['Cratio'].apply(classification)



# Dili Categories Bar plot

tox21_new['Conorm_Class'].value_counts()
col = tox21_new['Conorm_Class'].map({'MostDili':'g', 'LessDili':'b'})

ax = tox21_new['Conorm_Class'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Dili Categories",color=col, fontsize= 13)

ax.set_xlabel("Dili Types")
ax.set_ylabel("Frequency")
ax.set_alpha(0.8)

sns.set_style('darkgrid')
sns.distplot(tox21_new['EC'])


list = ['EC','CmaxStand','Cratio']


def histo(df,val):
    for i in range(0,len(val)):
        nam = val[i]
        plt.figure(i+1)
        sns.set_style('darkgrid')
        sns.distplot(df[nam])


plt.show()

histo(tox21_new,list)

# sns.catplot(x="Conorm_Class", kind = "count", palette="ch:.25", data=tox21_new)
#
# tox21_new.hist(by ='Conorm_Class')


# counting the number of drugs
tox21_new['tgt_abbr'].nunique()

to = tox21_new.loc[:,['chnm','CmaxStand','EC','Cratio','Conorm_Class']]


# plotting a correlation chart
for i in to.Conorm_Class.astype('category'):
    to[i] = to.Conorm_Class == i

for i in to.chnm.astype('category'):
    to[i] = to.chnm == i


# calculate the correlation matrix
corr = to.corr()

# plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)



for i in range(len(tox21_new))