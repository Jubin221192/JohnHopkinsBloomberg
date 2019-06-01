import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.plotly as py


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
# tox21_new['tgt_abbr'].nunique()

"""
to = tox21_new.loc[:,['chnm','CmaxStand','EC','Cratio','Conorm_Class']]


# plotting a correlation chart
for i in to.Conorm_Class.astype('category'):
    to[i] = to.Conorm_Class == i

for i in to.chnm.astype('category'):
    to[i] = to.chnm == i

"""
# conversting the test targets
def score_to_numeric(x):
    if x=='LessDili':
        return 0
    else:
        return 1

tox21_new['Conorm_Class'] = tox21_new['Conorm_Class'].apply(score_to_numeric)




# Dropping unwanted columns Analysis
for col in ['chnm','testtarget']:
    tox21_new = tox21_new.drop(col, axis =1)
    
tox_new =tox21_new
tox_new = pd.get_dummies(tox_new, columns =['testname'], drop_first = True)

        
# calculate the correlation matrix
corr = tox_new.corr()

# plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)

x = tox_new.loc[:,tox_new.columns != 'Conorm_Class'].values
y = tox_new.loc[:,['Conorm_Class']].values


# Standardizing

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(x)

# Eigendecomposition - Computing Eigenvectors and Eigenvalues

mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# Selecting Principal Component

# Sorting eigen values in decreasing order
sorted_index = eig_vals.argsort()[::-1]
eig_vals = eig_vals[sorted_index]
eig_vecs = eig_vecs[:,sorted_index]
eig_vals
eig_vecs

eig_vecs = eig_vecs[:,:15]
eig_vecs

# Transforming data into new sample space
eig_vec_data = pd.DataFrame(np.real(eig_vecs))

eig_vec_data.columns = ['pc1', 'pc2', 'pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13','pc14','PC15']

finalDf = pd.concat([eig_vec_data, tox_new[['Conorm_Class']]], axis = 1)

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=15)
Y_sklearn = sklearn_pca.fit_transform(X_std)



