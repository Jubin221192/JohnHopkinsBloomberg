import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.plotly as py
from sklearn.model_selection import train_test_split
import math as m
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

"""
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

"""
med_nas = pd.read_csv("D:/INVITRODB_V3_1_SUMMARY/JM_med_dili.csv")

med_nas = med_nas.drop('Unnamed: 0', axis = 1)

def cal(x, y):
    return x/y

med_nas['Cratio'] = cal(med_nas['Cmax'],med_nas['EC50'])

med_nas['EC50'].dtypes

def classification(col):
    if col>= 1:
        return "MostDili"
    elif 0 < col < 1:
        return "NoDili"
    #elif 0.2 <= col < 1:
        #return "LessDili"
    #elif 0 < col <= 0.2:
        #return "NoDili"
    else:
        return "AmbiDili"

med_nas['Conorm_Class'] = med_nas['Cratio'].apply(classification)



# Dili Categories Bar plot

med_nas['Conorm_Class'].value_counts()
col = med_nas['Conorm_Class'].map({'MostDili':'g', 'NoDili':'y'})

ax = med_nas['Conorm_Class'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Dili Categories",color=col, fontsize= 13)
med_nas.isnull().values.any()

ax.set_xlabel("Dili Types")
ax.set_ylabel("Frequency")
ax.set_alpha(0.8)

sns.set_style('darkgrid')
sns.distplot(med_nas['Cratio'])

med_nas[med_nas['Cratio'].isnull()]

med_nas['Cratio'] = pd.to_numeric(med_nas['Cratio'], errors='coerce')
med_nas = med_nas.dropna(subset=['Cratio'])

med_nas['Cratio'] = med_nas['Cratio'].astype(int)

list = ['EC50','Cmax','Cratio']


def histo(df,val):
    for i in range(0,len(val)):
        nam = val[i]
        plt.figure(i+1)
        sns.set_style('darkgrid')
        sns.distplot(df[nam])


plt.show()

histo(med_nas,list)

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
    if x=='NoDili':
        return 0
    else:
        return 1

med_nas['Conorm_Class'] = med_nas['Conorm_Class'].apply(score_to_numeric)

med_nas['chnm'].nunique()

#encoding the drugs 

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(med_nas['chnm'])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

print(le_name_mapping)
med_nas['chnm'].replace(le_name_mapping, inplace= True)

# inversing the dictionary:

inv_map = {v: k for k, v in le_name_mapping.items()}
med_nas['chnm'].replace(inv_map, inplace= True)

# Dropping unwanted columns Analysis

med_nas_new = med_nas.drop(med_nas.columns[[0,1]], axis =1)
med_nas_ml = med_nas.drop('code', axis = 1) 

med_targ = pd.get_dummies(med_nas_ml, columns =['TestName'], drop_first = True)

        
# calculate the correlation matrix
corr = med_nas_new.corr()

med_nas['TestName'].nunique()

# plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# Segregating the data set

x= med_targ.drop(med_targ.columns[[3,4]], axis = 1).values
#x = med_targ.loc[:,med_targ.columns != 'Conorm_Class'].values
y = med_targ.loc[:,['Conorm_Class']].values


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

finalDf = pd.concat([eig_vec_data, med_targ[['Conorm_Class']]], axis = 1)


# Splitting data into training and test:

len(finalDf.columns)

finalDf = finalDf.dropna(how='any',axis=0) 
finalDf.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(finalDf.iloc[:,0:15], finalDf[['Conorm_Class']], test_size=0.2)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# Importing necessary libraries...
import collections 
import numpy as np

def pre_prob(y):
    y_dict = collections.Counter(y)
    pre_probab = np.ones(2)
    for i in range(0, 2):
        pre_probab[i] = y_dict[i]/y.shape[0]
    return pre_probab

def mean_var(X, y):
    n_features = X.shape[1]
    m = np.ones((2, n_features))
    v = np.ones((2, n_features))
    n_0 = np.bincount(y)[np.nonzero(np.bincount(y))[0]][0]
    x0 = np.ones((n_0, n_features))
    x1 = np.ones((X.shape[0] - n_0, n_features))
    
    k = 0
    for i in range(0, X.shape[0]):
        if y[i] == 0:
            x0[k] = X[i]
            k = k + 1
    k = 0
    for i in range(0, X.shape[0]):
        if y[i] == 1:
            x1[k] = X[i]
            k = k + 1
        
    for j in range(0, n_features):
        m[0][j] = np.mean(x0.T[j])
        v[0][j] = np.var(x0.T[j])*(n_0/(n_0 - 1))
        m[1][j] = np.mean(x1.T[j])
        v[1][j] = np.var(x1.T[j])*((X.shape[0]-n_0)/((X.shape[0]
                                                      - n_0) - 1))
    return m, v # mean and variance 

def prob_feature_class(m, v, x):
    n_features = m.shape[1]
    pfc = np.ones(2)
    for i in range(0, 2):
        product = 1
        for j in range(0, n_features):
            product = product * (1/m.sqrt(2*3.14*v[i][j])) * m.exp(-0.5
                                 * pow((x[j] - m[i][j]),2)/v[i][j])
        pfc[i] = product
    return pfc

def GNB(X, y, x):
    m, v = mean_var(X, y)
    pfc = prob_feature_class(m, v, x)
    pre_probab = pre_prob(y)
    pcf = np.ones(2)
    total_prob = 0
    for i in range(0, 2):
        total_prob = total_prob + (pfc[i] * pre_probab[i])
    for i in range(0, 2):
        pcf[i] = (pfc[i] * pre_probab[i])/total_prob
    prediction = int(pcf.argmax())
    return m, v, pre_probab, pfc, pcf, prediction


# executing the Gaussian Naive Bayes for the test instance...
m, v, pre_probab, pfc, pcf, prediction = GNB(X_train, y_train, X_test)
print(m) # Output given below...(mean for 2 classes of all features)
print(v) # Output given below..(variance for 2 classes of features)
print(pre_probab) # Output given below.........(prior probabilities)
print(pfc) # Output given below............(posterior probabilities)
print(pcf) # Conditional Probability of the classes given test-data
print(prediction) # Output given below............(final prediction)


# Logistic Regression


X_train, X_test, y_train, y_test = train_test_split(finalDf.iloc[:,0:15], finalDf[['Conorm_Class']], test_size=0.2)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

log_pred_file = pd.concat(X_test, y_pred)

print(classification_report(y_test, y_pred))

# Random Forest


# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(X_train, y_train)

# Actual class predictions
rf_predictions = model.predict(X_test)

print('Accuracy of random forest on test set: {:.2f}'.format(model.score(X_test, y_test)))

# Probabilities for each class
rf_probs = model.predict_proba(X_test)[:, 1]

# Calculate roc auc
roc_value = roc_auc_score(y_test, rf_probs)









