## kickstarter project
## YULIN HONG


# Load Libraries
import pandas
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split

########################
##### Import data ######
########################

from pathlib import Path
path = Path.cwd() # current working directory

df = pandas.read_excel(path.joinpath('Kickstarter.xlsx'))

########################
##### Pre-Processing ###
########################
df.isna().sum()
## The column launch_to_state_change_days is mostly nan
## This is a Variables for which 90% or more of the values are missing
## decide to remove this variable
df = df.drop(columns=['launch_to_state_change_days'])

## the variable catagory still has some missing values
## will deal with it later (after remove other states)

## for all tasks, only want observation that state take "successful" or "failure"
sat = ["failed","successful"]
for i in range(0,len(df["state"])):
    if df.loc[i,"state"] not in sat:
        df = df.drop([i])

df['state'].value_counts() ## check there are no other states in the dataframe

## check nan values again
df.isna().sum()

## since "category" has around 1500 missing value 
## it is not a small number nor big number
## decide to fill the missing values using the underlying distribution of this variable

prob = df.category.value_counts(normalize=True) ## calculate the probability of each vairable
#print(prob)
missing = df['category'].isnull() ## record missing values
df.loc[missing,'category'] = np.random.choice(prob.index, size=len(df[missing]),p=prob.values)## Fill missing values with probability 
df.isna().sum() ## no missing value
df.category.value_counts()
########################
## Setup the variables #
########################
## First of all, there are a lot of time related variable in this data
## for example "created_at"
## Based on my knoledge these variable should not have huge effect on the final state of the project
## thus I decide to remove these varaiable
## Secondely, project name and ID has nothing to do with the outcome of the project

df = df.drop(columns=["name","project_id","pledged","deadline","state_changed_at","created_at","launched_at","staff_pick","backers_count","usd_pledged","spotlight"])
a = df.columns.get_loc("deadline_weekday")
b = df.columns.get_loc("launched_at_hr")
df = df.drop(df.columns[a:b+1], axis=1)


## process categorical variables
df = pandas.get_dummies(df, columns = ['disable_communication',"category"])


## dealing with variable country
country_us = []
for i in range(0,len(df.country)):
    if df.country.iloc[i] == "US":
        country_us.append(1)
    else:
        country_us.append(0)
df['US'] = country_us

## dealing with variable currency
currency_usd = []
for i in range(0,len(df.currency)):
    if df.currency.iloc[i] == "USD":
        currency_usd.append(1)
    else:
        currency_usd.append(0)
df['USD'] = currency_usd

df = df.drop(columns = ['country','currency'])

## normalize goal values
normalized_goal = []
for i in range(0,len(df.goal)):
    x = df.goal.iloc[i]*df.static_usd_rate.iloc[i]
    normalized_goal.append(x)
df['normalized goal'] = normalized_goal

df = df.drop(columns = ['goal','static_usd_rate'])

y = df["state"]
var = df.drop(columns=["state"])
y_dummy = pandas.get_dummies(y, columns = ["state"])
y_dummy = y_dummy.drop(columns = ["failed"]) ## only use seccess indicator as y variable

##################
#### feature #####
#### selection ###
##################

## find collinear terms

df = pandas.concat([var,y_dummy], axis=1)

correlation = df.corr()
correlation_var = []
for j in range(0,len(correlation)-1):
    i=j+1
    while i < len(correlation):
        if abs(correlation.iloc[i,j]) >= 0.7:
            correlation_var.append([correlation.index[i],correlation.columns[j],correlation.iloc[i,j]])
        i = i+1

## need to perform feature selection on predictors
## Random Forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state = 0)

model = randomforest.fit(var,y)
model.feature_importances_
feature_result = pandas.DataFrame(list(zip(var.columns,model.feature_importances_)), columns = ['predictor','feature importance'])

## set the threshold of random forest feature selection result to be 0.01
## obtained the list of predictors
var_list = []
for i in range(len(feature_result.predictor)):
    if feature_result.loc[i,'feature importance'] >= 0.01:
        var_list.append(feature_result.predictor.loc[i])

## but some predictor in this var_list set is collinear
## refer to the correlation _var list
## should remove "name_len_clean", "blurb_len_clean"
var_list.remove("name_len_clean")
var_list.remove("blurb_len_clean")

X_rf = var[var_list]

## PCA
## PCA can deal with collinear terms no need to delete collinear term before pca

#from sklearn.decomposition import PCA
#pca = PCA(n_components=11)
#pca.fit(var)
#pca.explained_variance_ratio_
#pca.components_

## first 2 pca can explain 0.9996215448 of variability
#pca = PCA(n_components=2)
#pca.fit(var)
#X_pca = pca.transform(var)

########################
### Develope ###########
### Classification #####
### Models #############
########################

## LOGISTIC REGRESSION

# X_train,X_test,y_train,y_test = train_test_split(X_rf,y_dummy,test_size = 0.33,random_state = 5)

# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(max_iter=2000) 
# model1 = lr.fit(X_train,y_train)


# y_test_pred = model1.predict(X_test)
# ls_score_rf = metrics.accuracy_score(y_test, y_test_pred)

# X_train,X_test,y_train,y_test = train_test_split(X_pca,y_dummy,test_size = 0.33,random_state = 5)
# lr = LogisticRegression(max_iter=2000) 
# model1 = lr.fit(X_train,y_train)
# y_test_pred = model1.predict(X_test)
# ls_score_pca = metrics.accuracy_score(y_test, y_test_pred)

# ## KNN

# standardizer = StandardScaler()
# X_std = standardizer.fit_transform(X_rf)
# X_train,X_test,y_train,y_test = train_test_split(X_std, y_dummy,test_size = 0.33,random_state = 5)

# from sklearn.neighbors import KNeighborsClassifier
# ## uncomment the following lines to see how to find the optimal n_neighbors value
# #for i in range(1,20):
# #    knn = KNeighborsClassifier(n_neighbors=i)
# #    model2 = knn.fit(X_train,y_train)
# #    y_test_pred = model2.predict(X_test)
# #    print(i,metrics.accuracy_score(y_test,y_test_pred))
# ## not much difference, then choose i = 18
# knn = KNeighborsClassifier(n_neighbors=18)
# model2 = knn.fit(X_train,y_train)
# y_test_pred = model2.predict(X_test)
# knn_score_rf = metrics.accuracy_score(y_test,y_test_pred)

# standardizer = StandardScaler()
# X_std = standardizer.fit_transform(X_pca)
# X_train,X_test,y_train,y_test = train_test_split(X_std, y_dummy,test_size = 0.33,random_state = 5)
# ## uncomment the following lines to see how to find the optimal n_neighbors value
# #for i in range(1,20):
# #    knn = KNeighborsClassifier(n_neighbors=i)
# #    model2 = knn.fit(X_train,y_train)
# #    y_test_pred = model2.predict(X_test)
# #    print(i,metrics.accuracy_score(y_test,y_test_pred))
# ## i = 12 
# knn = KNeighborsClassifier(n_neighbors=12)
# model2 = knn.fit(X_train,y_train)
# y_test_pred = model2.predict(X_test)
# knn_score_pca = metrics.accuracy_score(y_test,y_test_pred) 


## RANDOM FOREST
# X_train,X_test,y_train,y_test = train_test_split(X_rf, y,test_size = 0.33,random_state = 5)

# from sklearn.ensemble import RandomForestClassifier
# randomforest = RandomForestClassifier(random_state=5)

# model3 = randomforest.fit(X_train,y_train) 
# y_test_pred = model3.predict(X_test)
# rf_score_rf = metrics.accuracy_score(y_test,y_test_pred)

# X_train,X_test,y_train,y_test = train_test_split(X_pca, y,test_size = 0.33,random_state = 5)

# from sklearn.ensemble import RandomForestClassifier
# randomforest = RandomForestClassifier(random_state=5)

# model3 = randomforest.fit(X_train,y_train) 
# y_test_pred = model3.predict(X_test)
# rf_score_pca = metrics.accuracy_score(y_test,y_test_pred)


## GRADIENT BOOST
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

## uncomment the following lines to see how to find the optimal min_samples_split value
#for i in range(1,10):
#    model4 = GradientBoostingClassifier(random_state=0,min_samples_split = i*2, n_estimators = 100)
#    scores = cross_val_score(estimator = model4, X=X_rf, y=y, cv=5)
#    print(i*2,':',np.average(scores))
## i =8

X_train,X_test,y_train,y_test = train_test_split(X_rf, y,test_size = 0.33,random_state = 5)
gbt = GradientBoostingClassifier(random_state=0,min_samples_split = 8, n_estimators = 100)
model4 = gbt.fit(X_train,y_train)
y_test_pred = model4.predict(X_test)
gbt_score_rf = metrics.accuracy_score(y_test,y_test_pred)

## uncomment the following lines to see how to find the optimal min_samples_split value
#for i in range(1,15):
#    model4 = GradientBoostingClassifier(random_state=0,min_samples_split = i*2, n_estimators = 100)
#    scores = cross_val_score(estimator = model4, X=X_pca, y=y, cv=5)
#    print(i*2,':',np.average(scores))
## i =22

# X_train,X_test,y_train,y_test = train_test_split(X_pca, y,test_size = 0.33,random_state = 5)
# gbt = GradientBoostingClassifier(random_state=0,min_samples_split = 8, n_estimators = 100)
# model4 = gbt.fit(X_train,y_train)
# y_test_pred = model4.predict(X_test)
# gbt_score_pca = metrics.accuracy_score(y_test,y_test_pred)

# ## combine all the testing result:
# rf_result = [ls_score_rf,knn_score_rf,rf_score_rf,gbt_score_rf]
# pca_result = [ls_score_pca,knn_score_pca,rf_score_pca,gbt_score_pca]
# models = ["logestic regression","knn","random forest","gradient boost"]
# classification_result = pandas.DataFrame({'random forest feature': rf_result,'pca features': pca_result,'models':models})
# classification_result = classification_result.set_index('models')

# import dataframe_image as dfi
# dfi.export(classification_result, '/Users/hong/Desktop/McGill/INSY662/ind_project/dataframe.png')

## gbt with rf features achieve the highest score

########################
### Develope ###########
### Cluster ############
### Models #############
########################
## features are considered invalid if they can only be realized after the prediction starts
## Thus for clustering model we still need to use the var list
## no need to do feature selection

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

standardizer = StandardScaler()
var_std = standardizer.fit_transform(var)

cluster_number = []
cluster_score = []
for i in range (2,11):    
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(var_std)
    labels = model.labels_
    cluster_number.append(i)
    cluster_score.append(np.average(silhouette_score(var,labels)))

kmeans_Result = pandas.DataFrame({'Number of Clusters': cluster_number,'Avg Silhouette Score':cluster_score})

kmeans = KMeans(n_clusters=3)
model = kmeans.fit(var_std)
labels = model.predict(var_std)


## find out who are in each cluster
cluster_3 = pandas.DataFrame(columns = var.columns.tolist())
for i in range(0,len(labels)):
    if labels[i] == 2:
        cluster_3 = cluster_3.append(var.iloc[i])

cluster_1 = pandas.DataFrame(columns = var.columns.tolist())
for i in range(0,len(labels)):
    if labels[i] == 1:
        cluster_1 = cluster_1.append(var.iloc[i])
        
cluster_2 = pandas.DataFrame(columns = var.columns.tolist())
for i in range(0,len(labels)):
    if labels[i] == 0:
        cluster_2 = cluster_2.append(var.iloc[i])
## describe clusters     
des1 = cluster_1.describe()    
des2 = cluster_2.describe()  
des3 = cluster_3.describe()