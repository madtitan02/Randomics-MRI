
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.preprocessing as PreProc
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
# from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, matthews_corrcoef  #variance regression score function
from time import time
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from operator import xor
import pdb
import os
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, chi2, f_classif, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import itertools
# from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, precision_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics, feature_selection
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from skopt.space import Real, Categorical, Integer
from sklearn import preprocessing
from xgboost import plot_importance
from matplotlib import pyplot
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from featboost import FeatBoostClassifier

##############################################################################
##############################################################################

# LOADING DATASET

df = pd.read_csv('example.csv')
df_label = pd.read_csv('example1.csv')
label_temp=df_label['CHD_CNT_CODE']
label_temp1=pd.DataFrame(label_temp)
label_temp1=label_temp1.values.ravel()

df=df.select_dtypes(['number'])
df=df.drop(['Unnamed: 0'],axis=1)

colname=df.columns

df_dc = df.drop(['Scanner','Above_Below_32GW','DRUGS','MOTHER_DISEASE','SEX'],axis=1)
df_dummy=df[['Scanner','Above_Below_32GW','DRUGS','MOTHER_DISEASE','SEX']]
##############################################################################
##############################################################################
    
# Data Normalization Z-score

df_cleaned_tmp_norm = preprocessing.StandardScaler().fit(df_dc).transform(df_dc)

df_cleaned_tmp_norm=pd.DataFrame(df_cleaned_tmp_norm,columns=(df_dc.columns))
df_cleaned_tmp1=df_cleaned_tmp_norm

df_cleaned_tmp = pd.concat([df_dummy,df_cleaned_tmp1], axis=1)

##############################################################################
##############################################################################
 
# Variable Initialization

folds = LeaveOneOut()

res_LR=np.zeros((8,100))
res_SVC=np.zeros((8,100))
res_KNN=np.zeros((8,100))
res_RF=np.zeros((8,100))
res_XGB=np.zeros((8,100))
RES_TOT=np.zeros((8,5))

# %%
##############################################################################

    #%% MODEL RF  
    # Best param
# {'bootstrap': True,
#  'criterion': 'entropy',
#  'max_depth': 2,
#  'max_features': 'sqrt',
#  'min_samples_leaf': 2,
#  'min_samples_split': 2,
#  'n_estimators': 100}
   
    model1 = RandomForestClassifier(random_state=1)
    # size=len(name)
    if i==0:
        
        # rf=model1
        # rfe = RFECV(rf,cv=5,scoring="neg_mean_squared_error")
        # rfe.fit(df1, label)
        # selected_features = np.array(colname)[rfe.get_support()]
        # df_cleaned=df_cleaned[selected_features]
        
    # Number of trees in random forest
        criterion=['gini', 'entropy','log_loss']
        n_estimators = [int(x) for x in np.linspace(start = 80, stop = 500, num = 4)]
        # Number of features to consider at every split
        max_features = ['sqrt','log2']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(2, 6, num = 5)]
        #max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2,3]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [2,4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        
        # Create the random grid
        params = {'criterion':criterion,
                        'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}

        Grid=GridSearchCV(model1, params, cv=4, verbose=1,n_jobs=-1)
        Grid.fit(df_cleaned,label)
        bestEst=Grid.best_estimator_
        bestParamRF=Grid.best_params_
        
        model = RandomForestClassifier(**bestParamRF,random_state=1)
        # Fit on training data
        model.fit(df_cleaned, label)
    else:
        model = RandomForestClassifier(**bestParamRF,random_state=1)
        # Fit on training data
        model.fit(df_cleaned, label)

    results_pred=model.predict(X_test)
    RF_CF=confusion_matrix(y_test, results_pred)
 
    res_RF[0,i]=metrics.accuracy_score(y_test, results_pred)
    res_RF[1,i]=RF_CF[0,0]/(RF_CF[0,0]+RF_CF[0,1])
    res_RF[2,i]=RF_CF[1,1]/(RF_CF[1,1]+RF_CF[1,0])
    res_RF[3,i]=metrics.recall_score(y_test, results_pred)
    res_RF[4,i]=metrics.precision_score(y_test, results_pred)
    res_RF[5,i]=metrics.f1_score(y_test, results_pred)
    res_RF[6,i]=metrics.roc_auc_score(y_test, results_pred)
    res_RF[7,i]=metrics.matthews_corrcoef(y_test, results_pred)

    print('Random Forest')
    print('Accuracy: ',res_RF[0,i]*100,'%')

  