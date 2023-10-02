
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

##############################################################################
# MODEL SVC
##############################################################################
##############################################################################
    if i==0:
        param_grid = {'C': [0.01,0.02,0.05,0.08,0.1,0.2,0.4,0.5,0.7,0.6,0.8,1,2,4,6,8,10],
                      'gamma': [0.08,0.1,0.2,0.5,0.6,0.7,0.8,1,0.01,0.03,0.05,0.08, 0.001, 0.0001],
                      'kernel': ['rbf','linear','poly','rbf','sigmoid']}
        
        SVM_model = svm.SVC(random_state=0)
        grid = GridSearchCV(SVM_model, param_grid,cv=4, refit = True, verbose = 0)
        
        # fitting the model for grid search: MICE (Good performance)
        grid.fit(df_cleaned, label)
        
        # print best parameter after tuning
        # print(grid.best_params_)
        best_paramSVC=grid.best_params_
        SVM_model=svm.SVC(**best_paramSVC, random_state=0)
        SVM_model.fit(df_cleaned, label)
    else:
        SVM_model=svm.SVC(**best_paramSVC, random_state=0)
        SVM_model.fit(df_cleaned, label)
        
    results_pred=SVM_model.predict(X_test)
    SCV_CF=confusion_matrix(y_test, results_pred)
    
    res_SVC[0,i]=metrics.accuracy_score(y_test, results_pred)
    res_SVC[1,i]=SCV_CF[0,0]/(SCV_CF[0,0]+SCV_CF[0,1])
    res_SVC[2,i]=SCV_CF[1,1]/(SCV_CF[1,1]+SCV_CF[1,0])
    res_SVC[3,i]=metrics.recall_score(y_test, results_pred)
    res_SVC[4,i]=metrics.precision_score( y_test, results_pred)
    res_SVC[5,i]=metrics.f1_score( y_test, results_pred)
    res_SVC[6,i]=metrics.roc_auc_score( y_test, results_pred)
    res_SVC[7,i]=metrics.matthews_corrcoef(y_test, results_pred)

    print('Support Vector Classification')
    print('Accuracy: ',res_SVC[0,i]*100,'%')

