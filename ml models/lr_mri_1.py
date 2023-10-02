
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
# MODEL LOGISTIC REGRESSION
##############################################################################
##############################################################################
print('Starting 100 iteration in train and test.')
for i in range(0, 100):
    
    print('iteration:', i)
    
    df_cleaned, X_test, label, y_test = train_test_split(df_cleaned_tmp, label_temp1, test_size=0.35, stratify=label_temp1)
    if i==0:
        param_grid = [
            {'penalty' :  ['l1', 'l2'],
            'C' :  np.logspace(-4, 4, 20),
            'solver' : ['liblinear']}
        ]
        
        # Create grid search object
        clf = GridSearchCV(LogisticRegression(random_state=0), param_grid, cv = 4, verbose=0, n_jobs=-1)
        
        # Fit on data
        
        best_clf = clf.fit(df_cleaned, label)
        bestParamsLR=clf.best_params_
        
        model_LM = LogisticRegression(**bestParamsLR,random_state=1,max_iter=15000)
        model_LM.fit(df_cleaned, label)
        
    else:
        model_LM = LogisticRegression(**bestParamsLR,random_state=1,max_iter=15000)
        model_LM.fit(df_cleaned, label)     
    
    results_pred=model_LM.predict(X_test)

    LR_CF=confusion_matrix(y_test, results_pred)
    
    res_LR[0,i]=metrics.accuracy_score(y_test, results_pred)
    res_LR[1,i]=LR_CF[0,0]/(LR_CF[0,0]+LR_CF[0,1])
    res_LR[2,i]=LR_CF[1,1]/(LR_CF[1,1]+LR_CF[1,0])
    res_LR[3,i]=metrics.recall_score(y_test, results_pred)
    res_LR[4,i]=metrics.precision_score(y_test, results_pred)
    res_LR[5,i]=metrics.f1_score(y_test, results_pred)
    res_LR[6,i]=metrics.roc_auc_score(y_test, results_pred)
    res_LR[7,i]=metrics.matthews_corrcoef(y_test, results_pred)
    
    print('Logistic Regression')
    print('Accuracy: ',res_LR[0,i]*100,'%')
# %%
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

# %%
##############################################################################
# MODEL KNN
##############################################################################
##############################################################################

    if i==0:
        param_grid = {'n_neighbors': [1,2,3,4,5,7],
                      'weights': ['uniform','distance'],
                      'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}
        
        grid = GridSearchCV(KNeighborsClassifier(), param_grid,cv=4, verbose = 0,n_jobs=-1)
        grid.fit(df_cleaned, label)
        
        # print best parameter after tuning
        # print(grid.best_params_)
        best_paramKNN=grid.best_params_
        neigh=KNeighborsClassifier(**best_paramKNN)
        neigh.fit(df_cleaned, label)
    else:
        neigh=KNeighborsClassifier(**best_paramKNN)
        neigh.fit(df_cleaned, label)  
    
    results_pred=neigh.predict(X_test)
    KNN_CF=confusion_matrix(y_test, results_pred)
    
    
    res_KNN[0,i]=metrics.accuracy_score(y_test, results_pred)
    res_KNN[1,i]=KNN_CF[0,0]/(KNN_CF[0,0]+KNN_CF[0,1])
    res_KNN[2,i]=KNN_CF[1,1]/(KNN_CF[1,1]+KNN_CF[1,0])
    res_KNN[3,i]=metrics.recall_score(y_test, results_pred)
    res_KNN[4,i]=metrics.precision_score(y_test, results_pred)
    res_KNN[5,i]=metrics.f1_score(y_test, results_pred)
    res_KNN[6,i]=metrics.roc_auc_score(y_test, results_pred)
    res_KNN[7,i]=metrics.matthews_corrcoef(y_test, results_pred)

    print('K - nearest neighbour')
    print('Accuracy: ',res_KNN[0,i]*100,'%')

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

    #%% MODEL XGBOOST   
    # LOADING DATASET: ricarico un dataset diverso per tenere conto dei missing
    
    # df = pd.read_csv('/home/dati/00_DATA_ANALYSES/17_DATA_FETAL_CHD/01_ML/01_data/Dataset_999.csv')
    
    #  Feature Importance - and - Selection
    
    # model = XGBClassifier(random_state=1)
    # model.fit(df,label)
    # # feature importance
    # print(model.feature_importances_)
    
    # Best param
# {'colsample_bytree': 0.8,
#  'eta': 0.3,
#  'gamma': 0.2,
#  'learning_rate': 0.1,
#  'max_depth': 3,
#  'min_child_weight': 1,
#  'n_estimators': 100,
#  'reg_alpha': 0.5,
#  'reg_lambda': 2,
#  'subsample': 0.6,
#  'tree_method': 'approx'}
    
    # params = {
    #     'learning_rate': [0.1, 0.01, 0.001],  # Learning rate of the XGBClassifier
    #     'max_depth': [3, 5, 7, 9],  # Maximum depth of the decision trees in the XGBClassifier
    #     'n_estimators': [50, 100, 200],  # Number of decision trees in the XGBClassifier
    #     'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required to make a split in the XGBClassifier
    #     'subsample': [0.6, 0.8, 1.0],  # Fraction of observations to be randomly sampled for each tree in the XGBClassifier
    #     'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of columns to be randomly sampled for each tree in the XGBClassifier
    #     'reg_alpha': [0, 0.1, 0.5],  # L1 regularization term on weights of the XGBClassifier
    #     'reg_lambda': [1, 1.5, 2],  # L2 regularization term on weights of the XGBClassifier
    #     'min_child_weight': [1, 3, 5, 7],  # Minimum sum of instance weight (hessian) needed in a child in the XGBClassifier
    #     'eta':[.3, .2, .1, .05, .01, .005],
    # }
    if i==0:
        params = {
            'learning_rate': [0.1, 0.01],  # Learning rate of the XGBClassifier
            'max_depth': [3, 4],  # Maximum depth of the decision trees in the XGBClassifier
            'n_estimators':[int(x) for x in np.linspace(start = 100, stop = 1000, num = 5)],  # Number of decision trees in the XGBClassifier
            'gamma': [0.1, 0.2],  # Minimum loss reduction required to make a split in the XGBClassifier
            'subsample': [0.6, 0.8],  # Fraction of observations to be randomly sampled for each tree in the XGBClassifier
            'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of columns to be randomly sampled for each tree in the XGBClassifier
            'reg_alpha': [0.1, 0.5],  # L1 regularization term on weights of the XGBClassifier
            'reg_lambda': [1, 2],  # L2 regularization term on weights of the XGBClassifier
            'min_child_weight': [1,2],  # Minimum sum of instance weight (hessian) needed in a child in the XGBClassifier
            'eta':[.3, .2],
            'tree_method':['auto', 'exact', 'approx'],
        }
        
        # rnd=RandomizedSearchCV(XGBClassifier(random_state=1),params, scoring='accuracy', n_iter=300, cv=folds, verbose=1, random_state=1,n_jobs=-1)
        # rnd.fit(df,label)
        
        # print(rnd.best_params_)
        # #score=[]
        # #score.append({'bestEst':rnd.best_estimator_,'bestParams': rnd.best_params_,'bestscore': rnd.best_score_})
        # bestEst_Random=rnd.best_estimator_
        # bestParams_Random=rnd.best_params_
        # bestscore_Random=rnd.best_score_
        # #Una volta trovati i parametri migliori esegui un check sul vicinato
        # #GRIDSEARCH VICINATO
        # lr=bestParams_Random.get("learning_rate")
        # mcw=bestParams_Random.get("min_child_weight")
        # nestim=bestParams_Random.get("n_estimators")
        # gmma=bestParams_Random.get("gamma")
        # ssamp=bestParams_Random.get("subsample")
        # maxd=bestParams_Random.get("max_depth")
        # eta=bestParams_Random.get("eta")
        # #% Grid Search
        
        # paramet = {'n_estimators': [int(x) for x in np.linspace(start = (nestim-nestim*0.3), stop = (nestim+nestim*0.3), num = 4)],
        #               'min_child_weight': [mcw],                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        #               'learning_rate': [float(x) for x in np.linspace(start = (lr-lr*0.3), stop = (lr+lr*0.3), num = 4)],
        #               'gamma': [float(x) for x in np.linspace(start = (gmma-gmma*0.3), stop = (gmma+gmma*0.3), num = 4)], 
        #               'subsample': [float(x) for x in np.linspace(start = (ssamp-ssamp*0.3), stop = (ssamp+ssamp*0.3), num = 4)],
        #               'max_depth':[4,5,6],
        #               'eta':[float(x) for x in np.linspace(start = (eta-eta*0.5), stop = (eta+eta*0.5), num = 4)],
        #               'objective':['binary:logistic'],
        #               'tree_method':['hist'],
        #               'eval_metric':['logloss']}#[int(x) for x in np.linspace(start = (maxd-maxd*0.3), stop = (maxd+maxd*0.3), num = 5)]}
        # #GridSearchCV & Fitting
        # print()
        # print('Start of GridSearchCV and fitting')
        
        xgb_def=XGBClassifier(random_state=1,objective='binary:logistic')
        grid=GridSearchCV(estimator=xgb_def, param_grid=params, scoring='accuracy', cv=4, verbose=1, n_jobs=-1, refit=True)
        grid.fit(df_cleaned, label)
        
        #BestEst, score & rmse
        xgb_bestE=grid.best_estimator_
        xgb_bestP=grid.best_params_
        best_score=grid.best_score_  
        cv_results=grid.cv_results_
        
        xgb_calss=XGBClassifier(**xgb_bestP,random_state=1,objective='binary:logistic')
        
        xgb_calss.fit(df_cleaned, label)
    else:
        xgb_calss=XGBClassifier(**xgb_bestP,random_state=1,objective='binary:logistic')
        xgb_calss.fit(df_cleaned, label)

    results_pred=xgb_calss.predict(X_test)
    XGboost_CF=confusion_matrix(y_test, results_pred)
 
    res_XGB[0,i]=metrics.accuracy_score(y_test, results_pred)
    res_XGB[1,i]=XGboost_CF[0,0]/(XGboost_CF[0,0]+XGboost_CF[0,1])
    res_XGB[2,i]=XGboost_CF[1,1]/(XGboost_CF[1,1]+XGboost_CF[1,0])
    res_XGB[3,i]=metrics.recall_score(y_test, results_pred)
    res_XGB[4,i]=metrics.precision_score(y_test, results_pred)
    res_XGB[5,i]=metrics.f1_score(y_test, results_pred)
    res_XGB[6,i]=metrics.roc_auc_score(y_test, results_pred)
    res_XGB[7,i]=metrics.matthews_corrcoef(y_test, results_pred)

    print('XG boost')
    print('Accuracy: ',res_XGB[0,i]*100,'%')

RES_TOT[:,0]=np.mean(res_LR,axis=1)
RES_TOT[:,1]=np.mean(res_SVC,axis=1)
RES_TOT[:,2]=np.mean(res_KNN,axis=1)
RES_TOT[:,3]=np.mean(res_RF,axis=1)
RES_TOT[:,4]=np.mean(res_XGB,axis=1)

#  Boxplot variabili

data = df_cleaned_tmp.to_numpy()
df_cleaned_tmp['new_column'] = label_temp1.tolist()
df_cleaned_tmp=df_cleaned_tmp.rename(columns = {'new_column':'Label'})
df_cleaned_tmp.boxplot(by='Label',figsize=(12,12))



