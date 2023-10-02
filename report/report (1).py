# @author:utkarshraj01
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.preprocessing as PreProc
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
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
import shap

##############################################################################
##############################################################################
##############################################################################

# LOADING DATASET

df = pd.read_csv('example.csv')
df_label = pd.read_csv('example.csv')
label_temp=df_label['SEX']
label_temp1=pd.DataFrame(label_temp)
label_temp1=label_temp1.values.ravel()

df=df.select_dtypes(['number'])
df=df.drop(['Var1'],axis=1)

colname=df.columns

df_dc = df.drop(['Scanner','ABBel32','DRUGS','Mot_dise','SEX'],axis=1)
df_dummy=df[['Scanner','ABBel32','DRUGS','Mot_dise','SEX']]

##############################################################################
##############################################################################
    
# Data Normalization Z-score
df_cleaned_tmp_norm = preprocessing.StandardScaler().fit(df_dc).transform(df_dc)

df_cleaned_tmp_norm=pd.DataFrame(df_cleaned_tmp_norm,columns=(df_dc.columns))
df_cleaned_tmp1=df_cleaned_tmp_norm

df_cleaned_tmp = pd.concat([df_dummy,df_cleaned_tmp1], axis=1)

##############################################################################

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
# MODEL LOGISTIC REGRESSION USING SHAP
##############################################################################
##############################################################################
explainer = shap.Explainer(model_LM, df_cleaned_tmp)  # Initialize the SHAP explainer

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
    res_LR[4, i] = metrics.precision_score(y_test, results_pred, zero_division=1)
    res_LR[5,i]=metrics.f1_score(y_test, results_pred)
    res_LR[6,i]=metrics.roc_auc_score(y_test, results_pred)
    res_LR[7,i]=metrics.matthews_corrcoef(y_test, results_pred)
    
    print('Logistic Regression')
    print('Accuracy: ',res_LR[0,i]*100,'%')
    
    shap_values = explainer.shap_values(X_test)  # Calculate SHAP values for the test data
    feature_importances = np.abs(shap_values).mean(axis=0)
    for feature, importance in zip(df_cleaned_tmp.columns, feature_importances):
        print(f"Feature: {feature}, Importance: {importance}")

##accuracy
##LOGISTIC REGRESSION
##Accuracy:  70.0 %
##Average Accuracy:  61.3 %



##############################################################################
# MODEL SVC SHAP
##############################################################################
##############################################################################


print('Starting 100 iteration in train and test.')
accuracy_list = []  # List to store accuracy values
shap_values_list = []  # List to store SHAP values

for i in range(0, 100):
    print('iteration:', i)
    
    df_cleaned, X_test, label, y_test = train_test_split(df_cleaned_tmp, label_temp1, test_size=0.35, stratify=label_temp1)
    
    if i == 0:
        param_grid = {'C': [0.01,0.02,0.05,0.08,0.1,0.2,0.4,0.5,0.7,0.6,0.8,1,2,4,6,8,10],
                      'gamma': [0.08,0.1,0.2,0.5,0.6,0.7,0.8,1,0.01,0.03,0.05,0.08, 0.001, 0.0001],
                      'kernel': ['rbf','linear','poly','rbf','sigmoid']}
        
        SVM_model = svm.SVC(random_state=0, probability=True)
        grid = GridSearchCV(SVM_model, param_grid, cv=4, refit=True, verbose=0)
        
        # fitting the model for grid search: MICE (Good performance)
        grid.fit(df_cleaned, label)
        
        # print best parameter after tuning
        # print(grid.best_params_)
        best_paramSVC = grid.best_params_
        SVM_model = svm.SVC(**best_paramSVC, random_state=0, probability=True)
        SVM_model.fit(df_cleaned, label)
    else:
        SVM_model = svm.SVC(**best_paramSVC, random_state=0, probability=True)
        SVM_model.fit(df_cleaned, label)
        
    results_pred = SVM_model.predict(X_test)
    SCV_CF = confusion_matrix(y_test, results_pred)
    
    accuracy = metrics.accuracy_score(y_test, results_pred)
    accuracy_list.append(accuracy)  # Store accuracy in the list
    
    # Calculate SHAP values
    explainer = shap.KernelExplainer(SVM_model.predict_proba, df_cleaned)
    shap_values = explainer.shap_values(X_test)
    shap_values_list.append(shap_values)
    
    print('Support Vector Classification')
    print('Accuracy: ', accuracy * 100, '%')

# Calculate and print the average accuracy
average_accuracy = sum(accuracy_list) / len(accuracy_list)
print('Average Accuracy: ', average_accuracy * 100, '%')

# Calculate average SHAP values
average_shap_values = np.mean(shap_values_list, axis=0)

# Select features with non-zero importance
selected_feature_indices = np.where(feature_importances > 0)[0]
selected_features = df_cleaned.columns[selected_feature_indices]

# Filter the dataset to include only selected features
df_cleaned_selected = df_cleaned.iloc[:, selected_feature_indices]

# Print the content of the new dataset
print('Selected Features:')
print(df_cleaned_selected.head())


##accuracy
##Support Vector Classification
##Accuracy:  75.0 %
##Average Accuracy:  62.5 %

###################################################################################33
##MODEL SVC FCLSSIF
########################################################################################################

import shap
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif

print('Starting 100 iteration in train and test.')
accuracy_list = []  # List to store accuracy values
shap_values_list = []  # List to store SHAP values

for i in range(0, 2):
    print('iteration:', i)

    df_cleaned, X_test, label, y_test = train_test_split(df_cleaned_tmp, label_temp1, test_size=0.35, stratify=label_temp1)

    if i == 0:
        param_grid = {'C': [0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.4, 0.5, 0.7, 0.6, 0.8, 1, 2, 4, 6, 8, 10],
                      'gamma': [0.08, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 1, 0.01, 0.03, 0.05, 0.08, 0.001, 0.0001],
                      'kernel': ['rbf', 'linear', 'poly', 'rbf', 'sigmoid']}

        SVM_model = svm.SVC(random_state=0, probability=True)
        grid = GridSearchCV(SVM_model, param_grid, cv=4, refit=True, verbose=0)

        # fitting the model for grid search: MICE (Good performance)
        grid.fit(df_cleaned, label)

        # print best parameter after tuning
        # print(grid.best_params_)
        best_paramSVC = grid.best_params_
        SVM_model = svm.SVC(**best_paramSVC, random_state=0, probability=True)
        SVM_model.fit(df_cleaned, label)
    else:
        SVM_model = svm.SVC(**best_paramSVC, random_state=0, probability=True)
        SVM_model.fit(df_cleaned, label)

    results_pred = SVM_model.predict(X_test)
    SVC_CF = confusion_matrix(y_test, results_pred)  # Calculate confusion matrix

    accuracy = metrics.accuracy_score(y_test, results_pred)
    accuracy_list.append(accuracy)  # Store accuracy in the list

    # Calculate SHAP values
    explainer = shap.KernelExplainer(SVM_model.predict_proba, df_cleaned)
    shap_values = explainer.shap_values(X_test)
    shap_values_list.append(shap_values)

    print('Support Vector Classification')
    print('Accuracy:', accuracy * 100, '%')

# Calculate and print the average accuracy
average_accuracy = sum(accuracy_list) / len(accuracy_list)
print('Average Accuracy:', average_accuracy * 100, '%')

# Calculate average SHAP values
average_shap_values = np.mean(shap_values_list, axis=0)

#dataset
# Perform feature selection using f_classif
f_scores, p_values = f_classif(df_cleaned_tmp, label_temp1)

# Select features based on significance (p-value)
selected_features = df_cleaned_tmp.columns[p_values < 0.05]

# Filter the dataset to include only selected features
df_cleaned_selected = df_cleaned_tmp[selected_features]

# Print the content of the new dataset
print('Selected Features:')
print(df_cleaned_selected.head())


#####################################
#   Support Vector Classification
#   Accuracy: 66.66666666666666 %
#   Average Accuracy: 66.66666666666666 %


##################################3



##############################################################################
# MODEL KNN SHAP
##############################################################################
##############################################################################
from sklearn.feature_selection import SelectKBest, f_classif

model1 = RandomForestClassifier(random_state=1)
print('Starting 100 iteration in train and test.')
accuracy_list = []  # List to store accuracy values

for i in range(0, 100):
    print('iteration:', i)
    
    df_cleaned, X_test, label, y_test = train_test_split(df_cleaned_tmp, label_temp1, test_size=0.35, stratify=label_temp1)
    
    if i == 0:
        criterion = ['gini', 'entropy', 'log_loss']
        n_estimators = [int(x) for x in np.linspace(start=80, stop=500, num=4)]
        max_features = ['sqrt', 'log2']
        max_depth = [int(x) for x in np.linspace(2, 6, num=5)]
        min_samples_split = [2, 3]
        min_samples_leaf = [2, 4]
        bootstrap = [True, False]
        
        params = {'criterion': criterion,
                  'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap}

        grid = GridSearchCV(model1, params, cv=4, verbose=1, n_jobs=-1)
        grid.fit(df_cleaned, label)
        best_paramRF = grid.best_params_
        
        model = RandomForestClassifier(**best_paramRF, random_state=1)
        model.fit(df_cleaned, label)
    else:
        model = RandomForestClassifier(**best_paramRF, random_state=1)
        model.fit(df_cleaned, label)
    
    # Perform F-test feature selection
    kbest = SelectKBest(score_func=f_classif, k=10)  # Select top 10 features (change k value as needed)
    kbest.fit(df_cleaned, label)
    selected_features = df_cleaned.columns[kbest.get_support()]
    df_cleaned = df_cleaned[selected_features]
    X_test = X_test[selected_features]

    results_pred = model.predict(X_test)
    RF_CF = confusion_matrix(y_test, results_pred)
 
    accuracy = metrics.accuracy_score(y_test, results_pred)
    accuracy_list.append(accuracy)
    res_RF[0, i] = accuracy
    res_RF[1, i] = RF_CF[0, 0] / (RF_CF[0, 0] + RF_CF[0, 1])
    res_RF[2, i] = RF_CF[1, 1] / (RF_CF[1, 1] + RF_CF[1, 0])
    res_RF[3, i] = metrics.recall_score(y_test, results_pred)
    res_LR[4, i] = metrics.precision_score(y_test, results_pred, zero_division=1)
    res_RF[5, i] = metrics.f1_score(y_test, results_pred)
    res_RF[6, i] = metrics.roc_auc_score(y_test, results_pred)
    res_RF[7, i] = metrics.matthews_corrcoef(y_test, results_pred)

    print('Random Forest')
    print('Accuracy:', accuracy * 100, '%')

# Calculate average accuracy
average_accuracy = np.mean(accuracy_list)
print('Average Accuracy:', average_accuracy * 100, '%')


#################
#### Random Forest
#### Accuracy: 50.0 %
#### Average Accuracy: 71.58333333333333 % 
##################