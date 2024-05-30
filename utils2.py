#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import scikitplot as skplt

from numpy import percentile, argmax, sqrt
from scipy.stats import norm
from scipy import stats
import collections 
from collections import Counter
from prettytable import PrettyTable

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, OneClassSVM
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,roc_curve, auc, roc_auc_score, precision_recall_curve,average_precision_score,make_scorer
from sklearn.feature_selection import f_classif, chi2, SelectKBest

from xgboost import XGBClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from imblearn.pipeline import Pipeline as imbpipeline

import gc
import warnings

import csv
import os
import datetime

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', None) #show all columns in dataframe
pd.set_option('float_format', '{:.3f}'.format)
pd.set_option('display.show_dimensions', True)


# In[27]:

#__all__ = ["data_clean", "split_data", "preprocessing", "counter","classification_models", "sampler","predictions" ]


def data_clean(data, To=None, upcoming_month=None):
    """
    Cleans and prepares data for further use. If used for predicting upcoming month some features like 'To' will not be in 
    the data, so function can pass editting those features.
    Arguments:
    data -- data
    to -- column "To", should be a list: (["To"] or ["To1", "To2"])
    upcoming_month -- if 'no' edit all features
    
    Returns:
    data -- cleaned data
    """
    """#drop column containing same value for all rows
    data.drop('TP_logic', axis=1, inplace=True) 

    #combine net types containing "unknown" into "other" category
    data.replace(to_replace = data[data['Net Type Change'].str.contains("Unknown")]['Net Type Change'].values, value = 'Other', inplace = True)
    
    #combine net types with no change into "No change" category
    data['Net Type Change'].replace({'4G>>4G':'No change','3G>>3G':'No change', '2G>>2G':'No change','3G>>2.5G': 'Other'}, inplace = True)
    
    #combine device types with no change into "No change" category
    data['Device Type Change'].replace({'Smart>>Smart':'No change', 'Regular>>Regular':'No change'}, inplace = True)"""
    
    #set "MSISDN" as an index
    data.set_index('MSISDN', inplace = True)

    """#set all values bigger than 3 equal to 3
    data['TP Activations'].values[data['TP Activations'] > 3] = 3
    
    #map values of "Device change"
    data['Device Change'] = data['Device Change'].map({'Yes': 1, 'No': 0})
    
    #set "VnD Days" values which are more than 31 equal to 31
    if (data.loc[:, data.columns.str.endswith('_VnD_Days')].values > 31).any():
        data.loc[:, data.columns.str.endswith('_VnD_Days')] = data.filter(regex = "_VnD_Days$", axis = 1).apply(lambda x: [i if i <= 31 else 31 for i in x])
    else:
        pass"""
    
    if upcoming_month == 'no':
        for to in To:
            #replace tariff plan names, which were written wrongly
            data[to].replace({'City' : 'Viva: Viva 5500',
                'City 2': 'Viva: Viva 3500',
                'City 3': 'Viva: Viva 2500',
                'City Child': 'Viva: Viva 2500',
                'Andsnakan': 'Viva: Viva 7500',
                'New Youth Y': 'New Youth: Y',
                'New Youth X': 'New Youth: X',
                'Viva 9500': 'Viva: Viva 9500',
                'Viva: Z': 'New Youth: Z',
                'Viva: Y': 'New Youth: Y',
                'New Youth Y: Y': 'New Youth: Y',
                'New Youth X: X': 'New Youth: X',
                'New Youth Z: Z': 'New Youth: Z'}, inplace = True)
            #drop tariff plans, which names don't give enough information to identify tariff plan
            data = data.loc[data[to].isin(data.From)]
            #data.drop(data.loc[data.To.isin(['Viva', 'New Youth'])].index, inplace=True)
    else:
        pass
        
    return data


def split_data(tariff_plan, data, tag, keep, list_col):
    """
    This function splits data into X and y and removes unnecessary features for inputed tariff plan
    Arguments:
    tariff_plan -- the name of tariff plan we want to work with
    data -- data    
    tag -- target column in the data 
    keep -- binarized target column that should be kept ("Target_Upsell" or "Target_Downsell")
    list_col -- list of columns that should be dropped from X (['From', 'To', 'Region', 'Days_Not_Active', 'change'])
    
    Returns:
    data -- data of tariff plan before binarizing target column and removing features
    new_df -- transformed data
    X -- the feature columns
    y -- the target column
    """
    data = data[data['From'] == tariff_plan]
    features = list(data.loc[:, data.columns != tag].columns) #name of features for X
    df = pd.get_dummies(data = data, columns = [tag]) #dummifying target column
    X = df[features].drop(list_col, axis = 1)
    y = df[keep]
    new_df = pd.concat([X,y],axis=1)
    return data, new_df, X, y

def load_saved_model(location,model_name):
    """
    Loads the saved models
    Arguments:
    location -- the directory of the saved model
    model_name -- name of the model saved 

    Returns: 
    model -- the saved model
    """
    model=pickle.load(open(location+model_name+'.sav','rb'))
    return(model)



def preprocessing(X):
    """
    This function transforms data for being used later in the pipeline. Steps include encoding categorical
    variables, imputing missing values and feature scaling. Some models don't require feature scaling or one hot encoding, 
    so there is are options in function to do data preprocessing without scaling and one hot encoding.
    Arguments:
    X -- the feature columns
    
    Returns:
    preprocessor_scaled_imp_enc -- transforming both categorical and numerical features
    preprocessor_imp_enc - transforming only categorical features by imputing missing values and one hot encoding
    preprocessor_imp -- transforming only categorical features by only imputing missing values
    """
    
    #Transformer type
    categorical_transformer_with_enc = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing') ), #imputing missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))]) #transforming the categorical values into integers
    
    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))]) #imputing missing values

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
      
    #Feature types
    categorical_features = X.select_dtypes(include=['object']).columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

    preprocessor_scaled_imp_enc = ColumnTransformer(transformers=[   #(name, transformer, columns)
        ('cat', categorical_transformer_with_enc, categorical_features), # Transformer for categorical variables
        ('num', numeric_transformer, numeric_features)])       # Transformer for numeric variables     
        
    preprocessor_imp_enc = ColumnTransformer(transformers=[  #(name, transformer, columns)
        ('cat', categorical_transformer_with_enc, categorical_features)], remainder = 'passthrough')
    
    preprocessor_imp = ColumnTransformer(transformers=[  #(name, transformer, columns)
        ('cat', categorical_transformer, categorical_features)], remainder = 'passthrough')

    return preprocessor_scaled_imp_enc, preprocessor_imp_enc, preprocessor_imp



def counter(y):
    """
    Gives necessary estimates for being used as parameters in the models
    Arguments:
    y -- target column
    
    Returns:
    estimate -- majority class divided by minority class
    minority_prop -- minority class devided by majority class"""
    
    counter = Counter(y)
    estimate = counter[0] / counter[1]
    minority_prop = counter[1]/ counter[0]
    return(estimate, minority_prop)


def clf_models(estimate, minority_prop):
    """Returns classification models with their hyperparameters for GridSearchCV """
    
    classification_models={'RF':RandomForestClassifier(random_state=42),
                           'RF_param_grid': { #'max_features': ['auto', 'sqrt', 'log2'],
                                              'n_estimators': [100,300],
                                              'max_depth' : [5,9,13,15,19],
                                              #'criterion' :['gini', 'entropy'],
                                              'class_weight':['balanced','balanced_subsample']},
                           'RF_name':'Random Forest',

                           'LR':LogisticRegression(random_state=42),
                           'LR_param_grid':{'penalty' : ['l2'],#,'none'],
                                            'C':np.logspace(-2, 4, 20),
                                            'solver': ['liblinear'],
                                            'class_weight':['balanced','balanced_subsample']},
                           'LR_name':'Logistic Regression', 

                           'SVC':SVC(random_state=42),
                           'SVC_param_grid':{'C':np.logspace(-4, 4, 10),
                                             'gamma':[1,0.1,0.001,0.0001],
                                             'kernel':['poly', 'rbf', 'sigmoid'],
                                             'class_weight':['balanced', {0:1, 1: 90}]
                                            },
                           'SVC_name':'SVC',

                           'KNN':KNeighborsClassifier(),
                           'KNN_param_grid':{'n_neighbors':[3,5,9,13,15,20],
                                             'weights':['uniform','distance'],
                                             #'algorithm':['auto','ball_tree','kd_tree','brute'],
                                             'leaf_size':[10,20,30,40,50],
                                             'p':[1,2,3]},
                           'KNN_name':'K Neighbors',


                           'XGB': XGBClassifier(scale_pos_weight = estimate),
                           'XGB_param_grid': {'max_depth' : [3, 5, 7],
                                              'min_child_weight' : [ 4, 7],
                                              #'reg_lambda': [0.75,1,1.25],
                                              'gamma': [1], #[0.5, 1]
                                              'subsample': [0.5, 0.7, 1.0],
                                              'n_estimators': [100,300,500],
                                              'colsample_bytree': [0.6, 0.8]},
                                              #'n_estimators': [100, 300]},
                           'XGB_name': 'XGBoost with scale_pos_weight',


                           'ADA': AdaBoostClassifier(random_state = 22),
                           'ADA_param_grid': {'n_estimators': [100, 300],
                                             'learning_rate' : [0.01,0.1,0.3]},
                           'ADA_name': 'AdaBoost',

                          #Algorithms for anomaly detection

                           'OneClassSVM': OneClassSVM(nu = minority_prop),
                           'OCSVM_param_grid':{'gamma':[1,0.1,0.001,0.0001],
                                             'kernel':['poly', 'rbf', 'sigmoid']},
                                              #'nu':[outlier_prop, 0.25, 0.5, 0.75, 0.9]},
                           'OneClassSVM_name': 'OneClassSVM',


                           'IF': IsolationForest(contamination = minority_prop),
                           'IF_param_grid': {'n_estimators':[200, 500]},
                           'IF_name': 'IsolationForest'}
    return classification_models


def sampler(minority_ratio):
    """
    Returns dictionary of different sampling methods, which are used for resampling imbalanced data
    Arguments:
    minority ratio -- ratio of minority class to majority class
    
    Returns:
    sampler -- dictionary with sampling methods"""
    
    sampler ={'random_oversampler':RandomOverSampler(sampling_strategy = minority_ratio, random_state=2),
           'smote_oversampler': SMOTE(sampling_strategy = minority_ratio, random_state = 22),
           'random_undersampler': RandomUnderSampler(sampling_strategy = minority_ratio, random_state = 22)}
    return (sampler)


"""Dictionary of scoring values for outlier/anomaly detecting models"""
scoring_out = {'ROC_AUC': 'roc_auc', 'F1': make_scorer(f1_score,pos_label=-1), 
              'Recall':make_scorer(recall_score,pos_label=-1), 
              'Precision':make_scorer(precision_score,pos_label=-1), 
              'Specificity':make_scorer(recall_score,pos_label=1)}

"""Dictionary of scoring values for other models"""
scoring = {'ROC_AUC': 'roc_auc', 'F1': 'f1', 'Recall':'recall', 'Precision':'precision',
           'Specificity': make_scorer(recall_score,pos_label=0)}


def create_model(X, y, cv, model, param, preprocessor, scoring, refit, sampler_method = 'no', min_ratio=None, algorithm= 'normal', save_model= None, saved_model_name= None):
    """
    Returns the model after gridsearchcv and prints the best parameters for that model for the given score 
    
    Arguments:
    X -- the feature columns
    y -- the target columns
    cv -- how many batches the data are divided, generally either 3 or 5
    model -- classification models
    param -- dict of parameters for the model
    preprocessor -- part of the pipeline for preprocessing data
    scoring -- one of the scoring dictionaries 
    refit -- estimator which is used to find the best parameters 
    sampler_method -- choosing one of the sampling methods or nothing
    min_ratio -- ratio of minority class if sampler_method is  used
    algorithm -- type of model: "outliers" for outlier/anomaly detecting models, "normal" for others
    save_model -- yes to save None for don't 
    saved_model_name -- name of the model saved

    Returns: 
    cv_model -- the fitted model
    """
    name = saved_model_name
    new_params = {'clf__' + key: param[key] for key in param}
    
    #pipeline with resampling technique
    if sampler_method in sampler(min_ratio).keys():
        pipeline = imbpipeline([('preprocessing', preprocessor),('sampling', sampler(min_ratio)[sampler_method]), ('clf', model)])    
    else:     #pipeline without resampling technique
        pipeline = imbpipeline([('preprocessing', preprocessor), ('clf', model)]) 
        
    cv_model = GridSearchCV(pipeline, param_grid=new_params,cv=cv, scoring=scoring, refit=refit, return_train_score=True, verbose=4, n_jobs=2, pre_dispatch = '2*n_jobs')
    
    if algorithm == 'normal':
        cv_model.fit(X, y)
    elif algorithm == 'outliers':
        yd = np.where(y==0,1,-1)
        cv_model.fit(X, yd)
        
    print('\033[1m'+'Best parameters: '+'\033[0m',cv_model.best_params_)
    print('\033[1m'+'Best score (highest %s score): '%(refit) +'\033[0m',round(cv_model.best_score_,4))
    print('\033[1m'+'Results for the best score'+'\033[0m')
    for k in cv_model.cv_results_:
        if k.startswith('mean_t'):
            print(k, round(cv_model.cv_results_[k][cv_model.best_index_], 4))
    if save_model=='yes':
        pickle.dump(cv_model,open(name,'wb'))
    else:
        print('model is not saved')
    return(cv_model)


def models_performance_cv(clfs):
    """
    Returns dataframe of fitted models and their scores.
    Arguments:
    clfs -- list of classification models
    
    Returns:
    fin_df -- dataframe with scores
    """
    
    model_scores = {}
    for model in clfs:
        model_name = model.best_estimator_['clf'].__class__.__name__
        model_scores[model_name] = {}
        scores_ = {}
        for k in model.cv_results_:
            if k.startswith('mean_t'):
                scores_[k] = round(model.cv_results_[k][model.best_index_],4)
        model_scores[model_name] = scores_
    fin_df = pd.DataFrame(model_scores).T.sort_values(by='mean_test_ROC_AUC',ascending=True )
    return fin_df



def voting_classifier(clfs, X, y, voting_type, cv_num, n_jobs, save_model=None):
    """
    Function for a voting classifier which is a model that trains on an ensemble of numerous models and 
    predicts an output (class) based on their highest probability(in case of soft classifier) or highest majority of vote
    (in case of hard voting).
    Arguments:
    clfs -- list of classification models
    X -- features
    y -- target
    voting_type -- voting type: soft or hard
    
    Returns:
    scores -- score of voting classifier
    voting_clf -- fitted model
    """
    named_estimators = [] #group/ensemble of models
    for clf in clfs:
        named_estimators.append((clf.best_estimator_['clf'].__class__.__name__, clf.best_estimator_))
    voting_clf = VotingClassifier(named_estimators, voting=voting_type)
    cv = StratifiedKFold(cv_num, shuffle=True, random_state=22)
    scores = cross_val_score(voting_clf, X, y, scoring='roc_auc', cv=cv, n_jobs=n_jobs)
    if save_model=='yes':
        pickle.dump(voting_clf,open('voting_model','wb'))
    else:
        print('model is not saved')
    return scores, voting_clf




def plot_feature_importance(cv, top_n, X, asc):
    """
    Plots most and least important features according to different models
    
    Arguments:
    cv -- pipeline including the best model
    top_n -- number of features
    X -- the feature columns
    asc -- 'True' for least important features, 'False' for most important features
    """
    col_names = list(cv.best_estimator_['preprocessing'].transformers_[0][1]['onehot'].get_feature_names(X.select_dtypes(include=['object']).columns)) + list(X.select_dtypes(include=['int64', 'float64']).columns)
    feat_imp = pd.DataFrame({'importance':cv.best_estimator_['clf'].feature_importances_, 'features': col_names})
    feat_imp.sort_values(by='importance', ascending=asc, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]
    feat_imp.set_index('features', drop=True)
    plt.figure(figsize=(10,5))
    plt.barh(feat_imp.features, feat_imp.importance, height=0.5)
    plt.gca().invert_yaxis()
    plt.title(cv.best_estimator_['clf'].__class__.__name__)
    plt.show()



def feat_imp (score, X, y, num, asc): 
    """
    Plots feature importance according to univariate measures such as Chi-Squared(chi2()) and ANOVA(f_classif())
    
    Arguments:
    score -- f_classif or chi2
    X -- feature columns
    y -- target columns
    num -- number of features
    asc -- 'True' for least important features, 'False' for most important features
    """
    X = pd.get_dummies(data = X, columns = X.select_dtypes(include = 'object').columns)
    fs = SelectKBest(score_func=score)
    X_selected = fs.fit_transform(X, y)     # applying feature selection
    dfscores = pd.DataFrame(fs.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    fsv = featureScores.sort_values(by = 'Score', ascending = asc)[:num]
    plt.figure(figsize=(8,10))
    plt.barh(fsv.Specs, fsv.Score,  align='center') 
    plt.title("Feature importance according to %s" %(score.__name__), fontsize=14)
    plt.xlabel("Feature importance")
    plt.margins(y=0.01)
    plt.gca().invert_yaxis()
    plt.show()



def p_values(score, X, y, num):
    """
    This function returns p values of features
    Arguments:
    score -- f_classif or chi2
    X -- features
    y -- target
    num -- how many features to show
    
    Returns:
    p_df -- table with highest p values for both ANOVA and Chi-squared scores
    """
    X = pd.get_dummies(data = X, columns = X.select_dtypes(include = 'object').columns)
    scores = score(X, y)
    p_values = pd.DataFrame(scores[1],columns = ['p_value'],index = X.columns)
    p_df = p_values.sort_values(by ='p_value',ascending = False)[:num]
    return p_df 

#############################################################################
def prep_data_for_scoring(tariff_plan, data, list_col):
    """
    This function prepares data for scoring
    Arguments:
    tariff_plan -- the name of tariff plan we want to work with
    data -- data    
    list_col -- list of columns that should be dropped from X (['From', 'To', 'Region', 'Days_Not_Active', 'change'])
    
    Returns:
    data -- data of tariff plan

    """
    data = data[data['From'] == tariff_plan]
    features = list(data.columns) #name of features 
    data = data[features].drop(list_col, axis = 1)

    return data

def pipeline_new_data(X, y, pipeline,save_model=None,saved_model_name=None):
    """
    Fits saved pipelines with the best hyperparameters on a new data
    Arguments:
    X -- the feature columns
    y -- the target column
    pipeline -- pipeline with best estimators saved from gridsearchcv results
 
    Returns: 
    pipeline -- fitted pipeline"""
    
    name = saved_model_name   
    pipeline.best_estimator_.fit(X, y)

    if save_model=='yes':
        pickle.dump(pipeline,open(name,'wb'))
    else:
        print('model is not saved')
    return pipeline


def predictions(X, model):
    """
    Loads the model in a pipeline and gives predictions.
    
    Arguments:
    X -- the feature columns
    model -- the loaded model
    
    Returns:
    predicted_target -- predicted classes
    predicted_target_prob -- predicted probabilities for classes
    df_prob -- dataframe with probabilities
    """
    predicted_target =  model.predict(X)
    predicted_target_prob = model.predict_proba(X)
    df_prob = pd.DataFrame({'Probabilities':predicted_target_prob[:,1]}, index 
                       = X.index).sort_values(by='Probabilities', ascending=False)
    
    return predicted_target, predicted_target_prob, df_prob


from downloading_file import *
#for one file
def delete_called_numbers(data_with_probs, table_name='OUTBOUND_CALLS', lib_name='UpAction' ):
    #downloading file with already called numbers from SAS
    called_msisdn = LoadingFiles(table_name, lib_name).from_sas()
    #keeping rows which has comment (those numbers were called)
    called_msisdn = called_msisdn[called_msisdn['Comment'].notna()]
    #from called list dropping numbers which were "unreachable" or "no answer", so that we can call them again
    called_msisdn = called_msisdn.drop(called_msisdn[called_msisdn.Comment.isin(["Unreachable", "No answer"])].index)
    #removing called numbers from data
    data_without_called_nums = data_with_probs.drop(data_with_probs[data_with_probs.index.isin(called_msisdn.MSISDN)].index)
    return data_without_called_nums


def model_performance(y, predicted_target, predicted_target_prob):
    """
    Prints and returns the model performance
    y -- the target column
    predicted_target -- the predicted classes
    predicted_target_prob -- the predicted probability of classes
    
    Returns:
    score_list -- list with scores  
    """
    f1 = f1_score(y, predicted_target)
    precision = precision_score(y, predicted_target)
    recall = recall_score(y, predicted_target)
    roc_auc = roc_auc_score(y, predicted_target_prob)
    acc = accuracy_score(y,predicted_target)

    cm = confusion_matrix(y,predicted_target)
    false_pos_rate = cm[0,1]/(cm[0,1]+cm[0,0])
    false_neg_rate = cm[1,0]/(cm[1,0]+cm[1,1])
    specificity = cm[0,0]/(cm[0,0]+cm[0,1]) #true negative rate
    
    skplt.metrics.plot_confusion_matrix(y, predicted_target,title='Confusion matrix')
    plt.plot(figsize=(5, 10))
    plt.show() 
    print('\n F1 score = %.3f\n Precision = %.3f\n Recall = %.3f\n ROC_AUC = %.3f\n Accuracy=%.3f\n False positive rate =%.3f\n False negative rate = %.3f\n Specificity = %.3f '% (f1, precision, recall, roc_auc, acc, false_pos_rate, false_neg_rate, specificity))
    score_list = [f1, precision, recall, roc_auc, acc, false_pos_rate, false_neg_rate, specificity]
    return score_list


def model_performance_values(clfs, X, y):
    """
    Calculates and combines all model scores in one dataframe
    Arguments:
    clf -- models
    X -- the features column
    y -- target column
    
    Returns:
    scores_df -- dataframe with scores
    """
    model_scores= {}
    for clf in clfs:
        scores_ = {}
        model_name = clf['clf'].__class__.__name__
        pred_target, pred_target_prob = predictions(X, clf)
        scores_ = model_performance(y, pred_target, pred_target_prob[:,1])
        model_scores[model_name] = scores_
    scores_df = pd.DataFrame(model_scores, index=['F1', 'Precision', 'Recall', 'ROC AUC score', 'Accuracy', 'False positive rate', 'False negative rate', 'Specificity']).T

    return scores_df

#check
def threshold_roc(y, y_probs, X, model):
    """
    The approach is to test the model with each threshold returned from the call roc_auc_score() and select the threshold 
    with the largest G-Mean value. The Geometric Mean or G-Mean is a metric for imbalanced classification that, if 
    optimized, will seek a balance between the sensitivity and the specificity.
    G-Mean = sqrt(Sensitivity * Specificity)
    This function prints scores and confusion matrix with classes after new threshold and plots ROC curve
    Arguments:
    y -- target column
    y_probs -- predicted probabilities
    X - the features column
    model -- classification model
    
    Returns:
    preds -- predicted classes with a new threshold
    score_list -- list with scores
    """
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y, y_probs[:, 1])
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    plt.plot(fpr, tpr, marker='.')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.show()
    threshold = thresholds[ix]
    preds = np.where(model.predict_proba(X)[:,1] > threshold, 1, 0)
    score_list = model_performance(y, preds, y_probs[:,1])

    '''print('\n F1 score = %.3f\n Precision = %.3f\n Recall = %.3f\n ROC_AUC = %.3f\n Accuracy=%.3f\n False positive rate = %.3f\n False negative rate = %.3f\n Specificity = %.3f '% (f1, precision, recall, roc_auc, acc, false_pos_rate, false_neg_rate, specificity))
    score_list = [f1, precision, recall, roc_auc, acc, false_pos_rate, false_neg_rate, specificity]'''
    return preds, score_list


def df_top(data, num, probs, pred_labels, act_labels):
    """
    Takes top n percent of predicted probabilities
    Arguments: 
    data -- data with actual class labels, predicted class labels and predicted probabilities
    num -- top percent
    probs -- probability column
    pred_labels -- column with predicted labels
    act_labels -- column with actual labels
    
    Returns:
    model_scores -- scores for top n% users
    """
    pr_df_top= data.sort_values(by=probs,ascending=False).head(int(len(data)*(num/100)))
    model_scores = model_performance(pr_df_top[act_labels], pr_df_top[pred_labels], pr_df_top[probs])
    return model_scores

#############################################################
def stratified_sampling(predicted_target_prob, df, top_fraction, fraction_crit):
    """
    Stores predicted probabilities into dataframes in a necessary format. After creating dataframe with top probabilities 
    and adding "strata" column this function splits data into control and target by keeping the same proportion of each 
    desired variable (strata) that is present in the population. 
    
    Arguments:
    predicted_target_prob -- predicted probabilities
    df -- dataframe which is used to take indexes (phone numbers)
    top_fraction -- the fraction of top probabilities which will be taken
    fraction_crit -- criteria for definig subpopulations: "1" for probabilities less than "fraction_crit", "2" 
    for higher probs
    
    Returns:
    top_prob -- dataframe with top probabilities and MSISDN
    control -- stratified random sample, 25% of top_prob
    target -- stratified random sample, 75% of top_prob
    """
    df_prob = pd.DataFrame({'Probabilities':predicted_target_prob[:,1] },
                          index = df.index).sort_values(by='Probabilities', ascending=False)
    #Taking top % probabilities
    top_prob = df_prob[:int(len(df_prob)* top_fraction)]
    #Defining the subpopulations we want to sample from
    top_prob['strata'] = [1 if i < fraction_crit else 2 for i in top_prob.Probabilities]
    #Setting the split criteria
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.25)
    #Performing data frame split
    for x, y in split.split(top_prob, top_prob['strata']):
        #stratified random samples
        control = top_prob.iloc[y].sort_values(by='MSISDN')
        target = top_prob.iloc[x].sort_values(by='MSISDN')
        
    return top_prob, control, target


def create_dir(category, month, target, control):
    """
    Creates folders and saves files there
    Arguments:
    category -- folder: "Upsell" or "Downsell" 
    month -- folders for months
    target -- target data containing MSISDN and probabilities
    control -- control data containing MSISDN and probabilities
    """
    #Current working directory
    parent_dir = os.getcwd()
    #Creating directories
    directory = r"\Tariff_plan\{}\{}".format(category, month)
    path = parent_dir + directory
    os.makedirs(path)
    #Saving files in created folders
    target.sort_values(by='Probabilities', ascending=False).to_csv(path + "\Target.csv", header=True)
    control.sort_values(by='Probabilities', ascending=False).to_csv(path + "\Control.csv", header=True)

#############################################################

#Column names for dataframe
header = ("Date", "Action", "Subject", "Classifier", "Score")
filename = "output.csv"

def writer(header, data, filename, option):
    """
    Writes csv file
    Arguments:
    header -- column names
    data -- dataframe data (list)
    filename -- name of the file
    option -- if "write" function is used for creating csv file, if "update" function is used for "updater" 
    function
    """
    with open (filename, "w", newline = "") as csvfile:
        if option == "write":
            result = csv.writer(csvfile)
            result.writerow(header)
            for x in data:
                result.writerow(x)
        elif option == "update":
            writer = csv.DictWriter(csvfile, fieldnames = header)
            writer.writeheader()
            writer.writerows(data)
        else:
            print("Option is not known")


def updater(filename, row_number, col_name, value):
    """
    Updates an existing row. First it opens the file defined in the filename variable and then saves all the 
    data it reads from the file inside of a variable named readData. The second step is to hard code the new 
    value and place it instead of the old one.
    Then calls the writer function by adding a new parameter update that will tell the function that you 
    it should do an update.
    
    Arguments:
    filename -- name of the file
    row_number -- number of row which should contain the new value
    col_name -- column name which should contain the new value
    value -- new value
    """
    with open(filename, newline= "") as file:
        readData = [row for row in csv.DictReader(file)]
        #print(readData)
        readData[row_number][col_name] = value
        #print(readData)

    readHeader = readData[0].keys()
    writer(readHeader, readData, filename, "update")
    
def append_dict_as_row(filename, dict_of_elem, header):
    """
    Appends a dictionary as a new row. Can append dictionary with missing entries as well.
    Arguments:
    filename -- name of the file where row should be added
    dict_of_elem -- dictionary with elements . Keys in this dictionary match with the column names
    header -- column names
    """
    # Opening file in append mode
    with open(filename, 'a+', newline='') as write_obj:
        # Creating a writer object from csv module
        dict_writer = csv.DictWriter(write_obj, fieldnames = header)
        # Adding dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)
        
 

