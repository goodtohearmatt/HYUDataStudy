# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import copy
import itertools

from decimal import Decimal
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2

from sklearn import preprocessing
import scipy as sp

sns.set(palette='hls', font_scale=1.5)

#%%

var_dict = {'categorical': ['loan_status', 'emp_title', 'home_ownership', 
                            'verification_status', 'issue_d', 'purpose', 
                            'delinq_2yrs', 'inq_last_6mths', 'pub_rec'],
            'continuous': ['int_rate', 'annual_inc', 'loan_amnt',
                           'emp_length', 'dti', 'revol_bal', 'total_acc']}

def get_summary(dataframe, model):
    """ DataFrame must be same with the dataframe when modling """
    dataframe = get_dummy(dataframe)
    X = dataframe.drop('loan_status', axis=1)
    y = dataframe['loan_status']
    scores, pvalues = chi2(X, y)
    
    summary = pd.DataFrame([X.columns, model.coef_[0], [round(x,4) for x in pvalues]]).T
    summary.columns = ['Feature', 'Coefficient', 'p-value']
    return summary

def under_sampling(dataframe):
    paid, default = dataframe['loan_status'].value_counts()
    remove_num = paid - default
    remove_ix = dataframe[dataframe['loan_status']==1].sample(remove_num).index.tolist()
    dataframe = dataframe.drop(dataframe.index[remove_ix])
    return dataframe
    
def plot_confusion_matrix(test_data, model,
                          classes=[0,1],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    cm = confusion_matrix(test_data['loan_status'], model.predict(test_data.drop('loan_status', axis=1)))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    print(classification_report(test_data['loan_status'], model.predict(test_data.drop('loan_status', axis=1))))
    
def prep_data(dataframe, dummy=True):
    dataframe = dataframe.drop('grade', axis=1)
    dataframe = var_scaler(dataframe, feature_range=(0, 10))
    if dummy: dataframe = get_dummy(dataframe)
    return dataframe

def get_dummy(dataframe):
    all_var = []
    all_var.extend(var_dict['categorical'])
    all_var.extend(var_dict['continuous'])
    
    ctg_var = copy.deepcopy(var_dict['categorical'])
    ctg_var.remove('loan_status')
    else_var = [x for x in all_var if x not in ctg_var]
    else_df = dataframe[else_var]

    for var in ctg_var:
        else_df = else_df.join(pd.get_dummies(dataframe[var], prefix=var))
    return else_df
    
def get_logistic(dataframe):
    dataframe = get_dummy(dataframe)
    X = dataframe.drop('loan_status', axis=1)
    y = dataframe['loan_status']
    
    lr = LogisticRegression()
    lr_fit = lr.fit(X, y)
    cv = StratifiedKFold(n_splits=3)
    lr_cv = cross_val_score(lr, X, y, cv = cv)
    score = round(Decimal(np.mean(lr_cv)*100), 2)
    print('Accuracy: {}%'.format(score))
    return lr_fit
    
def heatmap_display(dataframe):
    ctg_list = []
    ctg_list.extend(var_dict['continuous'])
    ctg_list.append('grade')
    fig = plt.figure(figsize = (10, 8))
    corrmat = dataframe[ctg_list].corr()
    sns.heatmap(corrmat)
    plt.title("Correlation between Features")
    plt.show()
    
def hist_display(dataframe, var_type):
    N=int(len(var_dict[var_type])/3); M=3
    fig = plt.figure(figsize=(10,6))  # figure size
    
    plt.subplots_adjust(top=1, bottom=0, hspace=0, wspace=0)  # subplot setting
    
    # make subplot with face data index
    for i in range(N):
        for j in range(M):
            ax = fig.add_subplot(N, M, i * M + j + 1)
            plt.hist(dataframe[var_dict[var_type][i * M + j]], bins=12)
            plt.title(var_dict[var_type][i * M + j])
            
    plt.tight_layout()
    plt.suptitle("Histograms for {} variables".format(var_type), y=1.02, fontsize=15)
    plt.show()
    
def qq_display(dataframe):
    N=2; M=3;  # set row and column of the figure
    fig = plt.figure(figsize=(10,5))  # figure size
    plt.subplots_adjust(top=1, bottom=0, hspace=0, wspace=0)  # subplot setting
    
    # make subplot with face data index
    for i in range(N):
        for j in range(M):
            ax = fig.add_subplot(N, M, i * M + j + 1)
            sp.stats.probplot(dataframe[var_dict['continuous'][i * M + j]], plot=plt)
            plt.title(var_dict['continuous'][i * M + j])
            
    plt.tight_layout()
    plt.suptitle("QQ-Plots for continuous variables", y=1.02, fontsize=15)
    plt.show()
    
def var_scaler(dataframe, feature_range=(0, 10)):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
    scaled_var = min_max_scaler.fit_transform(dataframe[var_dict['continuous']])
    scaled_df = pd.DataFrame(scaled_var)
    scaled_df.columns = var_dict['continuous']
    dataframe[var_dict['continuous']] = scaled_df[var_dict['continuous']]
    return dataframe

def remove_outlier(dataframe, var_type, threshold=0.01):
    outliers = int(len(dataframe) * threshold)
    out_ix = []
    for ctg in var_dict[var_type]:
        out_large_list = dataframe[ctg].nlargest(outliers).index.tolist()
        out_small_list = dataframe[ctg].nsmallest(outliers).index.tolist()
        out_ix.extend(out_large_list)
        out_ix.extend(out_small_list)
        out_ix = list(set(out_ix))
        
    dataframe = dataframe.drop(dataframe.index[out_ix])
    dataframe.index = range(len(dataframe))
    print("The number of outliers({}%): {}".format(threshold*100, len(out_ix)))
    return dataframe
    

