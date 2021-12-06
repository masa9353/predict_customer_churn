# library doc string
'''
Churn library

Author: xxxx
Date: November 2021

'''

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import normalize
# import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


BANK_DATA = './data/bank_data.csv'
IMAGE_PATH = './images/'
IMAGE_EDA_PATH = IMAGE_PATH + 'eda/'
SCORE_RESULT_PATH = './results.txt'
RESULT_IMAGE_PATH = './images/results/'
RESULT_LRC_PLOT_PATH = RESULT_IMAGE_PATH + 'lrc_plot.png'
RESULT_RFC_PLOT_PATH = RESULT_IMAGE_PATH + 'rfc_plot.png'

MODEL_RFC_PATH = './models/rfc_model.pkl'
MODEL_LOGISTIC_PATH = './models/logistic_model.pkl'

CLASSIFICATION_RFC_REPORT = RESULT_IMAGE_PATH + 'Random_Forest.png'
CLASSIFICATION_LRC_REPORT = RESULT_IMAGE_PATH + 'Logistic_Regression.png'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    # df = pd.read_csv(r"pth")
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    try:
        print(df.shape)
        print(df.isnull().sum())
        print(df.describe)
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        quant_columns = [
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio'
        ]
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        my_fig = plt.figure(figsize=(20, 10))
        df['Churn'].hist()
        my_fig.savefig(IMAGE_EDA_PATH + "Churn_hist.png")

        my_fig = plt.figure(figsize=(20, 10))
        df['Customer_Age'].hist()
        my_fig.savefig(IMAGE_EDA_PATH + "Customer_Age_hist.png")

        my_fig = plt.figure(figsize=(20, 10))
        df.Marital_Status.value_counts('normalize').plot(kind='bar')
        my_fig.savefig(IMAGE_EDA_PATH + "Material_Status.png")

        my_fig = plt.figure(figsize=(20, 10))
        sns.distplot(df['Total_Trans_Ct'])
        my_fig.savefig(IMAGE_EDA_PATH + "Total_Trans_Ct.png")

        my_fig = plt.figure(figsize=(20, 10))
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        my_fig.savefig(IMAGE_EDA_PATH + "heat_map.png")

    except Exception:
        return False

    return True


def encoder_helper(df, category_lst, response: str):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument 
                        that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    try:
        for cat in category_lst:
            tmp_list = []
            groups = df.groupby(cat).mean()[response]
            for val in df[cat]:
                tmp_list.append(groups.loc[val])
            df[cat + '_' + response] = tmp_list
    except Exception as e:
        raise e

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    df_ = encoder_helper(df, cat_columns, response)
    y = df[response]
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    my_fig = plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    my_fig.savefig(CLASSIFICATION_RFC_REPORT)

    my_fig = plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    my_fig.savefig(CLASSIFICATION_LRC_REPORT)


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    if isinstance(model, RandomForestClassifier) == False:
        raise Exception

    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    my_fig = plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    my_fig.savefig(output_pth + "feature_importance.png")


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=200)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores in text file
    with open(SCORE_RESULT_PATH, mode='w') as f:
        f.write('random forest results')
        f.write('test results')
        f.write(classification_report(y_test, y_test_preds_rf))
        f.write('train results')
        f.write(classification_report(y_train, y_train_preds_rf))
        f.write('logistic regression results')
        f.write('test results')
        f.write(classification_report(y_test, y_test_preds_lr))
        f.write('train results')
        f.write(classification_report(y_train, y_train_preds_lr))

    # save images
    my_fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    lrc_plot = plot_roc_curve(lrc, X_test, y_test, ax=ax)
    my_fig.savefig(RESULT_LRC_PLOT_PATH)

    my_fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_,
                              X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    my_fig.savefig(RESULT_RFC_PLOT_PATH)

    # save model
    joblib.dump(cv_rfc.best_estimator_, MODEL_RFC_PATH)
    joblib.dump(lrc, MODEL_LOGISTIC_PATH)
