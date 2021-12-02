'''
Test churn library

Author: xxxx
Date: November 2021

'''

import os
import logging
import churn_library as churn
import pandas as pd
import pytest

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="module", autouse=True)
# @fixture(scope="module", autouse=True)
def import_bank_data():
    df = churn.import_data("./data/bank_data.csv")
    yield df


@pytest.fixture(scope="module", autouse=True)
# @fixture(scope="module", autouse=True)
def eda_fixture(import_bank_data):
    df = import_bank_data

    yield df


@pytest.fixture(scope="module", autouse=True)
def encoder_helper_fixture(import_bank_data):
    df = import_bank_data
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    yield df, cat_columns


def test_import(import_bank_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_bank_data
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(import_bank_data):
    '''
    test perform eda function
    '''

    try:
        df = import_bank_data
        assert churn.perform_eda(df) is True
        logging.info("Testing perform_eda: SUCCESS")

    except AssertionError as e:
        logging.error("Fail: Testing perform_eda")
        raise e


def test_encoder_helper(import_bank_data):
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    try:
        df: pd.DataFrame = import_bank_data
        result_df = churn.encoder_helper(df, cat_columns, 'Churn')
        for cat in cat_columns:
            assert result_df.columns.isin[cat + '_' + 'Churn'] == True
    except AssertionError as e:
        raise e


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    pass


def test_train_models(train_models):
    '''
    test train_models
    '''
    pass


if __name__ == "__main__":
    pass
