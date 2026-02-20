import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.outlier_treatment import treat_days_employed_outliers

@pytest.fixture
def mock_application_train() -> pd.DataFrame:
    """
    Generates a mock Pandas DataFrame simulating the application_train.csv table.
    Contains DAYS_EMPLOYED with regular, negative (valid), and erroneous (365243) values.
    """
    data = {
        'SK_ID_CURR': [100001, 100002, 100003, 100004, 100005],
        'NAME_CONTRACT_TYPE': ['Cash loans', 'Cash loans', 'Revolving loans', 'Cash loans', 'Cash loans'],
        'DAYS_EMPLOYED': [-1000, 365243, -500, 365243, -2500],
        'AMT_INCOME_TOTAL': [150000, 250000, 100000, 135000, 180000]
    }
    return pd.DataFrame(data)

def test_treat_days_employed_outliers_converts_to_nan(mock_application_train):
    """
    Tests that the outlier treatment function specifically replaces 365243 with NaN,
    but leaves the normal/negative values intact.
    """
    df_result = treat_days_employed_outliers(mock_application_train)
    
    # Assert that the erroneous values (indexes 1 and 3) were converted to NaN
    assert np.isnan(df_result.loc[1, 'DAYS_EMPLOYED'])
    assert np.isnan(df_result.loc[3, 'DAYS_EMPLOYED'])
    
    # Assert that the legitimate/negative values (indexes 0, 2, 4) remained untouched
    assert df_result.loc[0, 'DAYS_EMPLOYED'] == -1000
    assert df_result.loc[2, 'DAYS_EMPLOYED'] == -500
    assert df_result.loc[4, 'DAYS_EMPLOYED'] == -2500

    # Ensure other columns remain perfectly intact
    pd.testing.assert_series_equal(mock_application_train['SK_ID_CURR'], df_result['SK_ID_CURR'])
    pd.testing.assert_series_equal(mock_application_train['AMT_INCOME_TOTAL'], df_result['AMT_INCOME_TOTAL'])
    
    # Assert total count of NaNs is exactly 2 in the target column
    assert df_result['DAYS_EMPLOYED'].isna().sum() == 2
