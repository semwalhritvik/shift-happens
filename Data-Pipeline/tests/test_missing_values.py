import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.missing_value_treatment import impute_missing_values

@pytest.fixture
def mock_df_with_nans() -> pd.DataFrame:
    """
    Generates a mock Pandas DataFrame simulating mixed data types with missing values.
    Includes an edge case where an entire column is empty.
    """
    data = {
        'NUM_COL_1': [10.0, np.nan, 20.0, 30.0, 40.0],  # Median is 25.0
        'NUM_COL_2': [1.0, 1.0, np.nan, 2.0, 3.0],      # Median is 1.5
        'CAT_COL_1': ['A', 'B', np.nan, 'B', 'C'],      # Mode is 'B'
        'CAT_COL_2': [np.nan, 'X', 'Y', 'X', np.nan],   # Mode is 'X'
        'EMPTY_COL': [np.nan, np.nan, np.nan, np.nan, np.nan] # Entirely empty
    }
    return pd.DataFrame(data)

def test_impute_missing_values(mock_df_with_nans):
    """
    Tests that numerical NaNs use median, categorical use mode, and empty columns remain unchanged.
    """
    df_result = impute_missing_values(mock_df_with_nans)
    
    # Check NUM_COL_1: missing value at index 1 should be 25.0
    assert df_result.loc[1, 'NUM_COL_1'] == 25.0
    
    # Check NUM_COL_2: missing value at index 2 should be 1.5
    assert df_result.loc[2, 'NUM_COL_2'] == 1.5
    
    # Check CAT_COL_1: missing value at index 2 should be 'B'
    assert df_result.loc[2, 'CAT_COL_1'] == 'B'
    
    # Check CAT_COL_2: missing values at index 0 and 4 should be 'X'
    assert df_result.loc[0, 'CAT_COL_2'] == 'X'
    assert df_result.loc[4, 'CAT_COL_2'] == 'X'
    
    # Check EMPTY_COL: Should still be entirely NaN since no median/mode exists
    assert df_result['EMPTY_COL'].isnull().all()
    
    # Ensure no other NaNs exist in the first 4 columns
    assert df_result[['NUM_COL_1', 'NUM_COL_2', 'CAT_COL_1', 'CAT_COL_2']].isnull().sum().sum() == 0
