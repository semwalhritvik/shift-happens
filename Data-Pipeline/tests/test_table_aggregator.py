import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.table_aggregator import aggregate_and_merge_dfs

@pytest.fixture
def mock_application_train() -> pd.DataFrame:
    """
    Simulates the main application_train DataFrame holding the primary key SK_ID_CURR.
    """
    return pd.DataFrame({
        'SK_ID_CURR': [1001, 1002, 1003],
        'TARGET': [0, 1, 0],
        'AMT_INCOME_TOTAL': [50000, 60000, 70000]
    })

@pytest.fixture
def mock_bureau() -> pd.DataFrame:
    """
    Simulates the bureau DataFrame tracking previous credits with multiple rows per SK_ID_CURR.
    Includes mixed numerical and categorical columns to ensure we only aggregate numericals.
    """
    return pd.DataFrame({
        'SK_ID_CURR': [1001, 1001, 1002, 1003, 1003, 1003],
        'SK_ID_BUREAU': [501, 502, 503, 504, 505, 506],
        'DAYS_CREDIT': [-100, -200, -50, -300, -400, -500],
        'AMT_CREDIT_SUM': [1000, 2000, 3000, 4000, 5000, 6000],
        'CREDIT_ACTIVE': ['Closed', 'Active', 'Closed', 'Active', 'Active', 'Closed']
    })

def test_aggregate_and_merge_dfs(mock_application_train, mock_bureau):
    """
    Verifies that the table_aggregator correctly performs mean, sum, min, max, count 
    aggregations and merges into the main table successfully.
    """
    result_df = aggregate_and_merge_dfs(mock_application_train, mock_bureau)
    
    # 1. Assert row count is exactly identical to the main dataframe (no duplicates or drops)
    assert len(result_df) == 3
    assert len(result_df) == len(mock_application_train)
    
    # 2. Check SK_ID_CURR == 1001
    row_1001 = result_df[result_df['SK_ID_CURR'] == 1001].iloc[0]
    # Count: 2 rows in bureau
    assert row_1001['BUREAU_DAYS_CREDIT_COUNT'] == 2
    # Sum: -100 + -200 = -300
    assert row_1001['BUREAU_DAYS_CREDIT_SUM'] == -300
    # Mean: (-100 + -200)/2 = -150
    assert row_1001['BUREAU_DAYS_CREDIT_MEAN'] == -150
    # Max: -100
    assert row_1001['BUREAU_DAYS_CREDIT_MAX'] == -100
    # Min: -200
    assert row_1001['BUREAU_DAYS_CREDIT_MIN'] == -200
    # Check sum for amount
    assert row_1001['BUREAU_AMT_CREDIT_SUM_SUM'] == 3000
    
    # 3. Check SK_ID_CURR == 1003 (3 rows in bureau)
    row_1003 = result_df[result_df['SK_ID_CURR'] == 1003].iloc[0]
    assert row_1003['BUREAU_AMT_CREDIT_SUM_COUNT'] == 3
    assert row_1003['BUREAU_AMT_CREDIT_SUM_MEAN'] == 5000
    assert row_1003['BUREAU_DAYS_CREDIT_MIN'] == -500
    
    # 4. Check categorical exclusions
    # Ensure categorical columns like 'CREDIT_ACTIVE' weren't mistakenly aggregated
    assert 'BUREAU_CREDIT_ACTIVE_MEAN' not in result_df.columns
    # Ensure the identifier column 'SK_ID_BUREAU' wasn't aggregated
    assert 'BUREAU_SK_ID_BUREAU_COUNT' not in result_df.columns
