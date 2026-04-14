import pickle
import pandas as pd


def load_pickle_data(file_path: str) -> pd.DataFrame:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col) for col in df.columns]
    return df
