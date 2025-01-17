import os
import pandas as pd
from typing import Optional

def save_results_to_csv(
    df: pd.DataFrame,
    save_directory: str,
    filename: Optional[str] = "results.csv"
) -> str:
    """
    Saves a pandas DataFrame to a CSV file in the specified directory.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe to save.
    save_directory: str
        Directory where the file will be saved.
    filename: str
        Name of the file to save.

    Returns
    -------
    str
        Path to the saved file.
    """
    if not os.path.isdir(save_directory):
        raise ValueError(f"The directory {save_directory} does not exist or is not a directory.")
    
    file_path = os.path.join(save_directory, filename)
    try:
        df.to_csv(file_path, index=False)
        return file_path
    except Exception as e:
        return f"Error saving file: {e}"