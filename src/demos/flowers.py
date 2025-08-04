from sklearn.datasets import load_iris
import pandas as pd

import pandas as pd

def normalize_dataframe(df, columns=None):
    """
    Normalize specified columns in a DataFrame using Min-Max scaling.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be normalized.
    columns : list of str, optional
        A list of column names to normalize. If None, all numeric columns are normalized.

    Returns
    -------
    pd.DataFrame
        A DataFrame with normalized values in the specified columns.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # If no columns are specified, normalize all numeric columns
    if columns is None:
        columns = df_copy.select_dtypes(include=['float64', 'int64']).columns

    # Normalize each specified column using Min-Max scaling
    for col in columns:
        min_val = df_copy[col].min()
        max_val = df_copy[col].max()
        df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)

    return df_copy


def load_iris_data():
    """
    Load the Iris dataset and return it as a pandas DataFrame.

    This function retrieves the Iris dataset from scikit-learn,
    formats the data into a DataFrame, and adds columns for both
    the numerical species label and the species name.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Iris dataset with feature columns,
        the numeric species label (`species`), and the species name (`species_name`).
    """
    # Load the Iris dataset from scikit-learn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    
    # Add species label and name to the DataFrame
    df['species'] = iris.target
    df['species_name'] = [iris.target_names[i] for i in iris.target]
    return df

def main():
    """
    Main execution function.

    Loads the Iris dataset and prints the first few rows of the resulting DataFrame.
    """

    # Load the Iris dataset
    df = load_iris_data()
    
    # Normalize the specified columns in the DataFrame (values are between 0 and 1)
    dataset = normalize_dataframe(df, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])


if __name__ == "__main__":
    main()
