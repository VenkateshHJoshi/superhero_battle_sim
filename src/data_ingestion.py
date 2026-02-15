import pandas as pd
import os

def load_data():
    """
    Loads the superheroes dataset from the raw data folder.
    """
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to the project root and then into data/raw
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, 'data', 'raw', 'superheroes_nlp_dataset.csv')

    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully!")
        print(f"📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return None

if __name__ == "__main__":
    # Load the data
    df = load_data()

    if df is not None:
        # Display basic information
        print("\n--- Column Names ---")
        print(df.columns.tolist())

        print("\n--- First 5 Rows ---")
        print(df.head())

        print("\n--- Data Types & Non-Null Counts ---")
        print(df.info())

        print("\n--- Missing Values per Column ---")
        print(df.isnull().sum())