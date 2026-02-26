from sklearn.model_selection import train_test_split
import pandas as pd

SEED = 42  # Set a random seed


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    # Drop duplicate rows
    df = df.drop_duplicates()
    print("Number of duplicate rows after dropping:", df.duplicated().sum())

    return df


def preprocess_data(df):

    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    return X_train, X_test, y_train, y_test
