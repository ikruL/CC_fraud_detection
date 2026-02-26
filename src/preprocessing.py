from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

    # Scale the 'Time' and 'Amount' features
    scale_features = ['Time', 'Amount']
    scaler = StandardScaler()
    X_train[scale_features] = scaler.fit_transform(X_train[scale_features])
    X_test[scale_features] = scaler.transform(X_test[scale_features])

    # Handle imbalance
    smote = BorderlineSMOTE(random_state=SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test, scaler
