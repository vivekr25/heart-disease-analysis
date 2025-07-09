import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean and preprocess the DataFrame."""
    df = df.dropna()
    return df

def encode_features(df):
    """Perform one-hot encoding on categorical columns."""
    df_encoded = pd.get_dummies(df, columns=[
        'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'
    ], drop_first=True)
    return df_encoded

def split_and_scale(df):
    """Split the dataset into train and test sets and scale the features."""
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test