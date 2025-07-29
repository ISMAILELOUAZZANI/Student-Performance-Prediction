import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(path):
    df = pd.read_csv(path, sep=';')
    # Example: Create binary target (pass/fail)
    df['pass'] = df['G3'] >= 10
    X = df.drop(['G3', 'pass'], axis=1)
    y = df['pass']
    return train_test_split(X, y, test_size=0.2, random_state=42)