import pandas as pd


def load_data(filepath):
    data = pd.read_csv(filepath)
    bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89]
    labels = ['0s', '10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s']
    data['Age'] = pd.cut(data['Age'], bins=bins, labels=labels).astype(str)
    return data

def preprocess_data(model, data):
    return model.preprocess_data(data)