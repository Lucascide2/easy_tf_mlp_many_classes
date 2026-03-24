import pandas as pd
import numpy as np

def return_unique_values(dataset, target):
    df = pd.read_csv(dataset)
    uniques = np.sort(df[target].unique())
    print("unicos: ", uniques)
    return len(df[target].unique()), uniques[-1]

tasks = ['multi_class_classification', 'binary_classification', 'regression']