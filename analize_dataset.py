import numpy as np
import pandas as pd

ds = 'winequality.csv'

df = pd.read_csv(ds)
df = df.drop(columns=['Id'])

X, y = df.drop(columns=['quality']), df['quality']

y = y.map({
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5
})

df['quality'] = y

# Salvar o dataframe em csv
df.to_csv('winequality_processed.csv', index=False)

print(y.value_counts())