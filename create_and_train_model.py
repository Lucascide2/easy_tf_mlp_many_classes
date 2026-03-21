import os
from create_model import create_model
from ResultsAcessor import ResultsAcessor
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # remove esse spam específico

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_and_train_model(topology, hidden_activation, num_epochs, dataset):
    model = create_model(topology, hidden_activation=hidden_activation)

    # Carregando o dataset pandas
    df = pd.read_csv(dataset)

    X, y = df.drop(columns=['quality']), df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, shuffle = False)
    
    predictions = model.predict(X_test)
    pred_array = np.argmax(predictions, axis=1)
    pred_list = pred_array.tolist()

    y_test_list = y_test.values.tolist()

    acessor = ResultsAcessor()
    acessor.save_results(y_test_list, pred_list)

    return (y_test_list, pred_list)



