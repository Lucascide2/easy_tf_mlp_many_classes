import os

from create_and_train_model import create_and_train_model
from test_model import test_model
from ResultsAcessor import ResultsAcessor

"""
topology = (128, 64, 32, 16, 8, 6)
hidden_activation = 'sigmoid'
"""

num_epochs = 50
dataset = 'winequality_processed.csv'

hidden_activations = ['sigmoid']

topologies = [
    (16, 6),
    (32, 6),
    (64, 6)
]

temp = dataset.split('.')[0]
os.makedirs(temp, exist_ok=True)

ra = ResultsAcessor(
    acc_path = f'{temp}/accuracies.txt'
)

for hidden_activation in hidden_activations:
    for topology in topologies:
        y_test_list, pred_list = create_and_train_model(topology, hidden_activation, num_epochs, dataset)
        acc = test_model(y_test_list, pred_list)
        ra.save_accuracy(topology, hidden_activation, num_epochs, acc)



