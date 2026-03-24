import os
import time

from create_and_train_model import create_and_train_model
from test_model import test_model
from ResultsAcessor import ResultsAcessor
from ModelConfig import ModelConfig
from aux import return_unique_values, tasks

num_epochs = 50

dataset = 'heart.csv'
target='target'

hidden_activations = ['sigmoid', 'relu', 'tanh']
task = tasks[1] # 0: multi_class_classification, 1: binary_classification, 2: regression

if task == tasks[0]: 
    n, l = return_unique_values(dataset, target)
else: 
    n, l = 1, return_unique_values(dataset, target)[1]

topologies = [
    (128, n),
    (128, 64, n),
    (128, 64, 32, n),
    (256, 128, n)
]

temp = dataset.split('.')[0]
os.makedirs(temp, exist_ok=True)

ra = ResultsAcessor(
    acc_path = f'{temp}/accuracies.txt'
)


for hidden_activation in hidden_activations:
    for topology in topologies:
        
        mc = ModelConfig(
            topology=topology,
            hidden_activation=hidden_activation,
            num_epochs=num_epochs,
            dataset=dataset,
            target=target
        )

        config = mc.get_config(task)

        start = time.time()
        y_test_list, pred_list = create_and_train_model(**config)
        diff = time.time() - start

        acc = test_model(y_test_list, pred_list, config['loss'], l)
        ra.save_accuracy(topology, hidden_activation, num_epochs, acc, diff)



