class ModelConfig:
    def __init__(self, topology, hidden_activation, num_epochs, dataset, target):
        self.topology = topology
        self.hidden_activation = hidden_activation
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.target = target

    def get_multi_class_classification(self):
        output_activation = 'softmax'
        loss = 'sparse_categorical_crossentropy'

        return {
            "topology": self.topology,
            "hidden_activation": self.hidden_activation,
            "output_activation": output_activation,
            "loss": loss,
            "num_epochs": self.num_epochs,
            "dataset": self.dataset,
            "target": self.target
        }
    
    def get_binary_classification(self):
        output_activation = 'sigmoid'
        loss = 'binary_crossentropy'

        return {
            "topology": self.topology,
            "hidden_activation": self.hidden_activation,
            "output_activation": output_activation,
            "loss": loss,
            "num_epochs": self.num_epochs,
            "dataset": self.dataset,
            "target": self.target
        }
    
    def get_regression(self):
        output_activation = 'linear'
        loss = 'mean_squared_error'

        return {
            "topology": self.topology,
            "hidden_activation": self.hidden_activation,
            "output_activation": output_activation,
            "loss": loss,
            "num_epochs": self.num_epochs,
            "dataset": self.dataset,
            "target": self.target
        }
    
    def get_config(self, task):
        if task == 'multi_class_classification':
            return self.get_multi_class_classification()
        elif task == 'binary_classification':
            return self.get_binary_classification()
        elif task == 'regression':
            return self.get_regression()
