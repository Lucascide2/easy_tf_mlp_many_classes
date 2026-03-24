class ResultsAcessor:
    def __init__(self, path= 'predictions.txt', acc_path='accuracies.txt'):
        self.path = path
        self.acc_path = acc_path

    def save_results(self, y_test, predictions):
        with open(self.path, 'w') as f:
            f.write(str(y_test))
            f.write('\n')
            f.write(str(predictions))

    def load_results(self):
        with open(self.path, 'r') as f:
            y_test_str = f.readline()
            list_str = f.readline()
        return y_test_str, list_str
    
    def save_accuracy(self, topology, hidden_activation, num_epochs, acc, time):
        with open(self.acc_path, 'a') as f:
            f.write(f"Topology: {topology}, Hidden Activation: {hidden_activation}, Num Epochs: {num_epochs}, Accuracy: {acc}, Time: {time}\n")        