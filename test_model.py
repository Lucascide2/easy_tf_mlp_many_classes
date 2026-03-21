import numpy as np

def test_model(y_test, predictions):
    acc = np.array(predictions) == np.array(y_test)

    print("\n\n\######## Acurácia: ########")
    acc = acc.sum() / len(acc)
    
    print(acc)
    return acc
