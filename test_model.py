import numpy as np

def test_model(y_test, predictions, loss, l = None):
    if loss == 'sparse_categorical_crossentropy':
        acc = np.array(predictions) == np.array(y_test)
    elif loss == 'binary_crossentropy':
        pred_arr = np.array(predictions)
        acc = np.isclose(pred_arr, np.array(y_test), rtol=0.5)
    else:
        pred_arr = np.array(predictions)
        pred_arr[pred_arr > l] = l
        pred_arr[pred_arr < 0] = 0
        acc = np.isclose(pred_arr, np.array(y_test), rtol=0.5)

    print("\n\n\######## Acurácia: ########")
    acc = acc.sum() / len(acc)
    
    print(acc)
    return acc
