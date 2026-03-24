import numpy as np

def test_model(y_test, predictions, loss, l = None):
    if loss == 'sparse_categorical_crossentropy':
        acc = np.array(predictions) == np.array(y_test)
    elif loss == 'binary_crossentropy':
        pred_arr = np.array(predictions)
        pred_arr = (pred_arr > 0.5).astype(int)
        acc = pred_arr == np.array(y_test)
    else:
        pred_arr = np.array(predictions)
        pred_arr = np.clip(np.round(pred_arr), 0, l).astype(int)
        acc = pred_arr == np.array(y_test)      

    print("\n\n\######## Acurácia: ########")
    acc = acc.sum() / len(acc)
    
    print(acc)
    return acc
