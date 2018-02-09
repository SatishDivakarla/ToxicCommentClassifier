from sklearn.metrics import log_loss

import numpy as np
import pickle


def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    best_loss = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0

    while True:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        print("Predicting with the model")
        y_pred = model.predict(val_x, batch_size=batch_size)
        print("prediction complete")
        total_loss = 0
        for j in range(6):
            loss = log_loss(val_y[:, j], y_pred[:, j])
            total_loss += loss

        total_loss /= 6.
        print("Loss calculated")
        print("Epoch {0} loss {1} best_loss {2}".format(current_epoch, total_loss, best_loss))

        current_epoch += 1
        if total_loss < best_loss or best_loss == -1:
            print("Updating the best model")
            best_loss = total_loss
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            print("Check if the model is converging")
            if current_epoch - best_epoch == 5:
                print("Found a model")
                break

    model.set_weights(best_weights)
    return model


def train_folds(X, y, fold_count, batch_size, get_model_func):
    fold_size = len(X) // fold_count
    models = []
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = X[fold_start:fold_end]
        val_y = y[fold_start:fold_end]

        model = _train_model(get_model_func(), batch_size, train_x, train_y, val_x, val_y)
        pickle.dump(model, "model_"+fold_id+"epoch.pkl")
        models.append(model)

    return models
