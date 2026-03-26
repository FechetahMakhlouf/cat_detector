import numpy as np
import h5py


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig  = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# ─────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y)
    grads = {"dw": dw, "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        w -= learning_rate * grads["dw"]
        b -= learning_rate * grads["db"]
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost:.4f}")
    return {"w": w, "b": b}, grads, costs


def predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    return (A > 0.5).astype(int)


def model(X_train, Y_train, X_test, Y_test,
          num_iterations=2000, learning_rate=0.005, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train,
                                    num_iterations, learning_rate, print_cost)
    w, b = params["w"], params["b"]
    Y_pred_train = predict(w, b, X_train)
    Y_pred_test  = predict(w, b, X_test)

    if print_cost:
        print(f"Train accuracy: {100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100:.2f}%")
        print(f"Test  accuracy: {100 - np.mean(np.abs(Y_pred_test  - Y_test))  * 100:.2f}%")

    return {
        "costs": costs,
        "Y_prediction_test":  Y_pred_test,
        "Y_prediction_train": Y_pred_train,
        "w": w, "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    }
