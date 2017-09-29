import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

def load_data(filename):
    data = np.genfromtxt(filename, names=True, delimiter=',')
    # Assign about 1/5 to test
    data_size = np.size(data)
    test_size = data_size/5
    training_size = 4 * test_size

    training = data[0:training_size]
    test = data[training_size:]

    return training, test

def setup_axes(regression_xlabel, regression_ylabel, X, y):
    plt.ion()
    fig, axes = plt.subplots(2)
    axes[1].set_xlabel(regression_xlabel)
    axes[1].set_ylabel(regression_ylabel)

    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Loss')

    return fig, axes[0], axes[1]


def on_keypress(event):
    global waiting
    waiting = False


def wait_for_keypress():
    # Wait for a keypress
    global waiting
    waiting = True
    while waiting:
        fig.canvas.get_tk_widget().update()
        plt.pause(0.05)


def calculate_accuracy(model, X, y):
    diff = abs(np.divide(y - model.predict(X), y))
    return 1 - np.mean(diff)


if __name__ == "__main__":

    training, test = load_data('houses.csv')

    # Set our inputs and our training values
    X = training['GrLivArea']
    y = training['SalePrice']

    # plot our data points
    fig, loss_axes, plot_axes = setup_axes('Living Area (sqft)', 'Sale Price', X, y)
    fig.canvas.mpl_connect('key_press_event', on_keypress)
    plot_axes.scatter(X, y, marker='x', s=1)

    wait_for_keypress()

    # Set hyperparameters
    ##
    # TODO - Tune these. You may want to set aside a section of your training
    # data as your "validation" set and try lots of values to see what works best
    ##
    learning_rate = 0
    number_of_iterations = 50
    regularisation_strength = 0

    # May want to set this to false when tuning your hypeparameters
    animate = False

    # Train our linear regression model
    linear_regression = LinearRegression(loss_axes, plot_axes)
    linear_regression.train(
            X,
            y,
            learning_rate,
            number_of_iterations,
            regularisation_strength,
            animate)

    plot_axes.plot(X, linear_regression.predict(X))

    wait_for_keypress()

    # Caluclate final accuracy
    test_X = test['GrLivArea']
    test_y = test['SalePrice']

    training_acc = calculate_accuracy(linear_regression, X, y)
    test_acc = calculate_accuracy(linear_regression, test_X, test_y)

    print("W: {}".format(linear_regression.W))
    print("Training Accuracy: {}".format(training_acc))
    print("Test Accuracy: {}".format(test_acc))

    print("Normal equation solution:")
    Xe = np.array([X, np.ones(X.shape)])
    W = np.linalg.pinv(Xe.T).dot(y)
    print("W: {}".format(W))
    l2 = LinearRegression()
    l2.W = W
    plt.plot(X, l2.predict(X))
    print("Training Accuracy: {}".format(calculate_accuracy(l2, X, y)))
    print("Test Accuracy: {}".format(calculate_accuracy(l2, test_X, test_y)))

    wait_for_keypress()
