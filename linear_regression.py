import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, loss_axis=None, plot_axis=None):
        self.W = np.array([0.0, 0.0])

        # variables used for plotting the loss and model prediction
        self.loss_axis = loss_axis
        self.plot_axis = plot_axis
        self.loss_line = None
        self.plot_line = None

    def compute_loss(self, X, y, reg):
        '''Compute the Mean-squared loss.
           Inputs:
                X  - An array of training inputs of size N
                y  - An array of training values of size N
               reg - Regularisation strength
           Returns:
            loss
            dW - Gradient of the loss, same shape as W

           Remember to apply regularization to your Mean-squared loss!
           Try Weight Decay first but feel free to try others
       '''
        ##
        # TODO YOUR CODE HERE
        ##
        loss = 0
        dW = np.zeros(self.W.shape)

        # End
        return loss, dW

    def predict(self, X):
        '''Execute our linear model
           y = wX + b
           Inputs:
               X - An array of inputs of size N
           Returns:
               Array of predicted values of size N
        '''
        y = np.zeros(X.shape)
        ##
        # TODO YOUR CODE HERE
        ##
        return y


    def train(self, X, y, learning_rate, num_iters, reg, do_live_plotting=True):
        '''Do gradient descent and iteratively update your weights based on
           the loss
           Inputs:
               X             - array of training inputs of size N
               y             - training values of size N
               learning_rate - scalar value determining the learning rate
               num_iters     - number of iterations in the gradient descent
               reg           - Regularisation strength
           Note:
               After each iteration make sure you append the loss for that set of
               weights to the loss_history list, else the visualization won't work!
           '''

        loss_history = list()

        for i in range(0, num_iters):
            ##
            # TODO YOUR CODE HERE
            ##

            if do_live_plotting:
                self.update_live_plot(X, loss_history)

        if not do_live_plotting:
            self.plot_loss_history(loss_history)
            self.plot_model(X)

    def update_live_plot(self, X, loss_history):
        if self.loss_line is None and self.loss_axis is not None:
            self.plot_loss_history(loss_history)
        else:
            self.loss_line.set_xdata(np.arange(0, len(loss_history)))
            self.loss_line.set_ydata(loss_history)
            self.loss_axis.relim()
            self.loss_axis.autoscale_view()

        if self.plot_line is None and self.plot_axis is not None:
            self.plot_model(X)
        else:
            self.plot_line.set_ydata(self.predict(X))

        plt.draw()
        plt.pause(0.001)

    def plot_loss_history(self, loss_history):
        self.loss_line, = self.loss_axis.plot(
                            np.arange(0, len(loss_history)), loss_history)

    def plot_model(self, X):
        self.plot_line, = self.plot_axis.plot(X, self.predict(X))


