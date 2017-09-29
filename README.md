# Linear Regression

In this assignment you are going to be implementing and traning a linear
regression algorithm. The provided dataset and example is for predicting 
house prices based on the size of the living area, however you are
encouraged to try out other variables and datasets. 

The dataset in houses.csv is from Kaggle and contains much more than just
living space and house sizes. You can read more about it here:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

## Your environment

This exercise relies on python with the common libraries numpy and 
matplotlib. These can be installed using python pip like so:

pip install numpy matplotlib

If you don't have pip installed, see here for information: 
https://packaging.python.org/tutorials/installing-packages/

If you are struggling with python and numpy, see this tutorial:
http://cs231n.github.io/python-numpy-tutorial/


## The Task

You will be editing and completing the LinearRegression class in 
`linear_regression.py` as well as tuning your hyperparameters 
in `house_prices.py`

### Prediction

The first thing to do is to implement the `predict` method. This should
be straightforward. Remember that the equation for our linear system is

y = wx + b

### Cost functions

The next step is to implement the loss and gradient. The equation for
the Mean-squared loss is:

`J = 1/N ∑(f(X_i, W) - y_i)^2 + lambda*R(W)

Where f is our aforementioned prediction function and N is the number of
data points. W is the weights matrix [w b]. Lambda is the regularisation 
strength and R is our Regularisaion function. Feel free to try others, 
but start off with Weight Decay R(W) = W.T W

You will also need to calculate the partial derivatives of the loss, L with
respect to the two parts of our model, W and b.

Complete the method `compute_loss` and such that it returns the loss and
the gradient dW.

Hint: You may find it easier to write out the separate partial derivatives 
with respect to W ie. ∂J/∂w and ∂J/∂b, and derive those separately 
rather than trying to take the derivative of the matrix.

Numpy is optimized for vector operations and as such it's recommended that 
you write this method using numpy vector optimizations rather than looping 
over the dataset. It isn't essential to complete this task, however it'll
be useful in the future, so best to get some practice in!

If you're proficient with MATLAB or Octave, this should be familiar to you.
See the aforementioned python tutorial for more information.

### Gradient descent

For a dataset of this size, gradient descent is probably unnecessary with a
modern computer that is capable of inverting a large matrix. However,
with deep learning and other more complex datasets solving it directly 
won't be possible and gradient descent will be far more computationally
efficient.

Complete the `train` method using the gradients of the loss to update 
W and b. Make sure as well to append the calculated loss to the list
`loss_history`, else the visualization won't work!


## Training and running

Run the file `house_prices.py` from the command line like so:

```
python house_prices.py
```` 

It will load the data from houses.csv and set aside 1/5 your "test data" 
and the remainder will be used for training.

First it will plot the training data as a scatter point. Press any key to continue

It will then call the appropriate methods to train and run your linear 
regression model. If it's working you should see a nice exponential 
decline in your loss and the straight-line produced by your model move 
towards the best fit.

Once training has finised, press any key to close the visualization.
The accuracy of your model will then be printed to the console.

### Hyperparameters

After you've completed your Linear Regression code, 
running `python house_prices.py` probably won't work straight away.
There are three hyperparameters that you will need to tune: the number of 
iterations, the learning rate and the regularisation strength. 
You should find these in `house_prices.py`.

Setting the learning rate can be tricky. What I find works well is to
start very low so the loss doesn't decrease at all and then steadily 
increase the value until you see a good exponential decline in loss.

You may wish to separate some of your training data out in to a validation
set and try lots of different hyperparameters and choose the best ones.


## Results

If all goes well you should get an accuracy of just under 75%.


## Going further

Have a look at the dataset and see if you can find any other linear
relationships that you think would yeild better accuracy. If house
prices are too boring, try another dataset. 
Included in here is vgsales.csv - international sales data for videogames.
Perhaps train a regression model to predict Japanese sales based on American sales? 

https://www.kaggle.com/gregorut/videogamesales

Kaggle.com is a great resource for data sets, so perhaps find something on 
there that interests you.

Alternatively you could argue that the model is underfitting, so if you're
feeling adventurous perhaps try increasing its capacity.
You could try a multiple linear regression or a polynomial regression model. 

You can use your LinearRegression
class as a basis but it'll require some modification.

If you do something cool, share it among the study group!
