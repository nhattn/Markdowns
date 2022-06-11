---
title: Neural Network Code in d from Scratch
link: https://pythonalgos.com/create-a-neural-network-from-scratch-in-python-3/
author: Yujian Tang
---

In this post we're going to build a fully connected deep neural net (DNN)
from scratch in Python 3.

This code is adapted from
[Michael Nielson's Neural Networks and Deep Learning Book](http://neuralnetworksanddeeplearning.com/),
which was written for Python 2. Michael is way smarter than I am and
if you want a more in-depth (math heavy) explanation, I highly suggest
reading his book.

In this post we'll cover:

- Introduction to Neural Network Code in Python
  + Overview of the File Structure for Our Neural Network Code in Python 3
  + Setting Up Helper Functions
- Building the Neural Network Code from Scratch in Python
  + Feed Forward Function
  + Gradient Descent
  + Backpropagation for Neural Networks
    + Feeding Forwards
    + Backwards Pass
  + Mini-Batch Updating
  + Evaluation
  + Putting All The Neural Network Code in Python Together
  + Loading MNIST Data
  + Running Tests

You can find the [Github Here](https://github.com/ytang07/nn_examples). To
follow along to this tutorial you'll need to download the `numpy` Python library.
To do so, you can run the following command in the terminal:

```bash
pip3 install numpy

```

## Overview of File Structure for Our Neural Network Code in Python

There will be three files being made here. First, we have the `simple_nn.py`
file which will be outlined in "[Setting Up Helper Functions](https://pythonalgos.com/create-a-neural-network-from-scratch-in-python-3/#setting-up-helper-functions)"
and
"[Building the Neural Network from Scratch](https://pythonalgos.com/create-a-neural-network-from-scratch-in-python-3/#building-the-neural-network-from-scratch)".
We will also have a file to load the test data called `mnist_loader.py`,
outlined in
"[Loading MNIST Data](https://pythonalgos.com/create-a-neural-network-from-scratch-in-python-3/#loading-mnist-data)".
Finally, we will have a file to test our neural network called `test.py` that
will be run in the terminal. This file is outlined in
"[Running Tests](https://pythonalgos.com/create-a-neural-network-from-scratch-in-python-3/#running-tests)".

## Setting Up Helper Functions

![](https://lh5.googleusercontent.com/Y7q6wOwmF0TyT3niPhVPsDdj1WRlTicmNnDdFNJENPG1IIu0LpPViCdQJTiKROSkgylolJOQ4WOx9ojzievva_ayVY_4ylzbOacySWCrb7oDnjy8DxgW_YngVRmXVg4IPyKehaoi)

> Sigmoid Function, [Image from Wikipedia](https://en.wikipedia.org/wiki/Sigmoid_function)

At the start of our program we'll import the only two libraries we need,
`random`, and `numpy`. We've seen random used extensively via the Super Simply
Python series in programs like the
[Random Number Generator](https://pythonalgos.com/2021/11/24/super-simple-python-random-number-generator/),
[High Low Guessing Game](https://pythonalgos.com/2021/11/28/super-simple-python-high-low-guessing-game/),
and [Password Generator](https://pythonalgos.com/2021/11/29/super-simple-python-password-generator/).
We'll be using the random library to randomize the starting weights in our
neural network. We'll be using `numpy` or `np` (by convention it is usually
imported as np), to make our calculations faster.

After our imports, we'll create our two helper functions. A `sigmoid` function
and a `sigmoid_prime` function. We first learned about the sigmoid function in
[Introduction to Machine Learning: Logistic Regression](https://pythonalgos.com/2021/11/05/introduction-to-machine-learning-logistic-regression/).
In this program, we'll be using it as our activation function, the same way
as it's used to do classification in Logistic Regression. The `sigmoid_prime`
function is the derivative and is used in backpropagation to calculate the
`delta` or gradient. 

**neural network code in python - sigmoid and sigmoid prime**

```python
import random
import numpy as np
 
# helpers
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
 
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

```

## Building the Neural Network Code in Python from Scratch

![](https://lh4.googleusercontent.com/YOsY3KNbYr54Zv6FZkS2w9T68IZTBB-GAgIFfjEN4UGj_kZMtlX1TDi8s_j2B2Q5qMTP0fiZUGVKVcv6Shmn8avsc3t_bQZYeN6K61O1tSFo7tc9L2ZPWMX6dvV8-i-uT91WcMBl)

> Sample Deep Neural Network [Image from Stack Exchange](http://stats.stackexchange.com/questions/256342/how-many-learnable-parameters-does-a-fully-connected-layer-have-without-the-bias)

This entire section is dedicated to building a fully connected neural network.
All of the functions that follow will be under the network class. The
[full class code](https://pythonalgos.com/create-a-neural-network-from-scratch-in-python-3/#putting-it-all-together)
will be provided at the end of this section. The first thing we'll do in
our `Network` class is create the constructor.

The constructor takes one parameter, `sizes`. The `sizes` variable is a list
of numbers that indicates the number of input nodes at each layer in our
neural network. In our `__init__` function, we initialize four attributes.
The number of layers, `num_layers`, is set to the length of the `sizes` and
the list of the sizes of the layers is set to the input variables, `sizes`.
Next, the initial biases of our network are randomized for each layer after
the input layer. Finally, the weights connecting each node are randomized
for each connection between the input and output layers. For context, `np.random.randn()`
returns a random sample from the normal distribution.

**neural network code in python - network**

```python
class Network:
    # sizes is a list of the number of nodes in each layer
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]

```

## Feedforward Function

The `feedforward` function is the function that sends information forward
in the neural network. This function will take one parameter, ‘a', representing
the current activation vector. This function loops through all the biases
and weights in the network and calculates the activations at each layer.
The `a` returned is the activations of the last layer, which is the prediction.

**neural network code in python - feedforward**

```python
def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

```

## Gradient Descent

![](https://lh4.googleusercontent.com/zfTIA3Ag5H92f6ibsk0Gna3EPJUYv-Bk4-R5w8ac8gvbhGhE69gxU5DHHOR5_KfUNlnjbtTmQVCAZqNfenpAq_RQottF2LEeRSMIR-umzJdsj1eKi0FrM7GN4IYNI5pUPSEqKKVW)

> Image of Gradient Descent, [Image from Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html)

[Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) is the
workhorse of our `Network` class. In this version, we're doing an altered
version of gradient descent known as mini-batch (stochastic) gradient descent.
This means that we're going to update our model using a mini-batch of data
points. This function takes four mandatory parameters and one optional parameter.
The four mandatory parameters are the set of training data, the number of
epochs, the size of the mini-batches, and the learning rate (`eta`). We can
optionally provide test data. When we test this network later, we will provide
test data.

This function starts off by converting the `training_data` into a list type
and setting the number of samples to the length of that list. If the test
data is passed in, we do the same to that. This is because these are not
returned to us as lists, but `zip`s of lists. We'll see more about this when
we load the MNIST data samples later. Note that this type-casting isn't strictly
necessary if we can ensure that we pass both types of data in as lists.

Once we have the data, we loop through the number of training epochs. A training
epoch is simply one round of training the neural network. In each epoch,
we start by shuffling the data to ensure randomness, then we create a list
of mini-batches. For each mini-batch, we'll call the
[`update_mini_batch` method](https://pythonalgos.com/create-a-neural-network-from-scratch-in-python-3/#mini-batch-updating),
which is covered below. If the test data is there, we'll also return the
test accuracy.

**neural network code in python - SGD**

```python
def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        samples = len(training_data)
       
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
       
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, samples, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

```

## Backpropagation for Neural Networks

Backpropagation is the updating of all the weights and biases after we run
a training epoch. We use all the mistakes the network makes to update the
weights. Before we actually create the backpropagation function, let's create
a helper function called `cost_derivative`. The `cost_derivative` function
will determine if we made a mistake in our output layer. It takes two parameters,
the `output_activations` array and the expected output values, `y`.

**neural network code in python - cost derivative**

```python
def cost_derivative(self, output_activations, y):
        return(output_activations - y)

```

## Feeding Forwards

Now we're ready to do backpropagation. Our `backprop` function will take
two values, `x`, and `y`. The first thing we'll do is initialize our nablas
or 𝛁 to 0 vectors. This symbol represents the gradients. We also need to keep
track of our current activation vector, `activation`, all of the activation
vectors, `activations`, and the z-vectors, `zs`. The first activation is
the input layer.

After setting these up, we'll loop through all the biases and weights. In
each loop we calculate the `z` vector as the dot product of the weights and
activation, add that to the list of `zs`, recalculate the activation, and
then add the new activation to the list of `activations`.

## Backward Pass

Now comes the calculus. We start our backward pass by calculating the delta,
which is equal to the error from the last layer multiplied by the `sigmoid_prime`
of the last entry of the `zs` vectors. We set the last layer of `nabla_b`
as the delta and the last layer of `nabla_w` equal to the dot product of
the delta and the second to last layer of activations (transposed so we can
actually do the math). After setting these last layers up, we do the same
thing for each layer going backwards starting from the second to last layer.
Finally, we return the `nabla`s as a tuple.

**neural network code in python - backpropagation**

```python
def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # stores activations layer by layer
        zs = [] # stores z vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
       
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
       
        for _layer in range(2, self.num_layers):
            z = zs[-_layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-_layer+1].transpose(), delta) * sp
            nabla_b[-_layer] = delta
            nabla_w[-_layer] = np.dot(delta, activations[-_layer-1].transpose())
        return (nabla_b, nabla_w)

```

## Mini-Batch Updating

Mini-batch updating is part of our `SGD` (stochastic) gradient descent function
from earlier. I went back and forth on where to place this function since
it's used in `SGD` but also requires `backprop`. In the end I decided to
put it down here. It starts much the same way as our `backprop` function
by creating 0 vectors of the `nabla`s for the biases and weights. It takes
two parameters, the `mini_batch`, and the learning rate, `eta`.

Then, for each input, `x`, and output, `y`, in the `mini_batch`, we get the
delta of each nabla array via the `backprop` function. Next, we update the
`nabla` lists with these deltas. Finally, we update the weights and biases
of the network using the `nablas` and the learning rate. Each value is updated
to the current value minus the learning rate divided by the size of the minibatch
times the nabla value.

**neural network code in python - mini batch updating**

```python
def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

```

## Evaluation

The last function we need to write is the `evaluate` function. This function
takes one parameter, the `test_data`. In this function, we simply compare
the network's outputs with the expected output, `y`. The network's outputs
are calculated by feeding forward the input, `x`.

**neural network code in python - evaluation**

```python
def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(y[x]) for (x, y) in test_results)

```

## Putting All the Neural Network Code in Python Together

Here's what it looks like when we put all the code together.

**neural network code in python**

```python
import random
import numpy as np
 
# helpers
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
 
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
 
class Network:
    # sizes is a list of the number of nodes in each layer
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
       
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
   
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        samples = len(training_data)
       
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
       
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, samples, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")
   
    def cost_derivative(self, output_activations, y):
        return(output_activations - y)
   
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # stores activations layer by layer
        zs = [] # stores z vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
       
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
       
        for _layer in range(2, self.num_layers):
            z = zs[-_layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-_layer+1].transpose(), delta) * sp
            nabla_b[-_layer] = delta
            nabla_w[-_layer] = np.dot(delta, activations[-_layer-1].transpose())
        return (nabla_b, nabla_w)
   
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
       
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(y[x]) for (x, y) in test_results)

```

## Testing our Neural Network

Great, now that we've written our Neural Network, we have to test it. We'll
test it using the MNIST dataset. You can download the dataset (and original
Python 2.7 code) [here](https://github.com/mnielsen/neural-networks-and-deep-learning/archive/master.zip).

## Loading MNIST Data

The MNIST data comes in a `.pkl.gz` file type that we'll use `gzip` to open
and `pickle` to load. Let's create a simple function to load this data as
a tuple of size 3 split into the training, validation, and test data. To
make our data easier to handle, we'll create another function to encode the
`y` into an array of size 10. The array will contain all 0s except for a
1 which corresponds to the correct digit of the image. 

To load our data into a usable format, we'll use the simple `load_data` function
we created and the `one_hot_encode` functions. We will create another function
that will transform our `x` values into a list of size 784, corresponding
to the 784 pixels in the image, and our `y` values into their one hot encoded
vector form. Then we'll zip these `x` and `y` values together so that each
index corresponds to the other. We need to do this for the training, validation,
and test data sets. Finally, we return the modified data.

**loading MNIST data for neural network testing**

```python
import pickle
import gzip
 
import numpy as np
 
def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)
 
def one_hot_encode(y):
    encoded = np.zeros((10, 1))
    encoded[y] = 1.0
    return encoded
 
def load_data_together():
    train, validate, test = load_data()
    train_x = [np.reshape(x, (784, 1)) for x in train[0]]
    train_y = [one_hot_encode(y) for y in train[1]]
    training_data = zip(train_x, train_y)
    validate_x = [np.reshape(x, (784, 1)) for x in validate[0]]
    validate_y = [one_hot_encode(y) for y in validate[1]]
    validation_data = zip(validate_x, validate_y)
    test_x = [np.reshape(x, (784, 1)) for x in test[0]]
    test_y = [one_hot_encode(y) for y in test[1]]
    testing_data = zip(test_x, test_y)
    return (training_data, validation_data, testing_data)

```

## Running Tests

To run tests, we'll create another file that will import both the neural
network we created earlier (`simple_nn`) and the MNIST data set loader
(`mnist_loader`). All we have to do in this file is load the data, create
a `Network` which has an input layer of size 784 and an output layer of size
10, and run the network's `SGD` function on the training data and test with
the test data. Note that it doesn't matter what any of the values in between
784 and 10 are for our list of input layers. Only the input size and output
size are set, we can adjust the rest of the layers however we like. We don't
need 3 layers, we could also have 4 or 5, or even just 2. Play around with
it and have fun.

**neural network code in python - testing**

```python
import simple_nn
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_together()
 
net = simple_nn.Network([784, 30, 10])
net.SGD(training_data, 10, 10, 3.0, test_data=test_data)

```

When we run our test, we should see something like the following image:

![](https://lh3.googleusercontent.com/sO1rtGKJxBBaHjDfDiUvvpGAz9zue-hOUV11W9t0GpFKxiRdBsJEba6HKrLwi8mpXgpnazeSRzvDCR4ybDL-DdyuRbDdYcTKN-yboYpq6lTtY3WxWcC5-rhXK42nJDddUKzuKxC4)

> Output from Python Neural Network from Scratch
