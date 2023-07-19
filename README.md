## Coding ML and DL algorithms from Scratch.

## ML Model ##
Stochastic Gradient Descent and Mini-Batch Gradient Descent:
In this task, you will implement stochastic gradient descent (SGD) and mini-batch gradient descent from scratch in Python.

Stochastic Gradient Descent (SGD): SGD updates the model's parameters using the gradient of the loss computed on a single randomly selected training example at each iteration. It is computationally efficient but introduces more variance in the parameter updates compared to batch gradient descent. You will implement the SGD algorithm, including the random selection of training examples and the update step for the model's parameters.

Mini-Batch Gradient Descent: Mini-batch gradient descent is a compromise between batch gradient descent and stochastic gradient descent. It updates the model's parameters using a small batch of randomly selected training examples at each iteration. Mini-batch gradient descent reduces the variance of parameter updates and can leverage hardware optimizations for matrix operations. You will implement the mini-batch gradient descent algorithm, including the random selection of training examples, batching the examples, and the update step for the model's parameters.

Gradient Descent with Momentum from Scratch in Python: Comparison with PyTorch Implementation:
In this task, you will implement gradient descent with momentum from scratch in Python. Gradient descent with momentum is an optimization algorithm that accelerates convergence and helps overcome local minima by accumulating a velocity term that determines the direction and magnitude of parameter updates. You will implement the gradient descent with momentum algorithm, including the velocity term and the update step for the model's parameters. Additionally, you will compare the performance and convergence speed of your custom implementation with the equivalent implementation using PyTorch.

Gradient Descent with Nesterov Accelerated Gradient Descent:
In this task, you will implement gradient descent with Nesterov accelerated gradient descent (NAG) from scratch in Python. Nesterov accelerated gradient descent is an extension of gradient descent with momentum that further improves convergence by incorporating a lookahead step. The lookahead step adjusts the velocity term before computing the gradients, allowing the algorithm to better approximate the optimal direction for parameter updates. You will implement the Nesterov accelerated gradient descent algorithm, including the lookahead step, the velocity term, and the update step for the model's parameters.

By completing these tasks, you will gain a comprehensive understanding of various gradient descent optimization algorithms and their implementations both from scratch and using PyTorch. This will showcase your knowledge of optimization techniques and provide a comparison between custom implementations and a widely-used deep learning framework like PyTorch.






## Artificial Neural Networks ##

Implemented a single layer neural network for regression. Write backpropagation from scratch for this case in order to work for an arbitrary number of neurons in the hidden layers:
This task involves implementing a single-layer neural network for regression. The neural network should be able to handle an arbitrary number of neurons in the hidden layer. Additionally, you need to write the backpropagation algorithm from scratch, which calculates the gradients of the network's weights and biases with respect to a loss function. This algorithm enables the network to learn from the training data and update its parameters to minimize the loss.

Hyperparameter optimization to determine the appropriate number of neurons:
Hyperparameter optimization is the process of finding the best hyperparameters for a machine learning model. In this task, you need to perform hyperparameter optimization to determine the appropriate number of neurons for the hidden layer in your single-layer neural network. This can be achieved by exploring different configurations of neuron numbers and evaluating the performance of the network using a suitable metric (e.g., mean squared error for regression). Techniques like grid search or random search can be used for this purpose.

PyTorch implementation of the ANN for the same task:
PyTorch is a popular deep learning framework that provides efficient tools for building and training neural networks. In this task, you need to implement the same single-layer neural network for regression using PyTorch. You should utilize the PyTorch library to define the network architecture, initialize the weights and biases, perform the forward and backward passes, and update the parameters using an optimization algorithm (e.g., gradient descent). This implementation should closely resemble the one you developed from scratch but leverage the benefits and convenience of PyTorch.

These tasks involve building a regression model, optimizing hyperparameters, and implementing the model using both a custom approach and the PyTorch library. By completing these tasks and providing the implementations, you will showcase your understanding of neural networks, backpropagation, hyperparameter optimization, and PyTorch's capabilities.

