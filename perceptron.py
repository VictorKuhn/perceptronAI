import numpy as np
from utils import perceptron_function

class Perceptron:
    def __init__(self, learning_rate=0.5, max_iterations=1000, desired_error=0.01):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.desired_error = desired_error
        self.weights = None
        self.weights_evolution = []
        self.errors_evolution = []

    def train(self, inputs, desired_outputs):
        self.weights = np.random.rand(inputs.shape[1])
        for iteration in range(self.max_iterations):
            total_error = 0
            for input_vector, desired_output in zip(inputs, desired_outputs):
                output = self.predict(input_vector)
                error = desired_output - output
                self.weights += self.learning_rate * error * input_vector
                total_error += error ** 2
                self.weights_evolution.append(self.weights.copy())
                self.errors_evolution.append(total_error)
            if total_error <= self.desired_error:
                break

    def predict(self, input_vector):
        u = np.dot(input_vector, self.weights)
        return perceptron_function(u)
