import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

np.random.seed(0)
inputs = np.random.randn(100, 3)
inputs[:, -1] = -1
desired_outputs = np.random.choice([1, -1], 100)

perceptron = Perceptron()
perceptron.train(inputs, desired_outputs)

weights_evolution = np.array(perceptron.weights_evolution)
errors_evolution = np.array(perceptron.errors_evolution)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(weights_evolution)
plt.title("Evolução dos Pesos")
plt.xlabel("Iterações")
plt.ylabel("Valor dos Pesos")
plt.legend(['Peso 1', 'Peso 2', 'Bias'])

plt.subplot(1, 2, 2)
plt.plot(errors_evolution)
plt.title("Evolução do Erro")
plt.xlabel("Iterações")
plt.ylabel("Erro Total")
plt.yscale('log')

plt.tight_layout()
plt.show()
