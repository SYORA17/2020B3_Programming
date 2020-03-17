import numpy as np

def softmax(x):
    for i in range(x.shape[0]):
        x[i] -= np.max(x[i])
        x[i] = np.exp(x[i]) / np.sum(np.exp(x[i]))
    return x

def cross_entropy(x, t):
    n, k = x.shape
    y = softmax(x)
    l = -t * np.sum(np.log(y + 1e-8)) / n

x = np.array([[1.0, 0.5], [-0.4, 0.1]])
t = np.array([[1.0, 0.0], [0.0, 1.0]])

y = softmax(x)
print(y)

tmp = np.sum(np.log(y + 1e-8) * t * -1) / t.shape[0]
print(tmp)
