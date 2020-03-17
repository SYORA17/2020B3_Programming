import numpy as np

class Perceptron():
    def __init__(self, w1, w2, theta):
        self.w1 = w1
        self.w2 = w2
        self.theta = theta

    def forward(self, x1, x2):
        tmp = self.w1 * x1 + self.w2 * x2
        if tmp >= self.theta:
            y = 1
        else:
            y = 0
        return y

class SingleLayer():
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        return np.dot(self.W.T, x) + self.b

class Layer():
    def __init__(self, W, b):
        self.W = W
        self.b = b
    def forward(self, x):
        y = np.dot(self.W.T, x.T)
        b = np.tile(self.b, (y.shape[1], 1)).T
        y += b
        return y.T

class MLP_3Layer():
    def __init__(self):
        self.W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        self.b1 = np.array([0.1, 0.2, 0.3])
        self.W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        self.b2 = np.array([0.1, 0.2])
        self.W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
        self.b3 = np.array([0.1, 0.2])

        self.Layer_1 = Layer(self.W1, self.b1)
        self.Layer_2 = Layer(self.W2, self.b2)
        self.Layer_3 = Layer(self.W3, self.b3)


    def forawrd(self, x):
        y = self.Layer_1.forward(x)
        y = self.Layer_2.forward(y)
        y = self.Layer_3.forward(y)

        # for task2_5
        y = softmax(y)

        return y





def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    for i in range(x.shape[0]):
        x[i] -= np.max(x[i])
        x[i] = np.exp(x[i]) / np.sum(np.exp(x[i]))
    return x

def task2_1(x):
    print(sigmoid(x))

def task2_2(x):
    print(relu(x))

def task2_3():
    x = np.array([1.0, 0.5])
    W = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    b = np.array([0.1, 0.2, 0.3])

    Layer_1 = SingleLayer(W, b)
    print(Layer_1.forward(x))

def task2_4():
    # 各層のユニット数は[2, 3, 2, 2]、バッチサイズは4

    x = np.array([[1.0, 0.5], [-0.3, -0.2], [0.0, 0.8], [0.3, -0.4]])
    MLP_task2_4 = MLP_3Layer()
    y = MLP_task2_4.forawrd(x)
    print(y)

def task2_5():
    x = np.array([[1.0, 0.5], [-0.3, -0.2], [0.0, 0.8], [0.3, -0.4]])
    MLP_task2_4 = MLP_3Layer()
    y = MLP_task2_4.forawrd(x)
    print(y)



if __name__ == '__main__':
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    # task2_1()
    # [0.26894142 0.5        0.62245933 0.88079708]

    # task2_2(x)
    # [0.  0.  0.5 2. ]

    # task2_3()
    # [0.3 0.7 1.1]

    # task2_4()
    # [[0.426  0.912 ]
    # [0.1608 0.3334]
    # [0.3528 0.752 ]
    # [0.2016 0.4226]]

    task2_5()
    # [[0.38083632 0.61916368]
    # [0.4569568  0.5430432 ]
    # [0.40150456 0.59849544]
    # [0.44497378 0.55502622]]
