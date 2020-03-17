import numpy as np

class Multiply():
    def __init__(self):
        """ 逆伝播計算に必要な変数:forwardの入力値 """
        self.x = None
        self.y = None

    def forward(self, x, y):
        """ 順伝播計算:z = x * y """
        self.x = x
        self.y = y
        z=x*y
        return z
    def backprop(self, dz):
        """ 逆伝播計算: dz/dx = y, dz/dy = x """
        dx = dz * self.y
        dy = dz * self.x
        return dx, dy


class Add():
    def __init__(self):
        """ 逆伝播計算に必要な変数:forwardの入力値 """
        self.x = None
        self.y = None

    def forward(self, x, y):
        """ 順伝播計算:z = x * y """
        self.x = x
        self.y = y
        z=x+y
        return z

    def backprop(self, dz):
        """ 逆伝播計算: dz/dx = 1, dz/dy =1 """
        dx = dz
        dy = dz
        return dx, dy

class ReLU():
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        self.x = x
        self.y = np.maximum(x, 0)
        return self.y

    def backprop(self, dz):
        return dz * np.clip(self.y, 0, 1)

class Sigmoid():
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-1 * x))
        return self.y

    def backprop(self, dz):
        dx = dz * (self.y *(1 - self.y))
        return dx

class Affine():
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.y = None

    def forward(self, x):
        self.x = x
        self.y = np.dot(x, self.W) + self.b
        return self.y

    def backprop(self, dz):
        dx = np.dot(dz, self.W.T)
        dW = np.dot(self.x.T, dz)
        db = np.sum(dz, axis=0)

        return dx, dW, db

def softmax(x):
    for i in range(x.shape[0]):
        x[i] -= np.max(x[i])
        x[i] = np.exp(x[i]) / np.sum(np.exp(x[i]))
    return x

def cross_entropy(x, t):
    n, k = x.shape
    y = softmax(x)
    l = np.sum(np.log(y + 1e-8) * (-t)) / n
    return y, l


class SoftmaxCrossEntryopy():
    def __init__(self):
        self.x = None
        self.y = None
        self.t = None
        self.L = None

    def forward(self, x, t):
        self.x = x
        self.t = t
        self.y, self.L = cross_entropy(x, t)
        return self.L

    def backprop(self):
        dl = self.y - self.t
        return dl


def task3_1():
    a, b, c = 2, 3, 4
    add_layer = Add()
    mul_layer = Multiply()

    print("順伝播出力:", mul_layer.forward(add_layer.forward(a, b), c))

    dz, dc = mul_layer.backprop(1)
    da, db = add_layer.backprop(dz)

    print("逆伝播出力 da: {}, db: {}, dc: {}".format(
        da, db, dc
    ))

def task3_2():
    x = np.array([-0.5, 0.0, 1.0, 2.0])
    relu_layer = ReLU()
    print("順伝播:", relu_layer.forward(x))
    print("逆伝播:", relu_layer.backprop(1))

def task3_3():
    x = np.array([-0.5, 0.0, 1.0, 2.0])
    sigmoid_layer = Sigmoid()
    print("順伝播:", sigmoid_layer.forward(x))
    print("逆伝播:", sigmoid_layer.backprop(1))

def task3_4():
    x = np.array([[1.0, 0.5], [-0.4, 0.1]])
    W = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    b = np.array([0.1, 0.2, 0.3])

    affine_layer = Affine(W, b)
    y = affine_layer.forward(x)

    dz = np.array([[1, 1, 1], [1, 1, 1]])
    dx, dW, db = affine_layer.backprop(dz)

    print("順伝播出力:")
    print(y)
    print("逆伝播出力dx:")
    print(dx)
    print("逆伝播出力dw:")
    print(dW)
    print("逆伝播出力db:")
    print(db)

def task3_5():
    x = np.array([[1.0, 0.5], [-0.4, 0.1]])
    t = np.array([[1.0, 0.0], [0.0, 1.0]])

    output_layer = SoftmaxCrossEntryopy()
    l = output_layer.forward(x, t)
    dl = output_layer.backprop()
    print("順伝播出力:")
    print(l)
    print("逆伝播出力:")
    print(dl)



if __name__ == '__main__':
    # task3_1()
    # 順伝播出力: 20
    # 逆伝播出力 da: 4, db: 4, dc: 5

    # task3_2()
    # 順伝播: [0. 0. 1. 2.]
    # 逆伝播: [0. 0. 1. 1.]

    # task3_3()
    # 順伝播: [0.37754067 0.5        0.73105858 0.88079708]
    # 逆伝播: [0.23500371 0.25       0.19661193 0.10499359]

    # task3_4()
    # 順伝播出力:
    # [[0.3  0.7  1.1 ]
    # [0.08 0.12 0.16]]
    # 逆伝播出力dx:
    # [[0.9 1.2]
    # [0.9 1.2]]
    # 逆伝播出力dw:
    # [[0.6 0.6 0.6]
    # [0.6 0.6 0.6]]
    # 逆伝播出力db:
    # [2 2 2]

    task3_5()
    # 順伝播出力:
    0.47407698418010663
    # 逆伝播出力:
    # [[-0.37754067 0.37754067]
    # [ 0.37754067 -0.37754067]]
