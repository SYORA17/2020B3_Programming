import numpy as np
import random

from keras import backend as K
from keras.datasets import mnist
# from dataset.mnist import load_mnist

epochs = 100
batch_size = 100
DATA_NUM = 60000
TEST_NUM = 10000


class Affine():
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.y = None

    def forward(self, x):
        self.x = x
        self.y = np.matmul(x, self.W) + self.b
        return self.y

    def backprop(self, dz):
        dx = np.matmul(dz, self.W.T)
        dW = np.matmul(self.x.T, dz)
        db = np.sum(dz, axis=0)

        return dx, dW, db

def softmax(x):
    for i in range(x.shape[0]):
        x[i] -= np.max(x[i])
        x[i] = np.exp(x[i]) / np.sum(np.exp(x[i]))
    return x

def cross_entropy(x, t):
    y = softmax(x)
    l = np.sum(np.log(y + 1e-8) * (-t)) / x.shape[0]
    return y, l

def cal_acc(y, t):
    cnt = 0
    for i in range(y.shape[0]):
        if (y[i] == t[i]).all():
            cnt += 1
    return (cnt / y.shape[0]) * 100

def to_onehot(t):
    n_labels = len(np.unique(t))
    vec = np.eye(n_labels)[t]
    return vec

def y_to_label(y):
    # y = np.array(y)
    label = np.zeros(y.shape)
    idx = np.argmax(y, axis=1)
    for i, j in enumerate(idx):
        label[i, j] = 1
    return label


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


class NeuralNetwork():
    def __init__(self, input_size=784, hidden_size=500, output_size=10, batch_size=100):

        W_1 = np.random.normal(
            loc   = 0,      # 平均
            scale = 0.1,    # 標準偏差
            size  = (input_size, hidden_size),# 出力配列のサイズ
        )
        b_1 = np.zeros((hidden_size))
        layer_1 = Affine(W_1, b_1)

        W_2 = np.random.normal(
            loc   = 0,      # 平均
            scale = 0.1,    # 標準偏差
            size  = (hidden_size, output_size),# 出力配列のサイズ
        )
        b_2 = np.zeros((output_size))
        layer_2 = Affine(W_2, b_2)

        loss_layer = SoftmaxCrossEntryopy()

        self.x = None
        self.y = None
        self.L = None


        self.h_params = {'input_size': input_size,
                         'hidden_size': hidden_size,
                         'output_size': output_size,
                         'batch_size': batch_size}

        self.params = {'W_1': W_1,
                       'b_1': b_1,
                       'W_2': W_2,
                       'b_2': b_2}

        self.bp = {'dx_1': None,
                   'dW_1': None,
                   'db_1': None,
                   'dx_2': None,
                   'dW_2': None,
                   'db_2': None,
                   'dl'  : None}

        self.layers = {'layer_1': layer_1,
                       'layer_2': layer_2,
                       'loss_layer': loss_layer}

        self.io = {'x': None,
                   'y': None,
                   'L': None}

    def forward(self, x):
        self.x = x
        y = self.layers['layer_1'].forward(x)
        y = self.layers['layer_2'].forward(y)
        self.io['y'] = y
        return y

    def loss(self, x, t):
        self.io['x'] = x
        self.io['L'] = self.layers['loss_layer'].forward(x, t)
        return self.io['L']

    def backprop(self, x, t):
        self.bp['dl'] = self.layers['loss_layer'].backprop()
        self.bp['dx_2'], self.bp['dW_2'], self.bp['db_2'] = \
            self.layers['layer_2'].backprop(self.bp['dl'])
        self.bp['dx_1'], self.bp['dW_1'], self.bp['db_1'] = \
            self.layers['layer_1'].backprop(self.bp['dx_2'])

    def sgd(self, x, t, lr=0.0001):
        self.backprop(x, t)
        self.params['W_1'] -= lr * self.bp['dW_1']
        self.params['b_1'] -= lr * self.bp['db_1']
        self.params['W_2'] -= lr * self.bp['dW_2']
        self.params['b_2'] -= lr * self.bp['db_2']

    def print_shape(self):
        print(self.io['x'].shape) # (100, 10)
        print(self.io['y'].shape) # (100, 10)
        print(self.io['L'].shape) # ()
        print(self.bp['dl'].shape) # (100, 10)
        print(self.bp['dW_1'].shape) # (784, 500)
        print("*"*20)

    # def reset_loss(self):
    #     self.io['L'] = 0


if __name__ == '__main__':
    np.random.seed(1234)

    # load data
    # mnist = load_mnist()
    # train_x = mnist["train_img"]
    # train_t = mnist["train_label"]
    # test_x = mnist["test_img"]
    # test_t = mnist["test_label"]

    (train_x, train_t), (test_x, test_t) = mnist.load_data()
    # normarization
    train_x = train_x.reshape(60000, -1).astype('float32')
    train_x /= 255.
    test_x = test_x.reshape(10000, -1).astype('float32')
    test_x /= 255.

    train_t = to_onehot(train_t)
    test_t = to_onehot(test_t)

    print("loaded data!")

    # define model
    model = NeuralNetwork()

    # learning phase
    for epoch in range(epochs):

        train_loss = 0
        train_acc = 0

        idxes = [i for i in range(DATA_NUM)]
        random.shuffle(idxes)

        for i in range(int(DATA_NUM / batch_size)):
            idx = idxes[i*batch_size:(i+1)*batch_size]
            x, t = train_x[idx], train_t[idx]

            y = model.forward(x)
            loss = model.loss(y, t)
            model.sgd(x, t)

            y = y_to_label(y)
            train_acc += cal_acc(y, t)
            train_loss += loss

        train_acc /= (DATA_NUM / batch_size)
        train_loss /= (DATA_NUM / batch_size)


        print("epoch {} / {} :Loss: {}, acc: {} %".format(
            epoch+1, epochs,
            train_loss,
            train_acc
        ))

    '''
    test
    '''
    for i in range(int(TEST_NUM / batch_size)):
        idx = idxes[i*batch_size:(i+1)*batch_size]
        x, t = train_x[idx], train_t[idx]
        y = model.forward(x)
        loss = model.loss(y, t)

        acc_tmp += cal_acc(y, t)
        test_loss += loss

    print("Test")
    print("Loss : {}, acc :{}".format(
        test_loss / (TEST_NUM / batch_size),
        acc_tmp / (TEST_NUM / batch_size)
    ))
