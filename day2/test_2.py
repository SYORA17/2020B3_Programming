import numpy as np

def y_to_label(y):
    # y = np.array(y)
    label = np.zeros(y.shape)
    print(label)
    idx = np.argmax(y, axis=1)
    print(idx)
    for i, j in enumerate(idx):
        print(i, j)
        label[i, j] = 1
    return label

y = np.array([[2.19403821e-03, 9.00188963e-01, 2.85333600e-02, 1.37824177e-02
  ,1.07475698e-03, 6.14223559e-03,5.92765231e-04, 1.79361977e-03
  ,4.42433494e-02, 1.45449449e-03],
 [3.24456251e-04, 2.18322448e-03, 5.12492081e-03, 2.66511068e-01,
  1.60870590e-01, 1.01780177e-01, 3.59767447e-03, 9.88572845e-03,
  1.56370300e-01, 2.93351862e-01]])

tmp = y_to_label(y)
print(tmp)
