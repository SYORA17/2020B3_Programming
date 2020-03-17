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

def task1_2():
    x1_list = [1, 1, 0, 0]
    x2_list = [1, 0, 1, 0]

    and_gate = Perceptron(0.5, 0.5, 0.7)
    nand_gate = Perceptron(-0.5, -0.5, -0.7)
    or_gate  = Perceptron(0.5, 0.5, 0.3)

    for i in range(4):
        x1 = x1_list[i]
        x2 = x2_list[i]

        print('AND({0}, {1}) = {2}, NAND({0}, {1}) = {3}, OR({0}, {1}) = {4}'.format(
            x1, x2,
            and_gate.forward(x1, x2),
            nand_gate.forward(x1, x2),
            or_gate.forward(x1, x2)
        ))

def xor(x1, x2):
    and_gate = Perceptron(0.5, 0.5, 0.7)
    nand_gate = Perceptron(-0.5, -0.5, -0.7)
    or_gate  = Perceptron(0.5, 0.5, 0.3)

    tmp_1 = nand_gate.forward(x1, x2)
    tmp_2 = or_gate.forward(x1, x2)

    return and_gate.forward(tmp_1, tmp_2)

def task1_3():
    x1_list = [1, 1, 0, 0]
    x2_list = [1, 0, 1, 0]

    for i in range(4):
        x1 = x1_list[i]
        x2 = x2_list[i]

        print('XOR({0}, {1}) = {2}'.format(
            x1, x2,
            xor(x1, x2)
        ))


if __name__ == '__main__':
    # task1_2()

    # AND(1, 1) = 1, NAND(1, 1) = 0, OR(1, 1) = 1
    # AND(1, 0) = 0, NAND(1, 0) = 1, OR(1, 0) = 1
    # AND(0, 1) = 0, NAND(0, 1) = 1, OR(0, 1) = 1
    # AND(0, 0) = 0, NAND(0, 0) = 1, OR(0, 0) = 0

    task1_3()

    # XOR(1, 1) = 0
    # XOR(1, 0) = 1
    # XOR(0, 1) = 1
    # XOR(0, 0) = 0
