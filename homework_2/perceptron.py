import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    train_model(BATCH=True)


# TODO
# 对模型的训练加上一个计时函数, 分析不同学习率下的收敛速度
def train_model(pos_data=None, neg_data=None, w=None, b=None,
                learning_rate=None, BATCH=False):
    if pos_data is None:
        pos_data, neg_data = generate_data()
    train_data = generate_train_data(pos_data, neg_data)
    plt.plot(pos_data['x'], pos_data['y'])
    plt.plot(neg_data['x'], neg_data['y'])
    if w is None:
        w, b, learning_rate = initial_parameter()

    if BATCH:
        w, b = train_perceptron_in_batch(train_data, b, learning_rate)
    else:
        w, b = train_perceptron(train_data, w, b, learning_rate)
    plot_data(pos_data, neg_data, w, b)

    # w1, b1 = train_perceptron_in_batch(train_data, b, learning_rate)  # w2,
    #  b2 = train_perceptron(train_data, w, b, learning_rate)  #
    # plot_data_for_comparison([w1, w2], [b1, b2])


def generate_data():
    x_pos = np.random.uniform(2, 10, 50)
    y_pos = np.random.uniform(2, 10, 50)
    pos_labels = [1] * 50
    pos_data = pd.DataFrame({"x": x_pos, "y": y_pos, "label": pos_labels})
    x_neg = list(np.random.uniform(-6, 0, 50))
    y_neg = list(np.random.uniform(-2, 2, 50))
    neg_labels = [-1] * 50
    neg_data = pd.DataFrame({"x": x_neg, "y": y_neg, "label": neg_labels})
    return pos_data, neg_data


def generate_train_data(pos_data, neg_data):
    train_data = pd.concat([pos_data, neg_data])
    # train_data = train_data.sample(frac=1)  # 对数据行进行随机排列
    train_data = np.array(train_data)  # print(train_data)
    return train_data


def initial_parameter(dim=2):
    w = np.array([0] * dim)
    b = 0
    learning_rate = 0.3
    return w, b, learning_rate


def plot_data(pos_data, neg_data, w, b):
    x_line = np.linspace(-6, 10, 20)
    y_line = x_line * (-w[0] / w[1]) - b / w[1]
    plt.plot(x_line, y_line)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-6, 10)
    plt.ylim(-6, 10)
    plt.plot(pos_data['x'], pos_data['y'])
    plt.plot(neg_data['x'], neg_data['y'])
    plt.show()


def plot_data_for_comparison(w, b):
    x_line = np.linspace(-6, 10, 20)
    y_line = x_line * (-w[0][0] / w[0][1]) - b[0] / w[0][1]
    plt.plot(x_line, y_line)
    y_line = x_line * (-w[1][0] / w[1][1]) - b[1] / w[1][1]
    plt.plot(x_line, y_line)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-6, 10)
    plt.ylim(-6, 10)
    plt.show()


def train_perceptron(train_data, w, b, learning_rate):
    EPOCHS = 1000
    cnt = 0
    while cnt < EPOCHS:
        not_modified = True
        for x in train_data:
            label = x[0] * (np.dot(w.T, x[1:]) + b)
            if label <= 0:
                b += learning_rate * x[0]
                w = np.add(w, learning_rate * x[0] * x[1:])
                not_modified = False
        if not_modified:
            break
        cnt += 1
    print("w : %s and b is :%s" % (w, b))
    return w, b


def train_perceptron_in_batch(train_data, b, learning_rate):
    EPOCHS = 100
    dim = len(train_data)
    cnt = 0
    gram_matrix = []
    a = [0] * dim
    for data1 in train_data:
        for data2 in train_data:
            gram_matrix.append(np.dot(data1[1:].T, data2[1:]))
    gram_matrix = np.array(gram_matrix).reshape((dim, dim))
    while cnt < EPOCHS:
        index = cnt % dim
        cur = train_data[index]
        if cur[0] * (
                np.dot(a * train_data[:, 0], gram_matrix[:, index]) + b) <= 0:
            t = np.dot(a * train_data[:, 0], gram_matrix[:, index])
            # a = np.add(a, learning_rate_matrix)
            a[index] += learning_rate
            b += learning_rate * cur[0]
        cnt += 1
    print(a * train_data[:, 0])
    w = np.dot(a * train_data[:, 0], train_data[:, 1:])
    print("w : %s and b is :%s" % (w, b))
    return w, b


if __name__ == "__main__":
    main()
