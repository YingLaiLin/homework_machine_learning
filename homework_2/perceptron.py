import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    train_model(BATCH=True)


def train_model(pos_data=None, neg_data=None, w=None, b=None,
                learning_rate=None, BATCH=False):
    if pos_data is None:
        pos_data, neg_data = generate_data()
    train_data = generate_train_data(pos_data, neg_data)
    if w is None:
        w, b, learning_rate = initial_parameter()
    if BATCH:
        w, b = train_perceptron_in_batch(train_data, b, learning_rate)
    else:
        w, b = train_perceptron(train_data, w, b, learning_rate)
    plot_data(pos_data, neg_data, w, b)


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
    train_data = train_data.sample(frac=1)  # 对数据行进行随机排列
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


def train_perceptron(train_data, w, b, learning_rate):
    EPOCHES = 1000
    cnt = 0
    while cnt < EPOCHES:
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


def train_perceptron_in_batch(train_data,  b, learning_rate):
    EPOCHES = 100
    cnt = 0
    gram_matrix = []
    a = [0] * 100
    learning_rate_matrix = np.array([learning_rate] * 100).T
    for data1 in train_data:
        for data2 in train_data:
            gram_matrix.append(np.dot(data1[1:].T, data2[1:]))
    gram_matrix = np.array(gram_matrix).reshape((100, 100))
    while cnt < EPOCHES:
        not_modified = True
        cur = train_data[cnt]
        if cur[0] * \
                (np.dot(a * train_data[:, 0],gram_matrix[:, cnt])) <= 0:
            a = np.add(a, learning_rate_matrix)
            b += learning_rate * cur[0]

        if not_modified:
            break
        cnt += 1
    w = a * train_data[:, 0] * train_data[:, 1]
    print("w : %s and b is :%s" % (w, b))
    return w, b


def predict(w, b, x):
    if np.dot(w.T, x) + b > 0:
        return 1
    return 0


def get_mul(v1, v2):
    return v1.x * v2.x + v1.y * v2.y


if __name__ == "__main__":
    main()
